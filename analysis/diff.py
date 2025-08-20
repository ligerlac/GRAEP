from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================
import glob
import hashlib
import logging
import os
import pickle
import warnings
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# =============================================================================
# Third-Party Imports
# =============================================================================
import awkward as ak
import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import uproot
import vector
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from jaxopt import OptaxSolver
from tabulate import tabulate

# =============================================================================
# Project Imports
# =============================================================================
from analysis.base import Analysis
from user.cuts import lumi_mask
# from utils.jax_stats import build_channel_data_scalar, compute_discovery_pvalue
from utils.evm_stats import build_channel_data_scalar, compute_discovery_pvalue, fit_params
from utils.logging import BLUE, GREEN, RED, RESET, _banner
from utils.mva import JAXNetwork, TFNetwork
from utils.plot import (
    create_cms_histogram,
    plot_mva_feature_distributions,
    plot_mva_scores,
    plot_parameters_over_iterations,
    plot_pvalue_vs_parameters,
)
from utils.preproc import pre_process_dak, pre_process_uproot
from utils.tools import nested_defaultdict_to_dict, recursive_to_backend


# =============================================================================
# Backend & Logging Setup
# =============================================================================
ak.jax.register_and_check()
vector.register_awkward()

logger = logging.getLogger("DiffAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def merge_histograms(
    existing_histograms: dict[str, dict[str, dict[str, jnp.ndarray]]],
    new_histograms: dict[str, dict[str, dict[str, jnp.ndarray]]],
) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
    """
    Recursively merge new histograms into an existing nested histogram structure.

    Each histogram is assumed to follow the structure:
    histograms[variation][region][observable] = jnp.ndarray

    Parameters
    ----------
    existing_histograms : dict
        Existing nested histogram structure to be updated.
    new_histograms : dict
        New histogram data to merge in.

    Returns
    -------
    dict
        The updated nested histogram structure with new data merged in.
    """
    for variation, region_data in new_histograms.items():
        for region, observable_data in region_data.items():
            for observable, array in observable_data.items():
                # Ensure region exists under the current variation
                existing_histograms[variation].setdefault(region, {})
                # Merge or initialise the observable histogram
                if observable in existing_histograms[variation][region]:
                    existing_histograms[variation][region][observable] += array
                else:
                    existing_histograms[variation][region][observable] = array
    return existing_histograms


def infer_processes_and_systematics(
    processed_datasets: dict[str, list[tuple[Any, dict[str, Any]]]],
    systematics_config: list[dict[str, Any]],
    corrections_config: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Extract all unique process and systematic names from the config and processed datasets.

    Parameters
    ----------
    processed_datasets : dict
        Dictionary mapping dataset names to lists of (events, metadata) tuples.
    systematics_config : list
        Configuration entries for systematic variations.
    corrections_config : list
        Configuration entries for object-level corrections.

    Returns
    -------
    tuple[list[str], list[str]]
        Sorted list of process names and systematic variation base names.
    """
    # Pull out all process names from the processed datasets metadata
    process_names = {
        metadata.get("process")
        for events_list in processed_datasets.values()
        for events, metadata in events_list
        if metadata.get("process")
    }

    # Extract systematic names from both systematics and corrections configs
    systematic_names = {
        syst["name"] for syst in systematics_config + corrections_config
    }

    return sorted(process_names), sorted(systematic_names)


def _log_parameter_update(
    step: Union[int, str],
    old_p_value: float,
    new_p_value: float,
    old_params: Dict[str, Dict[str, Any]],
    new_params: Dict[str, Dict[str, Any]],
    initial_params: Optional[Dict[str, Dict[str, Any]]] = None,
    initial_pval: Optional[float] = None,
    mva_configs: Optional[List[Any]] = None,
) -> None:
    """
    Log a formatted and coloured table of parameter updates for a single
    optimisation step.

    This function creates two tables:
    1. A p-value comparison table showing old vs new p-values and percentage change
    2. A parameter table showing changes for all auxiliary and fit parameters

    The tables use colour coding:
    - Green: p-value improvement (decrease)
    - Red: p-value worsening (increase)
    - Blue: parameter values that have changed

    Neural network parameters are only logged if the corresponding MVA configuration
    has `grad_optimisation.log_param_changes` enabled, and are displayed as mean values.

    Parameters
    ----------
    step : Union[int, str]
        The current optimisation step number or a descriptive string (e.g., "Initial").
        If int, displays as "STEP XXX". If str, displays as "STATE: XXX" or just the
            string.
    old_p_value : float
        The p-value from the last logged step for comparison.
    new_p_value : float
        The p-value at the current step.
    old_params : Dict[str, Dict[str, Any]]
        Dictionary with 'aux' and 'fit' keys containing parameters from previous state.
        Structure: {"aux": {param_name: value, ...}, "fit": {param_name: value, ...}}
    new_params : Dict[str, Dict[str, Any]]
        Dictionary with 'aux' and 'fit' keys containing updated parameters.
        Same structure as old_params.
    initial_params : Dict[str, Dict[str, Any]]
        Dictionary with 'aux' and 'fit' keys containing initial parameters
        from the start
        of optimisation. Used to calculate percentage change from initial state.
        Same structure as old_params.
    initial_pval: float
        The initial p-value at the start of optimisation, used for percentage change
            from initial state.
    mva_configs : Optional[List[Any]]
        List of MVA configuration objects with 'name' and
        'grad_optimisation.log_param_changes'
        attributes. Used to determine which neural network parameters to log.

    Returns
    -------
    None
        Logs formatted tables to the logger but returns nothing.

    Notes
    -----
    - Parameters starting with "__NN" are treated as neural network parameters
    - Percentage changes are calculated as ((new - old) / old) * 100 for last step
        change
    - Percentage changes from initial are calculated as
        ((new - initial) / initial) * 100
    - If old/initial value is 0, percentage change is set to infinity for non-zero
        new values
    - Colour formatting uses ANSI escape codes defined in utils.logging
    """
    # --- P-value Table ---
    p_value_headers = ["Old p-value", "New p-value", "% Change"]
    if old_p_value and old_p_value != 0:
        p_value_change = ((new_p_value - old_p_value) / old_p_value) * 100
    else:
        p_value_change = float("inf") if new_p_value != 0 else 0.0

    p_value_row = [
        f"{old_p_value:.4f}",
        f"{new_p_value:.4f}",
        f"{p_value_change:+.2f}%",
    ]

    if initial_pval:
        p_value_headers.append("% Change from Initial")
        if initial_pval != 0:
            p_value_change_initial = (
                (new_p_value - initial_pval) / initial_pval
            ) * 100
        else:
            if new_p_value != 0:
                p_value_change_initial = float("inf")
            else:
                p_value_change_initial = 0.0
        p_value_row.append(f"{p_value_change_initial:+.2f}%")

    # Colour green for improvement (decrease), red for worsening (increase)
    if new_p_value < old_p_value:
        p_value_row_coloured = [
            f"{GREEN}{item}{RESET}" for item in p_value_row
        ]
    elif new_p_value > old_p_value:
        p_value_row_coloured = [f"{RED}{item}{RESET}" for item in p_value_row]
    else:
        p_value_row_coloured = p_value_row
    p_value_table = tabulate(
        [p_value_row_coloured], headers=p_value_headers, tablefmt="grid"
    )

    # --- Parameter Table ---
    table_data = []
    headers = ["Parameter", "Old Value", "New Value", "% Change"]

    # Create a map from MVA name to its config for easy lookup
    mva_config_map = {mva.name: mva for mva in mva_configs or []}

    # Combine all parameters for easier iteration
    all_old_params = {**old_params.get("aux", {}), **old_params.get("fit", {})}
    all_new_params = {**new_params.get("aux", {}), **new_params.get("fit", {})}
    all_initial_params = (
        {**initial_params.get("aux", {}), **initial_params.get("fit", {})}
        if initial_params
        else {}
    )

    for name, old_val in sorted(all_old_params.items()):
        new_val = all_new_params[name]
        initial_val = all_initial_params.get(name, 0.0)

        # Handle NN parameters
        if name.startswith("__NN"):
            mva_name = next(
                (mva.name for mva in (mva_configs or []) if mva.name in name),
                None,
            )

            # Check if logging is enabled for this MVA
            if (
                mva_name
                and mva_config_map[
                    mva_name
                ].grad_optimisation.log_param_changes
            ):
                old_val_display, new_val_display, initial_val_display = (
                    jnp.mean(old_val),
                    jnp.mean(new_val),
                    jnp.mean(initial_val),
                )
                name_display = f"{name} (mean)"
            else:
                continue  # Skip NN params if logging is not enabled
        else:
            old_val_display, new_val_display, initial_val_display = (
                old_val,
                new_val,
                initial_val,
            )
            name_display = name

        # Calculate percentage change from last step
        old_param, new_param, initial_param = jax.device_get(
            [old_val_display, new_val_display, initial_val_display]
        )
        if old_param != 0:
            percent_change = ((new_param - old_param) / old_param) * 100
        else:
            percent_change = float("inf") if new_param != 0 else 0.0

        # Format for table
        row = [
            name_display,
            f"{old_param:.4f}",
            f"{new_param:.4f}",
            f"{percent_change:+.2f}%",
        ]

        # Calculate percentage change from initial
        if initial_param is None:
            headers.append("% Change from Initial")
            if initial_param != 0:
                percent_change_from_initial = (
                    (new_param - initial_param) / initial_param
                ) * 100
            else:
                percent_change_from_initial = (
                    float("inf") if new_param != 0 else 0.0
                )

            row.append(f"{percent_change_from_initial:+.2f}%")

        # Colour the row blue if the parameter value has changed
        if not np.allclose(old_param, new_param, atol=1e-6, rtol=1e-5):
            row = [f"{BLUE}{item}{RESET}" for item in row]

        table_data.append(row)

    if isinstance(step, int):
        header = f"STEP {step:3d}"
    else:
        if step != "":
            header = f"STATE: {step}"
        else:
            header = ""

    if not table_data:
        logger.info(f"\n{header}\n{p_value_table}\n(No parameters to log)")
        return

    table_str = tabulate(table_data, headers=headers, tablefmt="grid")
    logger.info(f"\n{header}\n{p_value_table}\n{table_str}\n")


# -----------------------------------------------------------------------------
# Optimisation helper functions
# -----------------------------------------------------------------------------


def make_apply_param_updates(
    parameter_update_rules: Dict[str, callable],
) -> callable:
    """
    Build a function that updates parameters using user-defined rules.

    This function creates a parameter update function that applies custom transformation
    rules to auxiliary parameters during optimisation. Each rule defines how a parameter
    should be updated given its old value and the proposed change (delta).

    Parameters
    ----------
    parameter_update_rules : Dict[str, callable]
        Mapping from parameter name to update function. Each function should have the
        signature: `lambda old_value, delta -> new_value`. For example:
        - Clamping: `lambda old, delta: jnp.clip(old + delta, min_val, max_val)`
        - Exponential: `lambda old, delta: old * jnp.exp(delta)`

    Returns
    -------
    callable
        Function with signature `(old_params: dict, tentative_params: dict) -> dict`
        that applies the update rules to auxiliary parameters and returns updated
        parameter dictionary with structure {"aux": {...}, "fit": {...}}.

    Examples
    --------
    >>> rules = {
    ...     "threshold": lambda old, delta: jnp.clip(old + delta, 0.0, 1.0),
    ...     "scale": lambda old, delta: old * jnp.exp(delta)
    ... }
    >>> update_fn = make_apply_param_updates(rules)
    >>> new_params = update_fn(old_params, tentative_params)
    """

    def apply_rules(old_params: dict, tentative_params: dict) -> dict:
        updated_auxiliary = {}
        aux_old = old_params["aux"]
        aux_new_candidate = tentative_params["aux"]

        # Apply the user-defined update rule for each parameter
        for param_name, candidate_value in aux_new_candidate.items():
            if param_name in parameter_update_rules:
                delta = candidate_value - aux_old[param_name]
                updated_auxiliary[param_name] = parameter_update_rules[
                    param_name
                ](aux_old[param_name], delta)
            else:
                updated_auxiliary[param_name] = candidate_value

        return {"aux": updated_auxiliary, "fit": tentative_params["fit"]}

    return apply_rules


def make_clamp_transform(
    clamp_function: callable,
) -> optax.GradientTransformation:
    """
    Create an Optax transformation that clamps parameters after each update step.

    This function creates a custom Optax gradient transformation that applies parameter
    clamping/projection after the standard gradient update. This is useful for enforcing
    constraints on parameters during optimisation (e.g., keeping probabilities in [0,1]
    or ensuring positive values).

    Parameters
    ----------
    clamp_function : callable
        A function with signature `(old_params: dict, new_params: dict) -> dict` that
        takes the current parameters and tentatively updated parameters, then returns
        the clamped/projected parameters. This function should enforce any constraints
        on the parameter values.

    Returns
    -------
    optax.GradientTransformation
        An Optax transformation object that applies gradient updates followed by
        parameter clamping. The transformation has `init_fn` and `update_fn` methods
        compatible with the Optax interface.

    Notes
    -----
    The transformation works by:
    1. Applying the standard gradient update: new_params = old_params + updates
    2. Clamping the result: clamped_params = clamp_function(old_params, new_params)
    3. Computing effective updates: effective_updates = clamped_params - old_params

    Examples
    --------
    >>> def clamp_positive(old_params, new_params):
    ...     return jax.tree_map(lambda x: jnp.maximum(x, 0.0), new_params)
    >>> clamp_transform = make_clamp_transform(clamp_positive)
    >>> optimiser = optax.chain(optax.adam(0.01), clamp_transform)
    """

    def init_fn(initial_params: dict) -> None:
        """
        Initialise the transformation state.

        Parameters
        ----------
        initial_params : dict
            Initial parameter values (unused in this transformation).

        Returns
        -------
        None
            This transformation does not maintain state.
        """
        return None

    def update_fn(
        parameter_updates: dict,
        state: None,
        current_params: Optional[dict] = None,
    ) -> tuple[dict, None]:
        # Apply standard update
        updated_params = optax.apply_updates(current_params, parameter_updates)
        # Clamp values to allowed region
        clamped_params = clamp_function(current_params, updated_params)
        # Compute difference between clamped and original values
        effective_updates = jax.tree_util.tree_map(
            lambda new, old: new - old, clamped_params, current_params
        )
        return effective_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def make_lr_and_clamp_transform(
    auxiliary_lr_map: Dict[str, float],
    default_auxiliary_lr: float,
    default_fit_lr: float,
    neural_net_lr_map: Dict[str, float],
    clamp_function: callable,
    frozen_parameter_keys: Optional[set[str]] = None,
) -> callable:
    """
    Create an optimiser builder combining learning-rate scheduling and
    parameter clamping.

    This function creates an optimiser that applies different learning rates
    to different parameter groups (auxiliary parameters, neural network parameters, and
    fit parameters) while also applying parameter constraints through clamping.

    The optimiser uses Optax's `multi_transform` to apply different `Adam` optimisers
    with different learning rates to different parameter groups, followed by a custom
    clamping transformation to enforce parameter constraints.

    Parameters
    ----------
    auxiliary_lr_map : Dict[str, float]
        Custom learning rates for specific auxiliary parameters. Keys are parameter
        names, values are learning rates. Parameters not in this map use
        default_auxiliary_lr.
    default_auxiliary_lr : float
        Default learning rate for auxiliary parameters not specified in auxiliary_lr_map
    default_fit_lr : float
        Learning rate for 'fit' parameters (typically statistical model parameters).
    neural_net_lr_map : Dict[str, float]
        Learning rates for neural network parameters. Keys are MVA model names,
        values are learning rates for all parameters of that model.
    clamp_function : callable
        Function with signature `(old_params: dict, new_params: dict) -> dict` that
        enforces parameter constraints by clamping/projecting parameters to
        valid regions.
    frozen_parameter_keys : Optional[set[str]]
        Set of parameter names to freeze (prevent any updates). These parameters will
        use the 'no_update' transform which sets gradients to zero.

    Returns
    -------
    callable
        A builder function with signature
        `(initial_params: dict) -> (optax.GradientTransformation, dict)`.
        The returned function takes initial parameters and returns:
        - An Optax transformation that applies learning rates and clamping
        - A label mapping dictionary showing which transform each parameter uses

    Notes
    -----
    Parameter groups and their transforms:
    - `aux__<param_name>`: Specific auxiliary parameter with custom learning rate
    - `aux__default`: Default auxiliary parameters
    - `NN__<model_name>`: Neural network parameters for specific model
    - `NN__default`: Neural network parameters with default learning rate
    - `fit__default`: Fit parameters (statistical model parameters)
    - `no_update`: Frozen parameters (gradients set to zero)

    Examples
    --------
    >>> aux_lrs = {"threshold": 0.01, "bandwidth": 0.001}
    >>> nn_lrs = {"signal_classifier": 0.0001}
    >>> clamp_fn = lambda old, new: jax.tree_map(lambda x: jnp.clip(x, 0, 1), new)
    >>> optimiser_builder = make_lr_and_clamp_transform(
    ...     aux_lrs, 0.01, 0.001, nn_lrs, clamp_fn, {"frozen_param"}
    ... )
    >>> optimiser, labels = optimiser_builder(initial_params)
    """
    frozen_parameter_keys = frozen_parameter_keys or set()

    # Build learning-rate schedule map for different parameter groups
    sub_transforms = {
        **{
            f"aux__{key}": optax.adam(lr)
            for key, lr in auxiliary_lr_map.items()
        },
        **{
            f"NN__{key}": optax.adam(lr)
            for key, lr in neural_net_lr_map.items()
        },
        "aux__default": optax.adam(default_auxiliary_lr),
        "NN__default": optax.adam(default_auxiliary_lr),
        "fit__default": optax.adam(default_fit_lr),
        "no_update": optax.set_to_zero(),
    }

    def make_label_pytree(
        parameter_tree: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """
        Create a label tree assigning each parameter to a named sub-transform.

        This function analyses the parameter tree structure and assigns each parameter
        to an appropriate optimiser transform based on its name and type. The labels
        are used by Optax's multi_transform to apply different optimisers to different
        parameter groups.

        Parameters
        ----------
        parameter_tree : Dict[str, Dict[str, Any]]
            Nested dictionary of parameters with keys "aux" and "fit".
            Structure: {"aux": {param_name: param_value, ...},
                        "fit": {param_name: param_value, ...}}

        Returns
        -------
        Dict[str, Dict[str, str]]
            A tree structure with the same shape as parameter_tree, but  string labels
            instead of parameter values. Each label corresponds to a sub-transform name
            in the multi_transform dictionary.

        Notes
        -----
        Label assignment rules:
        - Parameters in frozen_parameter_keys -> "no_update"
        - Parameters containing "__NN" -> "NN__<model_name>" or "NN__default"
        - Auxiliary parameters in auxiliary_lr_map -> "aux__<param_name>"
        - Other auxiliary parameters -> "aux__default"
        - Fit parameters -> "fit__default" (unless frozen)
        """
        label_tree = {"aux": {}, "fit": {}}

        for param_name in parameter_tree["aux"]:
            if param_name in frozen_parameter_keys:
                label_tree["aux"][param_name] = "no_update"
            elif "__NN" in param_name:
                # Use specific NN LR if matched, otherwise fall back to default
                label_tree["aux"][param_name] = next(
                    (
                        f"NN__{name}"
                        for name in neural_net_lr_map
                        if name in param_name
                    ),
                    "NN__default",
                )
            elif param_name in auxiliary_lr_map:
                label_tree["aux"][param_name] = f"aux__{param_name}"
            else:
                label_tree["aux"][param_name] = "aux__default"

        for param_name in parameter_tree["fit"]:
            label_tree["fit"][param_name] = (
                "no_update"
                if param_name in frozen_parameter_keys
                else "fit__default"
            )

        return label_tree

    def optimiser_builder(
        initial_params: Dict[str, Dict[str, Any]],
    ) -> Tuple[optax.GradientTransformation, Dict[str, Dict[str, str]]]:
        """
        Build the optimiser with learning rate scheduling and parameter clamping.

        This function creates the final Optax gradient transformation by combining
        the multi-transform (for different learning rates) with the clamping transform.
        It also returns the label mapping for debugging and inspection purposes.

        Parameters
        ----------
        initial_params : Dict[str, Dict[str, Any]]
            Initial parameter values used to determine the label mapping structure.
            Must have the structure: {"aux": {param_name: value, ...},
                                     "fit": {param_name: value, ...}}

        Returns
        -------
        Tuple[optax.GradientTransformation, Dict[str, Dict[str, str]]]
            A tuple containing:
            - An Optax transformation that applies different learning rates to different
              parameter groups and then applies parameter clamping/projection.
            - A label mapping dictionary with the same structure as initial_params,
              showing which sub-transform each parameter is assigned to.

        Notes
        -----
        The optimiser chain applies transformations in this order:
        1. Multi-transform: Applies  Adam optimisers with different learning rates
        2. Clamping transform: Projects parameters to valid regions using clamp_function
        """
        # Map each parameter to its optimiser label
        label_mapping = make_label_pytree(initial_params)
        # Chain the learning rate transform with clamping logic
        optimiser = optax.chain(
            optax.multi_transform(sub_transforms, label_mapping),
            make_clamp_transform(clamp_function),
        )
        return optimiser, label_mapping

    return optimiser_builder


# -----------------------------------------------------------------------------
# DifferentiableAnalysis Class Definition
# -----------------------------------------------------------------------------
class DifferentiableAnalysis(Analysis):
    """
    Differentiable statistical analysis implementation.

    This class extends the base `Analysis` class to support differentiable workflows,
    including:
    - Managing histograms with JAX-compatible data structures.
    - Preparing directory structure for outputs, caches, and plots.
    - Applying selections per analysis channel for both analysis and MVA data branches.
    - Training MVA models using JAX or TensorFlow frameworks.
    """

    def __init__(self, config: dict[str, Any], processed_datasets: Optional[Dict[str, List[Tuple[Any, Dict[str, Any]]]]] = None) -> None:
        """
        Initialise the DifferentiableAnalysis with configuration and processed datasets.

        Parameters
        ----------
        config : dict
            Analysis configuration dictionary.
        processed_datasets : Optional[Dict[str, List[Tuple[Any, Dict[str, Any]]]]], optional
            Pre-processed datasets from skimming, by default None
        """
        super().__init__(config, processed_datasets)

        # Histogram storage:
        # histograms[variation][region][observable] = jnp.ndarray
        # Used for storing the final outputs from the differentiable histogramming step.
        self.histograms: dict[str, dict[str, dict[str, jnp.ndarray]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        self._prepare_dirs()

    def _set_histograms(
        self, histograms: dict[str, dict[str, dict[str, jnp.ndarray]]]
    ) -> None:
        """
        Set final histograms after processing.

        Parameters
        ----------
        histograms : dict
            Histogram dictionary to store.
        """
        # Replace the current histogram store with new results
        self.histograms = histograms

    def _prepare_dirs(self) -> None:
        """
        Create necessary output directories for analysis results.

        This includes:
        - General output directory structure
        - Cache directory for storing gradients
        - Optional preprocessed input directory
        - Directories for storing MVA models, optimisation plots, and fit plots
        """
        # Ensure the output/ directory and common structure are prepared
        self.dirs = super()._prepare_dirs()

        # Create cache directory for gradient checkpoints and intermediate results
        cache = Path(
            self.config.general.cache_dir or "/tmp/gradients_analysis/"
        )
        cache.mkdir(parents=True, exist_ok=True)


        # Directory for trained MVA models
        mva = self.dirs["output"] / "mva_models"
        mva.mkdir(parents=True, exist_ok=True)

        # Output plots from analysis cut optimisation step
        optimisation_plots = self.dirs["output"] / "plots" / "optimisation"
        optimisation_plots.mkdir(parents=True, exist_ok=True)

        # Output plots from fit or profiling stage
        fit_plots = self.dirs["output"] / "plots" / "fit"
        fit_plots.mkdir(parents=True, exist_ok=True)

        # Directory for MVA plots (subdirectories created during plotting)
        mva_plots = self.dirs["output"] / "mva_plots"
        mva_plots.mkdir(parents=True, exist_ok=True)

        # Register the created paths in the analysis directory registry
        self.dirs.update(
            {
                "cache": cache,
                "preproc": Path(preproc) if preproc else None,
                "mva_models": mva,
                "optimisation_plots": optimisation_plots,
                "fit_plots": fit_plots,
                "mva_plots": mva_plots,
            }
        )

    def _log_config_summary(self) -> None:
        """Logs a structured summary of the key analysis configuration options."""
        logger.info(_banner("Differentiable Analysis Configuration Summary"))

        # --- General Settings ---
        general_cfg = self.config.general
        general_data = [
            ["Output Directory", general_cfg.output_dir],
            [
                "Max Files per Sample",
                (
                    "All"
                    if general_cfg.max_files == -1
                    else general_cfg.max_files
                ),
            ],
            ["Run Skimming", general_cfg.run_skimming],
            ["Run MVA Pre-training", general_cfg.run_mva_training],
            ["Run Systematics", general_cfg.run_systematics],
            ["Run Plots Only", general_cfg.run_plots_only],
            ["Read from Cache", general_cfg.read_from_cache],
        ]
        logger.info(
            "General Settings:\n"
            + tabulate(general_data, tablefmt="grid", stralign="left")
        )

        # --- Channels ---
        channel_data = []
        for ch in self.config.channels:
            if ch.use_in_diff:
                channel_data.append([ch.name, ch.fit_observable])
        if channel_data:
            logger.info(
                "Differentiable Channels:\n"
                + tabulate(
                    channel_data,
                    headers=["Name", "Fit Observable"],
                    tablefmt="grid",
                )
            )

        # --- Processes ---
        if self.processed_datasets:
            processes = sorted(list({
                metadata["process"]
                for events_list in self.processed_datasets.values()
                for events, metadata in events_list
            }))
            if self.config.general.processes:
                processes = [p for p in processes if p in self.config.general.processes]
            processes_data = [[p] for p in processes]
            logger.info("Processes Included:\n" + tabulate(processes_data, headers=["Process"], tablefmt="grid"))

        # --- Systematics ---
        if self.config.general.run_systematics:
            syst_data = []
            all_systs = self.config.systematics + self.config.corrections
            for syst in all_systs:
                if syst.name != "nominal":
                    syst_data.append([syst.name, syst.type])
            if syst_data:
                logger.info(
                    "Systematics Enabled:\n"
                    + tabulate(
                        syst_data,
                        headers=["Systematic", "Type"],
                        tablefmt="grid",
                    )
                )

        if not self.config.jax:
            logger.warning("No JAX configuration block found.")
            return

        jax_cfg = self.config.jax
        # --- Optimisation Settings ---
        jax_data = [
            ["Run Optimisation", jax_cfg.optimise],
            ["Max Iterations", jax_cfg.max_iterations],
            ["Explicit Optimisation Loop", jax_cfg.explicit_optimisation],
        ]
        logger.info(
            "Optimisation Settings:\n"
            + tabulate(jax_data, tablefmt="grid", stralign="left")
        )

        # --- Optimisable Parameters ---
        if jax_cfg.params:
            params_data = []
            headers = ["Parameter", "Initial Value", "Clamped"]
            for name, value in jax_cfg.params.items():
                if "__NN" not in name:
                    is_clamped = (
                        "Yes" if name in jax_cfg.param_updates else "No"
                    )
                    params_data.append([name, value, is_clamped])
            if params_data:
                logger.info(
                    "Initial Optimisable Parameters:\n"
                    + tabulate(params_data, headers=headers, tablefmt="grid")
                )

        # --- Learning Rates ---
        lr_data = [["Default", jax_cfg.learning_rate]]
        if jax_cfg.learning_rates:
            for name, value in jax_cfg.learning_rates.items():
                lr_data.append([name, value])
        logger.info(
            "Learning Rates (Auxiliary Parameters):\n"
            + tabulate(
                lr_data,
                headers=["Parameter", "Learning Rate"],
                tablefmt="grid",
            )
        )

        # --- MVA Models ---
        if self.config.mva:
            mva_data = []
            headers = [
                "Name",
                "Framework",
                "# Features",
                "Balance Strategy",
                "Pre-train LR",
                "Optimise parameters",
                "Optimisation LR",
            ]
            for mva in self.config.mva:
                mva_data.append(
                    [
                        mva.name,
                        mva.framework,
                        len(mva.features),
                        mva.balance_strategy,
                        mva.learning_rate,
                        mva.grad_optimisation.optimise,
                        (
                            mva.grad_optimisation.learning_rate
                            if mva.grad_optimisation.optimise
                            else "N/A"
                        ),
                    ]
                )
            logger.info(
                "MVA Models:\n"
                + tabulate(mva_data, headers=headers, tablefmt="grid")
            )

    # -------------------------------------------------------------------------
    # Data categorisation (channels)
    # -------------------------------------------------------------------------
    def _get_channel_data(
        self,
        object_copies_analysis: dict[str, ak.Array],
        events_analysis: ak.Array,
        object_copies_mva: dict[str, ak.Array],
        events_mva: ak.Array,
        process: str,
        variation: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Apply per-channel selection to the corrected object and event collections,
        and return selected data for both physics analysis and MVA pre-training.

        This function performs selection using the analysis branch and returns:
        - selected objects and events for histogramming (from analysis branch)
        - selected objects and events for MVA pre-training (from MVA branch)
        - total number of selected events from each branch
        - unmasked inputs in a '__presel' entry

        See docstring inside `apply_selection` for further details.
        """
        if process == "data" and variation != "nominal":
            logger.debug(f"Skipping {variation} for data")
            return {}

        def apply_selection(
            channel,
            object_copies: dict[str, ak.Array],
            events: ak.Array,
            label: str,
        ) -> tuple[dict[str, ak.Array], ak.Array, ak.Array]:
            """
            Apply selection function to the input objects and events.

            This runs the channel-specific selection function (typically returning a
            PackedSelection object) and constructs the event mask to use.

            For data samples, applies the certified luminosity mask afterwards.

            Parameters
            ----------
            channel : object
                Analysis channel containing selection information.
            object_copies : dict[str, ak.Array]
                Corrected object collections.
            events : ak.Array
                Corresponding events.
            label : str
                Context label for logging.

            Returns
            -------
            tuple
                (Selected objects, events, and mask array)
            """
            # Ensure data is not backed by JAX (selection not traced)
            events = recursive_to_backend(events, "cpu")
            object_copies = recursive_to_backend(object_copies, "cpu")

            # Check for presence of selection logic
            if not channel.selection.function:
                logger.warning(
                    f"[{label}] No selection function for channel {channel.name}"
                )
                return object_copies, events, ak.ones_like(events, dtype=bool)

            # Extract arguments used by selection function
            sel_args = self._get_function_arguments(
                channel.selection.use,
                object_copies,
                function_name=channel.selection.function.__name__,
            )
            packed = channel.selection.function(*sel_args)

            if not isinstance(packed, PackedSelection):
                raise ValueError(
                    "Selection function must return PackedSelection"
                )

            # Generate mask from PackedSelection object
            mask = ak.Array(packed.all(packed.names[-1]))
            mask = recursive_to_backend(mask, "cpu")

            # Apply certified luminosity filter for data
            if process == "data":
                good_runs = lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                    jax=True,
                )
                mask = mask & ak.to_backend(good_runs, "cpu")
                logger.debug(f"[{label}] Applied luminosity mask for data")

            return object_copies, events, mask

        # Initialise dictionary for channel-wise outputs, and store unmasked input
        per_channel = defaultdict(dict)
        per_channel["__presel"] = {
            "objects": object_copies_analysis,
            "events": events_analysis,
            "mva_objects": object_copies_mva,
            "mva_events": events_mva,
        }

        # Iterate over all channels defined in configuration
        for channel in self.channels:

            channel_name = channel.name

            if not channel.use_in_diff:
                logger.warning(
                    f"Skipping channel {channel_name} in diff analysis"
                )
                continue

            # If a subset of channels is requested, skip the rest
            if (
                req := self.config.general.channels
            ) and channel_name not in req:
                logger.warning(
                    f"Skipping channel {channel_name} (not in requested channels)"
                )
                continue

            # Run selection and get event-level mask
            _, events_analysis_cpu, mask = apply_selection(
                channel,
                object_copies_analysis,
                events_analysis,
                label="analysis",
            )

            # Count number of events passing selection
            n_events = ak.sum(mask)
            if n_events == 0:
                logger.warning(
                    f"No events left in {channel_name} for {process} after selection"
                )
                continue

            # Apply mask to objects and events for both branches
            obj_analysis = {
                k: v[mask] for k, v in object_copies_analysis.items()
            }
            evt_analysis = events_analysis_cpu[mask]
            obj_mva = {k: v[mask] for k, v in object_copies_mva.items()}
            evt_mva = events_mva[mask]

            # Save results to output structure
            per_channel[channel_name] = {
                "objects": obj_analysis,
                "events": evt_analysis,
                "nevents": n_events,
                "mva_objects": obj_mva,
                "mva_events": evt_mva,
                "mva_nevents": len(evt_mva),
            }

        return per_channel

    # -------------------------------------------------------------------------
    # Running training of MVA models
    # -------------------------------------------------------------------------
    def _run_mva_training(
        self, events_per_process_per_mva: dict[str, list[Tuple[dict, int]]]
    ) -> dict[str, Any]:
        """
        Train each MVAConfig using the pre-collected object copies and event counts.

        Parameters
        ----------
        events_per_process_per_mva : dict
            Mapping from MVA name to process name to list of
            (input dictionary, event count).

        Returns
        -------
        dict[str, Any]
            Mapping of MVAConfig name to trained model or parameters.
        """
        trained = {}
        nets = {}

        for mva_cfg in self.config.mva:
            logger.info(f"Training MVA '{mva_cfg.name}'...")

            events_per_process = events_per_process_per_mva[mva_cfg.name]
            if set(events_per_process.keys()) != set(mva_cfg.plot_classes):
                raise ValueError(
                    f"Mismatch between MVA '{mva_cfg.name}' plot classes and provided "
                    "processes: "
                    f"{set(events_per_process.keys())} vs {set(mva_cfg.plot_classes)}"
                )

            # JAX-based model
            if mva_cfg.framework == "jax":
                net = JAXNetwork(mva_cfg)
                X_train, y_train, X_val, y_val, class_weights = (
                    net.prepare_inputs(events_per_process)
                )
                Xtr, Xvl = jnp.array(X_train), jnp.array(X_val)
                ytr, yvl = jnp.array(y_train), jnp.array(y_val)
                net.init_network()
                # Plot feature distributions
                plot_mva_feature_distributions(
                    net.process_to_features_map,
                    mva_cfg,
                    self.config.plotting,
                    self.dirs["mva_plots"],
                )
                params = net.train(Xtr, ytr, Xvl, yvl)
                trained[mva_cfg.name] = params
                nets[mva_cfg.name] = net

            # TensorFlow-based model
            else:
                net = TFNetwork(mva_cfg)
                X_train, y_train, X_val, y_val, class_weights = (
                    net.prepare_inputs(events_per_process)
                )
                net.init_network()
                sw_train = None

                # Optional: apply class-balancing weights to training
                if (
                    class_weights
                    and mva_cfg.balance_strategy == "class_weight"
                ):
                    sw_train = np.vectorize(class_weights.get)(y_train)

                fit_kwargs = (
                    {"sample_weight": sw_train} if sw_train is not None else {}
                )
                model = net.train(X_train, y_train, X_val, y_val, **fit_kwargs)
                trained[mva_cfg.name] = model
                nets[mva_cfg.name] = net

            logger.info(f"Finished training MVA '{mva_cfg.name}'")

        return trained, nets

    # -------------------------------------------------------------------------
    # Preparing data for JAX (untraced)
    # -------------------------------------------------------------------------
    def _prepare_data_for_tracing(
        self,
        events: ak.Array,
        process: str,
    ) -> tuple[dict[str, dict[str, dict[str, jnp.ndarray]]], dict[str, Any]]:
        """
        Preprocess events without JAX tracing for systematic variations.

        This method performs baseline selection and object correction, then
        returns all data needed to fill histograms per systematic variation.

        Parameters
        ----------
        events : ak.Array
            Raw NanoAOD events.
        process : str
            Sample label (e.g. "ttbar", "data").

        Returns
        -------
        tuple[dict, dict]
            - Dictionary mapping variation names to channel-wise data structures.
            - Dictionary with event count statistics for the nominal pass.
        """
        processed_data = {}

        def prepare_full_obj_copies(
            events, mask_set: str, label: str
        ) -> tuple[dict[str, ak.Array], dict[str, ak.Array], ak.Array]:
            """
            Perform object copying, masking, baseline selection, correction,
            and ghost observable computation.

            Parameters
            ----------
            events : ak.Array
                Raw event data.
            mask_set : str
                Name of mask set to apply.
            label : str
                Label for logging.

            Returns
            -------
            tuple[dict[str, ak.Array], dict[str, ak.Array], ak.Array]
                A tuple containing:
                - masked object copies (dict mapping object names to arrays)
                - corrected objects (dict mapping object names to corrected arrays)
                - selection mask (boolean array for event selection)
            """
            # Copy objects and apply good-object masks
            obj_copies = self.get_object_copies(events)
            obj_copies = self.apply_object_masks(obj_copies, mask_set=mask_set)
            logger.debug(
                f"[{label}] Created object copies and applied '{mask_set}' masks"
            )

            # Move all arrays to CPU for baseline selection
            obj_copies = recursive_to_backend(obj_copies, "cpu")

            # Apply baseline selection mask
            baseline_args = self._get_function_arguments(
                self.config.baseline_selection["use"],
                obj_copies,
                function_name=self.config.baseline_selection[
                    "function"
                ].__name__,
            )
            packed = self.config.baseline_selection["function"](*baseline_args)
            mask = ak.Array(packed.all(packed.names[-1]))
            mask = recursive_to_backend(mask, "cpu")

            # Apply selection mask to object copies
            obj_copies = {k: v[mask] for k, v in obj_copies.items()}

            # Compute ghost observables before corrections
            obj_copies = self.compute_ghost_observables(obj_copies)
            logger.debug(f"[{label}] Computed ghost observables")

            # Apply object-level corrections (nominal direction only)
            obj_copies_corrected = self.apply_object_corrections(
                obj_copies, self.corrections, direction="nominal"
            )
            obj_copies_corrected = recursive_to_backend(
                obj_copies_corrected, "cpu"
            )
            logger.debug(f"[{label}] Applied object corrections")

            return obj_copies, obj_copies_corrected, mask

        # Choose mask set for MVA branch ("mva" or fallback to "analysis")
        mva_mask_set = (
            "mva"
            if self.config.good_object_masks.get("mva", [])
            else "analysis"
        )

        # Prepare corrected object copies and masks for both branches
        obj_copies_analysis, obj_copies_corrected_analysis, mask_analysis = (
            prepare_full_obj_copies(
                events, mask_set="analysis", label="analysis"
            )
        )
        obj_copies_mva, obj_copies_corrected_mva, mask_mva = (
            prepare_full_obj_copies(events, mask_set=mva_mask_set, label="mva")
        )

        # Extract nominal per-channel data using corrected objects
        channels_data = self._get_channel_data(
            obj_copies_corrected_analysis,
            events[mask_analysis],
            obj_copies_corrected_mva,
            events[mask_mva],
            process,
            "nominal",
        )
        channels_data["__presel"]["nevents"] = ak.sum(mask_analysis)
        channels_data["__presel"]["mva_nevents"] = ak.sum(mask_mva)
        processed_data["nominal"] = channels_data

        # Collect stats for nominal pass
        stats = {
            "baseline_analysis": ak.sum(mask_analysis),
            "baseline_mva": ak.sum(mask_mva),
            "channels": {
                name: data["nevents"]
                for name, data in channels_data.items()
                if name != "__presel"
            },
        }

        # Loop over systematics if enabled
        if self.config.general.run_systematics:
            for syst in self.systematics + self.corrections:
                if syst.name == "nominal":
                    continue
                for direction in ["up", "down"]:
                    variation_name = f"{syst.name}_{direction}"

                    # Apply systematic correction to object copies
                    obj_copies_corrected_analysis = (
                        self.apply_object_corrections(
                            obj_copies_analysis, [syst], direction=direction
                        )
                    )
                    obj_copies_corrected_mva = self.apply_object_corrections(
                        obj_copies_mva, [syst], direction=direction
                    )

                    # Get channel data for the variation
                    channels_data = self._get_channel_data(
                        obj_copies_corrected_analysis,
                        events[mask_analysis],
                        obj_copies_corrected_mva,
                        events[mask_mva],
                        process,
                        variation_name,
                    )
                    channels_data["__presel"]["nevents"] = ak.sum(
                        mask_analysis
                    )
                    channels_data["__presel"]["mva_nevents"] = ak.sum(mask_mva)
                    processed_data[variation_name] = channels_data

        return processed_data, stats

    # -------------------------------------------------------------------------
    # Histogram building
    # -------------------------------------------------------------------------
    def _histogramming(
        self,
        processed_data: dict,
        mva_instances: dict[str, Any],
        process: str,
        variation: str,
        xsec_weight: float,
        params: dict[str, Any],
        event_syst: Optional[dict[str, Any]] = None,
        direction: Literal["up", "down", "nominal"] = "nominal",
        silent: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """
        Apply selections and fill histograms for each observable and channel.

        Parameters
        ----------
        processed_data : dict
            Preprocessed channel data
        mva_instances : dict[str, Any]
            Trained MVA instances per MVA name
        process : str
            Sample label (e.g., 'ttbar', 'data')
        variation : str
            Systematic variation label
        xsec_weight : float
            Cross-section normalisation factor
        params : dict
            JAX parameters for soft selections and KDE bandwidth
        event_syst : dict, optional
            Event-level systematic information
        direction : str, optional
            Systematic direction ('up', 'down', 'nominal')
        silent : bool, optional
            If True, demote all logs to debug level

        Returns
        -------
        dict[str, jnp.ndarray]
            Histograms for the channel and observables
        """
        jax_config = self.config.jax
        histograms = defaultdict(dict)
        info_logger = logger.info if not silent else logger.debug
        warning_logger = logger.warning if not silent else logger.debug

        # Skip non-nominal variations for data
        if process == "data" and variation != "nominal":
            logger.debug(f"Skipping {variation} for data")
            return histograms

        for channel in self.channels:
            # Skip channels not participating in differentiable analysis
            if not channel.use_in_diff:
                warning_logger(
                    f"Skipping channel {channel.name} (use_in_diff=False)"
                )
                continue

            channel_name = channel.name

            # Skip if channel is not listed in requested channels
            if (
                req := self.config.general.channels
            ) and channel_name not in req:
                warning_logger(
                    f"Skipping channel {channel_name} (not in requested channels)"
                )
                continue

            logger.debug(f"Processing channel: {channel_name}")

            # Extract object and event data for this channel
            obj_copies_ch = processed_data[channel_name]["objects"]
            events_ch = processed_data[channel_name]["events"]
            nevents = processed_data[channel_name]["nevents"]

            # Compute MVA features for this channel
            for mva_name, mva_instance in mva_instances.items():
                mva_features, _ = mva_instance._extract_features(
                    obj_copies_ch, mva_instance.mva_cfg.features
                )
                obj_copies_ch[mva_name] = {
                    "features": mva_features,
                    "instance": mva_instance,
                }

            # Move data to JAX backend
            obj_copies_ch = recursive_to_backend(obj_copies_ch, "jax")
            events_ch = recursive_to_backend(events_ch, "jax")
            logger.debug(f"Channel {channel_name} has {nevents} events")

            # Compute differentiable selection weights using soft cut function
            diff_args = self._get_function_arguments(
                jax_config.soft_selection.use,
                obj_copies_ch,
                function_name=jax_config.soft_selection.function.__name__,
            )
            diff_weights = jax_config.soft_selection.function(
                *diff_args, params
            )
            logger.debug("Computed differentiable weights")

            # Compute cross-section normalised event weights
            if process != "data":
                weights = (events_ch.genWeight * xsec_weight) / abs(
                    events_ch.genWeight
                )
                logger.debug(f"MC weights: xsec_weight={xsec_weight:.4f}")
            else:
                weights = ak.Array(np.ones(nevents))
                logger.debug("Using unit weights for data")

            # Apply event-level systematic correction if available
            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, obj_copies_ch
                )
                logger.debug(
                    f"Applied {event_syst.name} {direction} correction"
                )

            weights = jnp.asarray(ak.to_jax(weights))

            # Loop over observables and compute KDE-based histograms
            for observable in channel.observables:
                obs_name = observable.name

                # Skip observables that are not JAX-compatible
                if not observable.works_with_jax:
                    warning_logger(
                        f"Skipping observable {obs_name} - not JAX-compatible"
                    )
                    continue

                info_logger(
                    f"Histogramming: {process} | {variation} | {channel_name} | "
                    f"{obs_name} | Events (Raw): {nevents:,} | "
                    f"Events(Weighted): {ak.sum(weights):,.2f}"
                )

                # Evaluate observable function
                obs_args = self._get_function_arguments(
                    observable.use,
                    obj_copies_ch,
                    function_name=observable.function.__name__,
                )

                values = jnp.asarray(ak.to_jax(observable.function(*obs_args)))
                logger.debug(f"Computed {obs_name} values")

                # Parse binning specification
                binning = observable.binning
                if isinstance(binning, str):
                    low, high, nbins = map(float, binning.split(","))
                    binning = jnp.linspace(low, high, int(nbins))
                else:
                    binning = jnp.array(binning)

                # Kernel bandwidth for KDE
                bandwidth = params["kde_bandwidth"]

                # Compute cumulative density function (CDF) for each bin edge
                cdf = jax.scipy.stats.norm.cdf(
                    binning.reshape(-1, 1),
                    loc=values.reshape(1, -1),
                    scale=bandwidth,
                )

                # Multiply CDF by per-event weights and compute bin contributions
                weighted_cdf = (
                    cdf * diff_weights.reshape(1, -1) * weights.reshape(1, -1)
                )
                bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
                histogram = jnp.sum(bin_weights, axis=1)

                # Store histogram and binning
                histograms[channel_name][obs_name] = (
                    jnp.asarray(histogram),
                    binning,
                )
                info_logger(
                    f"Filled histogram for {obs_name} in {channel_name}"
                )

        return histograms

    # -------------------------------------------------------------------------
    # Histogram collection
    # -------------------------------------------------------------------------
    def _collect_histograms(
        self,
        processed_data: dict[str, dict[str, ak.Array]],
        metadata: dict[str, Any],
        params: dict[str, Any],
        silent=False,
    ) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
        """
        Run full analysis logic on events from one dataset.

        Parameters
        ----------
        processed_data : dict
            Per-variation channel data from `._prepare_data_for_tracing`
        metadata : dict
            Dataset metadata with keys:
            - 'process': process name
            - 'xsec': cross-section
            - 'nevts': number of generated events
            - 'dataset': dataset name
        params : dict
            JAX parameters for histogramming
        silent : bool, optional
            If True, demote all logs to debug level

        Returns
        -------
        dict
            Histograms keyed by variation/channel/observable
        """
        all_histograms = defaultdict(lambda: defaultdict(dict))
        process = metadata["process"]
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]
        lumi = self.config["general"]["lumi"]

        # Calculate cross-section weight for MC (unit weight for data)
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0
        logger.debug(
            f"Process: {process}, xsec: {xsec}, n_gen: {n_gen}, "
            f"lumi: {lumi}, weight: {xsec_weight:.6f}"
        )

        # Nominal histogramming
        logger.debug(f"Processing nominal variation for {process}")
        histograms = self._histogramming(
            processed_data["nominal"],
            processed_data["mva_nets"],
            process,
            "nominal",
            xsec_weight,
            params,
            silent=silent,
        )
        all_histograms["nominal"] = histograms

        # Loop over systematics (if enabled)
        if self.config.general.run_systematics:
            logger.info(f"Processing systematics for {process}")
            for syst in self.systematics + self.corrections:
                if syst.name == "nominal":
                    continue

                for direction in ["up", "down"]:
                    variation_name = f"{syst.name}_{direction}"
                    logger.info(f"Processing {variation_name} for {process}")

                    histograms = self._histogramming(
                        processed_data[variation_name],
                        processed_data["mva_nets"],
                        process,
                        variation_name,
                        xsec_weight,
                        params,
                        event_syst=syst,
                        direction=direction,
                        silent=silent,
                    )
                    all_histograms[variation_name] = histograms

        return all_histograms

    # -------------------------------------------------------------------------
    # pvalue Calculation
    # -------------------------------------------------------------------------
    def _calculate_pvalue(
        self,
        process_histograms: dict,
        params: dict,
        silent: bool = False,
    ) -> jnp.ndarray:
        """
        Calculate asymptotic p-value using evermore with multi-channel modelling.

        Parameters
        ----------
        process_histograms : dict
            Histograms organised by process, variation, region and observable.
        params : dict
            Fit parameters for p-value calculation.
        silent : bool, optional
            If True, demote all logs to debug level

        Returns
        -------
        jnp.ndarray
            Asymptotic p-value (sqrt(q0)) from profile likelihood ratio.
        """
        info_logger = logger.info if not silent else logger.debug
        info_logger("📊 Calculating p-value from histograms...")

        # Convert histograms to standard Python dictionaries (from nested defaultdicts)
        histograms = nested_defaultdict_to_dict(process_histograms).copy()

        # Use relaxed to compute p-value and maximum likelihood estimators
        pval, aux = compute_discovery_pvalue(
            histograms, self.channels, params
        )

        info_logger("✅ p-value calculation complete\n")
        return pval, aux

    # -------------------------------------------------------------------------
    # The analysis workflow being optimised [histograms + statistics]
    # -------------------------------------------------------------------------
    def _run_traced_analysis_chain(
        self,
        params: dict[str, Any],
        processed_data_events: dict,
        silent: bool = False,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """
        Collect histograms from preprocessed events and compute the final statistical
        p-value.

        This function iterates over the preprocessed events, builds histograms for each
        dataset and process, merges them, and then computes the overall p-value using a
        profile likelihood fit. The resulting histograms are also stored for
        later access (e.g. plotting).

        Parameters
        ----------
        params : dict
            Dictionary of model parameters containing:
            - 'aux': Auxiliary parameters (e.g. selection thresholds)
            - 'fit': Fit parameters for statistical inference (e.g. norm factors)
        processed_data_events : dict
            Nested dictionary containing preprocessed events produced by
            `run_analysis_processing`.
            Structure:
            {
                "<dataset>": {
                    "<file_key>": {
                        "<variation>": (processed_data, metadata),
                        ...
                    },
                    ...
                },
                ...
            }
        silent : bool, optional
            If True, demote all logs to debug level

        Returns
        -------
        tuple
            pvalue : jnp.ndarray
                The computed asymptotic p-value.
            mle_params : dict
                The maximum-likelihood estimated fit parameters obtained from the
                p-value fit.
        """
        info_logger = logger.info if not silent else logger.debug
        info_logger(
            _banner(
                "📊 Starting histogram collection and p-value calculation..."
            )
        )
        histograms_by_process = defaultdict(dict)

        # -------------------------------------------------------------------------
        # Loop over datasets
        # -------------------------------------------------------------------------
        for dataset_name, dataset_files in processed_data_events.items():
            process_name = dataset_name.split("___")[1]

            info_logger(
                f" ⏳ Processing dataset: {dataset_name} "
                f"(process: {process_name}, files: {len(dataset_files)})"
            )

            # Ensure process histogram container exists
            if process_name not in histograms_by_process:
                histograms_by_process[process_name] = defaultdict(
                    lambda: defaultdict(dict)
                )

            # ---------------------------------------------------------------------
            # Loop over files in the dataset
            # ---------------------------------------------------------------------
            info_logger(" 🔍 Collecting histograms for dataset...")
            for file_key, variations in dataset_files.items():
                for variation_name, (
                    processed_data,
                    metadata,
                ) in variations.items():
                    logger.debug(
                        f"  • Collecting histograms for file: {file_key} "
                        f"({variation_name})"
                    )

                    # Build histograms for this file and variation
                    file_histograms = self._collect_histograms(
                        processed_data, metadata, params["aux"], silent=silent
                    )

                    # Merge file histograms into the global container for this process
                    histograms_by_process[process_name] = merge_histograms(
                        histograms_by_process[process_name],
                        dict(file_histograms),
                    )

        # -------------------------------------------------------------------------
        # Compute statistical p-value from histograms
        # -------------------------------------------------------------------------
        info_logger(
            " ✅ Histogram collection complete. Starting p-value calculation..."
        )
        pvalue, aux = self._calculate_pvalue(
            histograms_by_process, params["fit"], silent=silent
        )

        # Store histograms for later plotting or debugging
        self._set_histograms(histograms_by_process)

        return pvalue, aux

    # -------------------------------------------------------------------------
    # Run the data processing code
    # -------------------------------------------------------------------------
    def _prepare_data(
        self,
        params: dict[str, Any],
        read_from_cache: bool = False,
        run_and_cache: bool = True,
        cache_dir: Optional[str] = "/tmp/gradients_analysis/",
        recreate_fit_params: bool = False,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Run full analysis on processed datasets with caching support.

        Parameters
        ----------
        params : dict
            Analysis parameters.
        read_from_cache : bool
            Read preprocessed events from cache.
        run_and_cache : bool
            Process events and cache results.
        cache_dir : str, optional
            Directory for cached events.

        Returns
        -------
        dict
            Preprocessed events keyed by dataset and file.
        """
        config = self.config
        all_events = defaultdict(lambda: defaultdict(dict))
        all_channel_names = {
            f"Channel: {c.name}" for c in config.channels if c.use_in_diff
        }
        summary_data = []

        logger.info(_banner("Processing skimmed data"))

        # Prepare dictionary to collect MVA training data
        mva_data: dict[str, dict[str, list[Tuple[dict, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Use processed datasets from skimming
        if not self.processed_datasets:
            raise ValueError("No processed datasets available for analysis")

        # Loop over processed datasets
        for dataset, events_list in self.processed_datasets.items():
            # Get metadata from first event in the list
            if not events_list:
                continue
            _, metadata = events_list[0]
            process_name = metadata["process"]

            # Skip datasets not explicitly requested in config
            if (req := config.general.processes) and process_name not in req:
                logger.info(
                    f"Skipping {dataset} (process {process_name} not in requested)"
                )
                continue

            dataset_stats = defaultdict(int)

            # Loop over events in the processed dataset
            for idx, (events, file_metadata) in enumerate(events_list):
                # Honour file limit if set in configuration
                if (
                    config.general.max_files != -1
                    and idx >= config.general.max_files
                ):
                    logger.info(
                        f"Reached max files limit ({config.general.max_files})"
                    )
                    break

                # Count skimmed events
                dataset_stats["Skimmed"] += len(events)

                # Run preprocessing pipeline and store processed results
                processed_data, stats = self._prepare_data_for_tracing(events, process_name)
                all_events[f"{dataset}___{process_name}"][f"file__{idx}"][f"events_{idx}"] = (processed_data, file_metadata)

                dataset_stats["Baseline (Analysis)"] += stats["baseline_analysis"]
                dataset_stats["Baseline (MVA)"] += stats["baseline_mva"]
                for ch, count in stats["channels"].items():
                    ch_name = f"Channel: {ch}"
                    dataset_stats[ch_name] += count

                # ------------------------------------------------------
                # If MVA training is enabled, collect data for MVA models
                # ------------------------------------------------------
                # Helper to extract class name and associated process names
                def parse_class_entry(entry: Union[str, dict[str, list[str]]]) -> tuple[str, list[str]]:
                    """
                    Parse MVA class entry to extract class name and associated process names.

                    Parameters
                    ----------
                    entry : Union[str, dict[str, list[str]]]
                        MVA class entry, either a string (process name) or a dictionary
                        mapping class name to list of process names.

                    Returns
                    -------
                    tuple[str, list[str]]
                        A tuple containing:
                        - class_name: Name of the MVA class
                        - process_names: List of process names associated with this class

                    Raises
                    ------
                    ValueError
                        If entry is neither a string nor a dictionary.
                    """
                    if isinstance(entry, str):
                        return entry, [entry]
                    if isinstance(entry, dict):
                        return next(iter(entry.items()))
                    raise ValueError(f"Invalid MVA class type: {type(entry)}. \
                                     Allowed types are str or dict.")

                # Helper to record MVA data
                def record_mva_entry(
                    mva_data: dict[str, dict[str, list[tuple[dict, int]]]],
                    cfg_name: str,
                    class_label: str,
                    presel_ch: dict[str, Any],
                    process_name: str
                ) -> None:
                    """
                    Record MVA training data for a specific class and process.

                    Parameters
                    ----------
                    mva_data : dict[str, dict[str, list[tuple[dict, int]]]]
                        Nested dictionary storing MVA training data, structured as:
                        mva_data[config_name][class_name] = [(objects_dict, event_count), ...]
                    cfg_name : str
                        Name of the MVA configuration.
                    class_label : str
                        Label for the MVA class (e.g., 'signal', 'background').
                    presel_ch : dict[str, Any]
                        Preselection channel data containing 'mva_objects' and 'mva_nevents'.
                    process_name : str
                        Name of the physics process being recorded.

                    Returns
                    -------
                    None
                        Modifies mva_data in place by appending new training data.
                    """
                    nevents = presel_ch["mva_nevents"]
                    logger.debug(
                        f"Adding {nevents} events from process '{process_name}' to MVA class '{class_label}'."
                    )
                    mva_data[cfg_name][class_label].append(
                        (presel_ch["mva_objects"], nevents)
                    )

                # Collect training data for MVA, if enabled
                if config.mva and config.general.run_mva_training:
                    nominal = processed_data.get("nominal", {})
                    presel_ch = nominal.get("__presel")
                    if presel_ch:
                        for mva_cfg in config.mva:
                            seen = set()  # track classes to avoid duplicates
                            # iterate training and plot classes in order
                            for entry in chain(mva_cfg.classes, mva_cfg.plot_classes):
                                class_name, proc_names = parse_class_entry(entry)
                                # fallback default
                                if not class_name or not proc_names:
                                    class_name = process_name
                                    proc_names = [process_name]
                                # skip duplicates
                                if class_name in seen:
                                    continue
                                seen.add(class_name)
                                # record only if this process applies
                                if process_name in proc_names:
                                    record_mva_entry(mva_data, mva_cfg.name, class_name, presel_ch, process_name)


            row = {"Dataset": dataset, "Process": process_name}
            row.update(dataset_stats)
            summary_data.append(row)

            # Log a summary for the just-completed dataset
            headers = [
                "Dataset",
                "Process",
                "Skimmed",
                "Baseline (Analysis)",
                "Baseline (MVA)",
            ] + sorted(list(all_channel_names))
            current_row_values = [row.get(h, 0) for h in headers]
            formatted_row = [current_row_values[0], current_row_values[1]] + [
                f"{int(val):,}" for val in current_row_values[2:]
            ]
            logger.info(
                f"📊 Summary for {dataset}:\n"
                + tabulate(
                    [formatted_row],
                    headers=headers,
                    tablefmt="grid",
                    stralign="right",
                )
                + "\n"
            )

        # After all datasets are processed, print summary table
        sorted_channel_names = sorted(list(all_channel_names))
        headers = [
            "Dataset",
            "Process",
            "Skimmed",
            "Baseline (Analysis)",
            "Baseline (MVA)",
        ] + sorted_channel_names
        table_data = []
        for row_data in summary_data:
            table_row = [row_data.get(h, 0) for h in headers]
            # Format numbers with commas
            formatted_row = [table_row[0], table_row[1]] + [
                f"{int(val):,}" for val in table_row[2:]
            ]
            table_data.append(formatted_row)

        logger.info(
            "📊 Data Processing Summary\n"
            + tabulate(
                table_data, headers=headers, tablefmt="grid", stralign="right"
            )
            + "\n"
        )

        # Run MVA training after all datasets are processed
        models = {}
        nets = {}
        # Attach empty dict of MVA nets to each file’s processed data
        for dataset_files in all_events.values():
            for file_dict in dataset_files.values():
                for skim_key, (processed_data, metadata) in file_dict.items():
                    processed_data["mva_nets"] = nets

        if (
            self.config.general.run_mva_training
            and (mva_cfg := self.config.mva) is not None
        ):
            logger.info(_banner("Executing MVA Pre-training"))
            models, nets = self._run_mva_training(mva_data)

            # Save trained models and attach to processed data
            for model_name in models.keys():
                model = models[model_name]
                net = nets[model_name]
                model_path = self.dirs["mva_models"] / f"{model_name}.pkl"
                net_path = (
                    self.dirs["mva_models"] / f"{model_name}_network.pkl"
                )
                with open(model_path, "wb") as f:
                    cloudpickle.dump(model, f)
                with open(net_path, "wb") as f:
                    cloudpickle.dump(net, f)
                logger.info(f"Saved MVA model '{model_name}' to {model_path}")
                logger.info(
                    f"Saved model network '{model_name}' to {net_path}"
                )

            # Attach MVA nets to each file’s processed data
            for dataset_files in all_events.values():
                for file_dict in dataset_files.values():
                    for skim_key, (
                        processed_data,
                        metadata,
                    ) in file_dict.items():
                        processed_data["mva_nets"] = nets

        return all_events, models, nets, mva_data

    # -------------------------------------------------------------------------
    # Cut Optimisation via Gradient Ascent
    # -------------------------------------------------------------------------
    def run_analysis_optimisation(
        self
    ) -> Tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Perform gradient-based optimisation of analysis selection cuts and
        MVA parameters.

        This function runs the entire workflow:
        - Reads and preprocesses events
        - Runs the initial (unoptimised) analysis and caches the result
        - Computes gradients and optimises selection and fit parameters
        - Saves optimised MVA parameters and produces pre/post-fit plots

        Parameters
        ----------
        fileset : dict
            Mapping of dataset names to metadata and ROOT file paths

        Returns
        -------
        tuple
            - Dictionary of final optimised KDE-based histograms
            - Final JAX scalar p-value
        """
        # Log a summary of the configuration being used for this run
        self._log_config_summary()
        cache_dir = "/tmp/gradients_analysis/"
        # ---------------------------------------------------------------------
        # If not just plotting, begin gradient-based optimisation chain
        # ---------------------------------------------------------------------
        if not self.config.general.run_plots_only:
            # ---------------------------------------------------------------------
            # 1. Configure caching and initialise analysis parameters
            # ---------------------------------------------------------------------
            read_from_cache = self.config.general.read_from_cache
            run_and_cache = not read_from_cache

            # Copy soft selection and NN parameters from config
            auxiliary_parameters = self.config.jax.params.copy()

            all_parameters = {
                "aux": auxiliary_parameters,
                "fit": fit_params,
            }

            # ---------------------------------------------------------------------
            # 2. Preprocess events and extract MVA models (if any)
            # ---------------------------------------------------------------------
            processed_data, mva_models, mva_nets, mva_data = (
                self._prepare_data(
                    all_parameters,
                    read_from_cache=read_from_cache,
                    run_and_cache=run_and_cache,
                    cache_dir=cache_dir,
                )
            )

            # Add MVA model parameters to aux tree (flattened by name)
            for model_name, model in mva_models.items():
                model_parameters = jax.tree.map(jnp.array, model)
                for name, value in model_parameters.items():
                    all_parameters["aux"][name] = value

            # Ensure all parameters are JAX arrays
            all_parameters["aux"] = {
                k: jnp.array(v) for k, v in all_parameters["aux"].items()
            }
            all_parameters["fit"] = {
                k: jnp.array(v) for k, v in all_parameters["fit"].items()
            }

            logger.info("✅ Event preprocessing complete\n")

            # ---------------------------------------------------------------------
            # 3. Run initial traced analysis to compute KDE histograms
            # ---------------------------------------------------------------------
            logger.info(
                _banner("Running initial p-value computation (traced)")
            )
            initial_pvalue, (mle_parameters, mle_parameters_uncertainties) = self._run_traced_analysis_chain(
                all_parameters, processed_data
            )
            initial_histograms = self.histograms

            # Log initial state before optimisation
            _log_parameter_update(
                step="Initial",
                old_p_value=float(initial_pvalue),
                new_p_value=float(initial_pvalue),
                old_params=all_parameters,
                new_params=all_parameters,
                initial_params=all_parameters,
                initial_pval=float(initial_pvalue),
                mva_configs=self.config.mva,
            )

            # ----------------------------------------------------------------------
            # Collect relevant processes and systematics
            # ----------------------------------------------------------------------
            processes, systematics = infer_processes_and_systematics(
                self.processed_datasets, self.config.systematics, self.config.corrections
            )
            logger.info(f"Processes: {processes}")
            logger.info(f"Systematics: {systematics}")

            # ----------------------------------------------------------------------
            # Compute gradients to seed optimiser
            # ----------------------------------------------------------------------
            logger.info(
                _banner("Computing parameter gradients before optimisation")
            )

            (_, _), gradients = jax.value_and_grad(
                self._run_traced_analysis_chain,
                has_aux=True,
                argnums=0,
            )(all_parameters, processed_data)

            # ----------------------------------------------------------------------
            # Prepare for optimisation
            # ----------------------------------------------------------------------
            logger.info(_banner("Preparing for parameter optimisation"))

            # Define objective for optimiser (p-value to minimise)
            def objective(
                params: dict[str, dict[str, Any]],
            ) -> tuple[jnp.ndarray, dict[str, Any]]:
                """
                Objective function for gradient-based optimisation.

                This function computes the p-value and maximum likelihood estimators
                for the given parameters, which serves as the objective to minimise
                during optimisation.

                Parameters
                ----------
                params : dict[str, dict[str, Any]]
                    Parameter dictionary with structure:
                    {"aux": auxiliary_parameters, "fit": fit_parameters}

                Returns
                -------
                tuple[jnp.ndarray, dict[str, Any]]
                    A tuple containing:
                    - p-value (JAX scalar to be minimised)
                    - maximum likelihood estimators (auxiliary output)
                """
                return self._run_traced_analysis_chain(
                    params, processed_data, silent=True
                )

            # Build parameter update/clamping logic
            clamp_fn = make_apply_param_updates(self.config.jax.param_updates)

            # Configure learning rates
            global_lr = self.config.jax.learning_rate
            manual_lrs = self.config.jax.learning_rates or {}
            nn_config_lr = {}
            frozen_keys = set()

            for mva_cfg in self.config.mva:
                if not mva_cfg.grad_optimisation.optimise:
                    for name in all_parameters["aux"]:
                        if "__NN" in name and mva_cfg.name in name:
                            frozen_keys.add(name)
                else:
                    nn_config_lr[mva_cfg.name] = (
                        mva_cfg.grad_optimisation.learning_rate
                    )

            # Construct optimiser with clamping
            tx, _ = make_lr_and_clamp_transform(
                manual_lrs,
                default_auxiliary_lr=global_lr,
                default_fit_lr=1e-3,
                neural_net_lr_map=nn_config_lr,
                clamp_function=clamp_fn,
                frozen_parameter_keys=frozen_keys,
            )(all_parameters)

            # Set up optimisation loop
            logger.info(_banner("Beginning parameter optimisation"))
            initial_params = all_parameters.copy()
            pval_history = []
            aux_history = {
                k: [] for k in all_parameters["aux"] if "__NN" not in k
            }
            mle_history = {k: [] for k in mle_parameters}

            # TODO: Add uncertainty tracking for MLE parameters, and do something with them
            logger.info(f"Uncertainties: {mle_parameters_uncertainties}")
            mle_unc_history = {
                k: [] for k in mle_parameters_uncertainties
            }

            # ----------------------------------------------------------------------
            # Optimisation
            # ----------------------------------------------------------------------
            def optimise_and_log(
                n_steps: int = 100,
            ) -> tuple[jnp.ndarray, dict[str, Any], dict[str, dict[str, Any]]]:
                """
                Run gradient-based optimisation with periodic logging.

                This function performs the main optimisation loop using OptaxSolver,
                logging parameter updates every 10 steps and recording the optimisation
                history for later analysis.

                Parameters
                ----------
                n_steps : int, optional
                    Number of optimisation steps to perform, by default 100.

                Returns
                -------
                tuple[jnp.ndarray, dict[str, Any], dict[str, dict[str, Any]]]
                    A tuple containing:
                    - final_pvalue: Final p-value after optimisation
                    - final_mle_params: Final maximum likelihood estimators
                    - final_parameters: Final optimised parameter dictionary
                """
                parameters = initial_params
                last_logged_params = initial_params
                last_logged_pvalue = initial_pvalue
                solver = OptaxSolver(
                    fun=objective,
                    opt=tx,
                    jit=False,
                    has_aux=True,
                    value_and_grad=False,
                    maxiter=self.config.jax.max_iterations,
                    tol=0.0,
                )
                state = solver.init_state(parameters)
                logger.info("Starting explicit optimisation loop...")

                for step in range(n_steps):
                    new_parameters, state = solver.update(parameters, state)

                    # Record progress
                    pval = state.value
                    (mle, mle_unc) = (
                        state.aux
                    )  # mle values are auxillary outputs of the optimiser
                    pval_history.append(float(pval))
                    for name, value in parameters["aux"].items():
                        if "__NN" not in name:
                            aux_history[name].append(float(value))
                    for name, value in mle.items():
                        mle_history[name].append(float(value))

                    if step % 10 == 0:
                        all_new_parameters = new_parameters
                        all_new_parameters.update({"fit": mle})
                        _log_parameter_update(
                            step=step + 1,
                            old_p_value=last_logged_pvalue,
                            new_p_value=pval,
                            old_params=last_logged_params,
                            new_params=all_new_parameters,
                            initial_params=initial_params,
                            initial_pval=initial_pvalue,
                            mva_configs=self.config.mva,
                        )
                        # Update the baseline for the next log
                        last_logged_params = new_parameters
                        last_logged_pvalue = pval
                    parameters = (
                        new_parameters  # Update for the next iteration
                    )

                return state.value, state.aux, parameters

            # Run optimiser and collect result
            final_pval, (final_mle_pars, final_mle_pars_uncs), final_params = optimise_and_log(
                n_steps=self.config.jax.max_iterations
            )

            # Log final summary table comparing initial and final states
            logger.info(_banner("Optimisation results"))
            _log_parameter_update(
                step="",
                old_p_value=float(initial_pvalue),
                new_p_value=float(final_pval),
                old_params=all_parameters,
                new_params=final_params,
                mva_configs=self.config.mva,
            )

            # ----------------------------------------------------------------------
            # Prepare post-optimisation histograms
            # ----------------------------------------------------------------------
            logger.info(
                _banner(
                    "Running analysis chain with optimised parameters (untraced)"
                )
            )
            # Run the traced analysis chain with final parameters
            _ = self._run_traced_analysis_chain(final_params, processed_data)

            logger.info(
                _banner("Re-computing NN scores/process for MVA models")
            )

            # Compute MVA scores with optimised parameters
            initial_mva_scores, final_mva_scores = {}, {}
            for net_name, net in mva_nets.items():
                final_scores = net.generate_scores_for_processes(
                    mva_data[net_name],
                    final_params["aux"],
                )
                final_mva_scores[net_name] = final_scores

                # Compute it for initial parameters as well
                initial_scores = net.generate_scores_for_processes(
                    mva_data[net_name],
                    initial_params["aux"],
                )
                initial_mva_scores[net_name] = initial_scores

            # ----------------------------------------------------------------------
            # Caching
            # ----------------------------------------------------------------------
            # Cache optimisation results
            with open(f"{cache_dir}/cached_result.pkl", "wb") as f:
                cloudpickle.dump(
                    {
                        "params": final_params,
                        "mle_pars": final_mle_pars,
                        "pvalue": final_pval,
                        "histograms": self.histograms,
                        "pvals_history": pval_history,
                        "aux_history": aux_history,
                        "mle_history": mle_history,
                        "gradients": gradients,
                        "initial_histograms": initial_histograms,
                        "initial_params": all_parameters,
                        "final_mva_scores": final_mva_scores,
                        "initial_mva_scores": initial_mva_scores,
                    },
                    f,
                )

            # Save optimised MVA parameters
            for model_name, model in mva_models.items():
                optimised_nn_params = {
                    p: v
                    for p, v in final_params["aux"].items()
                    if "__NN" in p and model_name in p
                }
                path = self.dirs["mva_models"] / f"{model_name}_optimised.pkl"
                with open(path, "wb") as f:
                    pickle.dump(jax.tree.map(np.array, optimised_nn_params), f)

        logger.info(_banner("Making plots and summaries"))
        # ---------------------------------------------------------------------
        # 4. Reload results and generate summary plots
        # ---------------------------------------------------------------------
        with open(f"{cache_dir}/cached_result.pkl", "rb") as f:
            results = cloudpickle.load(f)

        final_params = results["params"]
        initial_params = results["initial_params"]
        mle_pars = results["mle_pars"]
        final_pval = results["pvalue"]
        pval_history = results["pvals_history"]
        aux_history = results["aux_history"]
        mle_history = results["mle_history"]
        gradients = results["gradients"]
        histograms = results["histograms"]
        initial_histograms = results["initial_histograms"]
        final_mva_scores = results["final_mva_scores"]
        initial_mva_scores = results["initial_mva_scores"]

        logger.info(_banner("Generating parameter evolution plots"))
        # Generate optimisation progress plots
        if self.config.jax.explicit_optimisation:
            logger.info("Generating parameter history plots")
            lrs = self.config.jax.learning_rates or {
                k: self.config.jax.learning_rate for k in final_params["aux"]
            }
            plot_pvalue_vs_parameters(
                pvalue_history=pval_history,
                auxiliary_history=aux_history,
                mle_history=mle_history,
                gradients=gradients,
                learning_rates=lrs,
                plot_settings=self.config.plotting,
                filename=f"{self.dirs['optimisation_plots']}/pvalue_vs_parameters.pdf",
            )

            plot_parameters_over_iterations(
                pvalue_history=pval_history,
                auxiliary_history=aux_history,
                mle_history=mle_history,
                gradients=gradients,
                learning_rates=lrs,
                plot_settings=self.config.plotting,
                filename=f"{self.dirs['optimisation_plots']}/parameters_vs_iter.pdf",
            )

        # ---------------------------------------------------------------------
        # 5. Generate pre- and post-fit plots
        # ---------------------------------------------------------------------
        logger.info("Generating pre and post-fit plots")

        def make_pre_and_post_fit_plots(
            histograms_dict: dict[
                str, dict[str, dict[str, dict[str, jnp.ndarray]]]
            ],
            label: str,
            fitted_pars: dict[str, Any],
        ) -> None:
            """
            Generate pre-fit and post-fit histogram plots for all analysis channels.

            This function creates CMS-style histogram plots showing data vs. Monte Carlo
            predictions, either before fitting (pre-fit) or after fitting (post-fit)
            with the provided fitted parameters.

            Parameters
            ----------
            histograms_dict : dict[str, dict[str, dict[str, dict[str, jnp.ndarray]]]]
                Nested dictionary containing histograms structured as:
                histograms[process][variation][channel][observable]
                = (histogram, bin_edges)
            label : str
                Label to include in the output filename (e.g., 'prefit', 'postfit').
            fitted_pars : dict[str, Any]
                Dictionary of fitted parameters to use for scaling templates.

            Returns
            -------
            None
                Saves plots to the fit_plots directory but returns nothing.
            """
            channel_data_list, _ = build_channel_data_scalar(
                histograms_dict, self.config.channels
            )
            for channel_data in channel_data_list:
                channel_cfg = next(
                    c
                    for c in self.config.channels
                    if c.name == channel_data.name
                )
                fit_obs = channel_cfg.fit_observable
                obs_label = next(
                    observable.label
                    for observable in channel_cfg.observables
                    if observable.name == fit_obs
                )
                fig = create_cms_histogram(
                    bin_edges=channel_data.bin_edges,
                    data=channel_data.observed_counts,
                    templates=channel_data.templates,
                    fitted_params=fitted_pars,
                    plot_settings=self.config.plotting,
                    xlabel=obs_label,
                )
                fig.savefig(
                    f"{self.dirs['fit_plots']}/{label}_{channel_data.name}.pdf",
                    dpi=300,
                )

        # Plot post-fit (final) and post-fit (initial) comparison
        make_pre_and_post_fit_plots(histograms, "postopt_postfit", mle_pars)
        make_pre_and_post_fit_plots(
            initial_histograms, "preopt_postfit", mle_pars
        )
        make_pre_and_post_fit_plots(
            initial_histograms, "preopt_prefit", initial_params["fit"]
        )

        # ---------------------------------------------------------------------
        # 6. Plot NN scores distributions
        # ---------------------------------------------------------------------
        for net_name in final_mva_scores.keys():
            initial_scores = initial_mva_scores[net_name]
            final_scores = final_mva_scores[net_name]
            plot_mva_scores(
                initial_scores,
                self.config.plotting,
                self.dirs["mva_plots"],
                prefix=f"{net_name}_preopt_",
            )
            plot_mva_scores(
                final_scores,
                self.config.plotting,
                self.dirs["mva_plots"],
                prefix=f"{net_name}_postopt_",
            )

        return histograms, final_pval
