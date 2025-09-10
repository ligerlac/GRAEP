from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import evermore as evm

from jaxtyping import Array, Float, PyTree
from typing import TypeAlias, Any, Callable

import logging
# Configure module-level logger for debugging and monitoring
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

jax.config.update("jax_enable_x64", True)  # Use 64-bit precision

FScalar: TypeAlias = Float[Array, ""]  # Scalar float array type
Hist1D: TypeAlias = Float[Array, "bins"]  # 1D histogram type (sumw)
Hists1D: TypeAlias = dict[str, Hist1D]  # Dictionary of 1D histograms
Params: TypeAlias = dict[str, evm.AbstractParameter[FScalar]]  # Parameter dictionary


# Initial fit parameters for the analysis
# These are the "internal" parameters used by evermore for the fit and they contain more information than just the values (like names, constraints, etc)
# These parameters are meant to be extended!!
evm_params: Params = {
    "mu": evm.Parameter(value=1.0, name="mu"),  # Signal strength parameter
    "scale_ttbar": evm.Parameter(value=1.0, name="scale_ttbar"),  # ttbar scaling factor
}

# These are the pure values of the parameters that will be optimized by the analysis
# They are supposed to be trainable and do not contain any extra information (that is not meant to be optimized!)
fit_params = evm.tree.pure(evm_params)


# Function to update evermore (internal) parameters with new values from the analysis optimization loop
def update(params: Params, values: PyTree[FScalar]) -> Params:
    return jax.tree.map(
        evm.parameter.replace_value,
        params,
        values,
        is_leaf=evm.filter.is_parameter,
    )


def model_per_channel(params: Params, hists: Hists1D) -> Hists1D:
    # we put all modifiers into a list, so that we can compose them for application (scaling).
    # composing is important! it ensures that there's no order dependency in the application of the modifiers,
    # and it allows us to apply multiple modifiers at once through batching (vmap) of the same modifier types,
    # which greatly improves performance and reduces compiletime.
    out = {}

    for proc, hist in hists.items():
        # we can use pattern matching to select which modifiers to apply to which processes
        # this is available in python 3.10+ (and we use atleast python 3.11 in this repo due to JAX requirements)
        match proc:
            case "signal":
                # signal, add more modifiers to this list if needed
                modifier = evm.modifier.Compose(*[params["mu"].scale()])
                out[proc] = modifier(hist)
            case "ttbar_semilep":
                # ttbar semilep, add more modifiers to this list if needed
                modifier = evm.modifier.Compose(*[params["scale_ttbar"].scale()])
                out[proc] = modifier(hist)
            case _:
                # For all other processes, we currently do not apply any scaling
                # This is a placeholder for future modifications, e.g., for background processes
                out[proc] = hist

    # Other backgrounds stay as they are, no scaling applied (yet)
    return out


def loss_per_channel(dynamic: Params, static: Params, hists: Hists1D, observation: Hist1D) -> FScalar:
    params = evm.tree.combine(dynamic, static)
    expected = evm.util.sum_over_leaves(model_per_channel(params, hists))
    # Poisson NLL of the expectation and observation
    log_likelihood = (
        evm.pdf.PoissonContinuous(lamb=expected).log_prob(observation).sum()
    )
    # Add parameter constraints from logpdfs
    constraints = evm.loss.get_log_probs(params)
    # Sum over all constraints (i.e., priors)
    constraints = jax.tree.map(jnp.sum, constraints)
    log_likelihood += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(log_likelihood)


class ChannelData(eqx.Module):
    name: str = eqx.field(static=True)
    observed_counts: Hist1D
    templates: Hists1D
    bin_edges: jax.Array


def total_loss(dynamic: Params, static: Params, channels: list[ChannelData]) -> FScalar:
    loss = 0.0
    for channel in channels:
        # Compute loss for each channel
        loss += loss_per_channel(
            dynamic=dynamic,
            static=static,
            hists=channel.templates,
            observation=channel.observed_counts,
        )
    return loss



def fit(
    params: Params,
    channels: list[ChannelData],
) -> tuple[FScalar, tuple[Params, PyTree[FScalar]]]:
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)

    dynamic, static = evm.tree.partition(params, filter=evm.filter.is_not_frozen)

    # wrap the loss function to match optimistix's expectations
    def optx_loss(dynamic, args):
        return total_loss(dynamic, *args)

    fitresult = optx.minimise(
        optx_loss,
        solver,
        dynamic,
        has_aux=False,
        args=(static, channels),
        options={},
        max_steps=10_000,
        throw=True,
    )
    # NLL
    nll = total_loss(fitresult.value, static, channels)

    # bestfit parameters
    bestfit_params = evm.tree.combine(fitresult.value, static)

    # bestfit parameter uncertainties
    # We use the Cramer-Rao bound to estimate uncertainties
    # use the bestfit parameters to compute the uncertainties, and split it by value of the parameters
    # we explicitly not use `filter=evm.filter.is_not_frozen` here, because we want to compute uncertainties
    # for all parameters, not just the "unfrozen" ones
    dynamic, static = evm.tree.partition(bestfit_params, filter=evm.filter.is_value)
    bestfit_params_uncertainties = evm.loss.cramer_rao_uncertainty(
        loss_fn=jax.tree_util.Partial(total_loss, static=static, channels=channels),
        tree=dynamic,
    )
    return (nll, (bestfit_params, bestfit_params_uncertainties))


@eqx.filter_jit  # JIT compile for performance
def pvalue_bestfitparams_uncertainties(
    params: Params,
    channels: list[ChannelData],
    test_poi: float,
    poi_where: Callable,
) -> tuple[FScalar, tuple[Params, PyTree[FScalar]]]:
    """Calculate expected p-values via q0 test."""
    # global fit
    two_nll, (bestfit_params, bestfit_params_uncertainties) = fit(params, channels)

    # conditional fit at test_poi
    # Fix `mu` and freeze the parameter
    params = eqx.tree_at(lambda t: poi_where(t).frozen, params, True)
    params = eqx.tree_at(lambda t: poi_where(t).raw_value, params, evm.parameter.to_value(test_poi))
    two_nll_conditional, *_ = fit(params, channels)


    # Calculate the likelihood ratio
    # q0 = -2 ln [L(μ=0, θ̂̂) / L(μ̂, θ̂)]
    likelihood_ratio = 2.0 * (two_nll_conditional - two_nll)

    poi_hat = poi_where(bestfit_params).value
    q0 = jnp.where(poi_hat >= test_poi, likelihood_ratio, 0.0)
    # p = 1 - Φ(√q₀)
    p0 = 1.0 - jax.scipy.stats.norm.cdf(jnp.sqrt(q0))
    # return p0, bestfit parameters (as pure values, i.e. without extra info like names, constraints, etc), and their uncertainties
    # need to return 2 elements for `jax.value_and_grad(..., has_aux=True)` to work properly
    aux = (evm.tree.pure(bestfit_params), bestfit_params_uncertainties)
    return (p0, aux)



def build_channel_data_scalar(
    histogram_dictionary: PyTree[Hist1D],
    channel_configurations: list[Any],
) -> tuple[list[ChannelData], None]:
    """
    Construct ChannelData objects from nested histogram dictionary.

    Parameters
    ----------
    histogram_dictionary : PyTree[Hist1D]
        Nested histogram structure:
            Level 1: Process names (e.g., "signal", "ttbar")
            Level 2: Systematic variations (e.g., "nominal", "scale_up")
            Level 3: Channel names
            Level 4: Observable names → (counts, bin_edges) or counts
    channel_configurations : list[Any]
        Channel configuration objects with attributes:
            - name: Channel identifier
            - fit_observable: Key for target observable
            - use_in_discovery: Flag to include channel

    Returns
    -------
    channel_data_list : list[ChannelData]
        Constructed channel data containers


    Notes
    -----
    - Only uses "nominal" systematic variation
    - Automatically creates zero templates for missing required processes
    - Skips channels not marked for discovery use
    - Converts all inputs to JAX arrays for compatibility
    """
    channel_data_list = []

    # Process each channel configuration
    for config in channel_configurations:
        # Skip channels excluded from discovery fit
        if not getattr(config, "use_in_discovery", True):
            continue
        if not getattr(config, "use_in_diff", True):
            continue

        channel_name = config.name
        observable_key = config.fit_observable

        # =====================================================================
        # Step 1: Extract observed data
        # =====================================================================
        try:
            # Navigate nested dictionary: data → nominal → channel → observable
            data_container = (
                histogram_dictionary.get("data", {})
                .get("nominal", {})
                .get(channel_name, {})
                .get(observable_key, None)
            )

            if data_container is None:
                logger.warning(f"Missing data for {channel_name}/{observable_key}")
                continue
        except KeyError:
            logger.exception(f"Data access error for {channel_name}")
            continue

        # Tuple format: (counts, bin_edges)
        observed_counts, bin_edges = data_container
        observed_counts = jnp.asarray(observed_counts)
        bin_edges = jnp.asarray(bin_edges)

        # =====================================================================
        # Step 2: Build process templates
        # =====================================================================
        process_templates = {}
        for process_name, variations in histogram_dictionary.items():
            # Skip data entry (already handled)
            if process_name == "data":
                continue

            try:
                # Extract nominal histogram for this process/channel/observable
                nominal_hist = (
                    variations
                    .get("nominal", {})
                    .get(channel_name, {})
                    .get(observable_key, None)
                )

                if nominal_hist is None:
                    continue
            except KeyError:
                logger.warning(
                    f"Missing nominal histogram for {process_name} in {channel_name}"
                )
                continue

            # Tuple format: (counts, edges) - extract counts only
            counts = jnp.asarray(nominal_hist[0])

            process_templates[process_name] = counts

        # =====================================================================
        # Step 3: Ensure required processes exist
        # =====================================================================
        # Create zero templates for required processes if missing
        zero_template = jnp.zeros_like(observed_counts)
        if "signal" not in process_templates:
            logger.info(f"Adding zero signal template for {channel_name}")
            process_templates["signal"] = zero_template
        if "ttbar_semilep" not in process_templates:
            logger.info(f"Adding zero ttbar template for {channel_name}")
            process_templates["ttbar_semilep"] = zero_template

        # =====================================================================
        # Step 4: Create channel container
        # =====================================================================
        channel_data = ChannelData(
            name=channel_name,
            observed_counts=observed_counts,
            templates=process_templates,
            bin_edges=bin_edges,
        )
        channel_data_list.append(channel_data)
    return channel_data_list, None


def compute_discovery_pvalue(
    histogram_dictionary: PyTree[Hist1D],
    channel_configurations: list[Any],
    parameters: PyTree[FScalar],
    signal_strength_test_value: float = 0.0,
) -> tuple[FScalar, tuple[Params, PyTree[FScalar]]]:
    """
    Calculate discovery p-value using profile likelihood ratio.

    Implements the test statistic:
        q₀ = -2 ln [ L(μ=0, θ̂̂) / L(μ̂, θ̂) ]
    where:
        μ = signal strength
        θ = nuisance parameters (here κ_tt)
        θ̂̂ = conditional MLE under μ=0
        (μ̂, θ̂) = unconditional MLE

    Parameters
    ----------
    histogram_dictionary : PyTree[Hist1D]
        Nested histogram structure (see build_channel_data_scalar)
    channel_configurations : list[Any]
        Channel configuration objects
    parameters : Params, optional
        Initial parameter values for optimization, default:
            {"mu": 1.0, "scale_ttbar": 1.0}
    signal_strength_test_value : float, optional
        Signal strength for null hypothesis (typically 0 for discovery)

    Returns
    -------
    p_value : jnp.ndarray
        Asymptotic p-value for discovery (1-tailed)
    mle_parameters : Dict[str, jnp.ndarray]
        Maximum Likelihood Estimates under null hypothesis

    Notes
    -----
    - Uses the "evermore" library for automatic differentiation-based inference
    - Implements the q₀ test statistic from arXiv:1007.1727
    - The p-value is computed using the asymptotic approximation:
        p = 1 - Φ(√q₀)
      where Φ is the standard normal CDF
    """
    # =====================================================================
    # Step 1: Prepare data and model
    # =====================================================================
    channels, _ = build_channel_data_scalar(
        histogram_dictionary, channel_configurations
    )

    # Handle case with no valid channels
    if not channels:
        logger.error("Discovery calculation aborted: no valid channels")
        return jnp.array(0.0), {}

    # update the internal evm parameters with the provided values optimized by the analysis
    # the internal evm_params have much more information that is needed for the fit (like names, constraints, etc), but they are not supposed to be trainable
    params = update(evm_params, parameters)
    return pvalue_bestfitparams_uncertainties(
        params=params,
        channels=channels,
        test_poi=signal_strength_test_value,
        poi_where=lambda t: t["mu"],  # Default path to signal strength parameter
    )
