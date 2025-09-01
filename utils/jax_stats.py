"""
Module for calculating relaxed discovery p-value in high-energy physics analyses.

Implements a statistical framework using JAX and Equinox for efficient computation of
discovery p-values using profile likelihood methods. The module handles:
  - Poisson log-likelihood calculation for binned data
  - Construction of statistical models with signal and background components
  - Hypothesis testing for new physics discovery

Key Concepts:
  - Profile Likelihood Ratio: Test statistic for discovery p-value
  - Asimov Dataset: Used for expected p-value calculations
  - Relaxed Inference: Automatic differentiation for statistical inference

Reference: CMS-NOTE-2011/005
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import relaxed

# Configure module-level logger for debugging and monitoring
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@jax.jit  # Just-In-Time compilation for performance
def poisson_log_likelihood(
    observed_counts: jnp.ndarray,
    expected_rates: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Poisson log-likelihood for observed counts given expected rates.

    The Poisson probability mass function is:
        P(k|λ) = (λ^k * e^{-λ}) / k!

    Taking the natural logarithm gives:
        log P = k·ln(λ) - λ - ln(k!)

    Parameters
    ----------
    observed_counts : jnp.ndarray
        Array of observed event counts per bin (k)
    expected_rates : jnp.ndarray
        Array of expected event rates per bin (λ)

    Returns
    -------
    jnp.ndarray
        Log-likelihood contribution for each bin

    Notes
    -----
    - The gamma function (gammaln) is used for ln(k!) = ln(Γ(k+1))
    - Small epsilon prevents numerical instability at λ=0
    - JIT-compiled for performance on GPU/TPU
    """
    EPSILON = 1e-12  # Numerical stability constant
    return (
        observed_counts * jnp.log(expected_rates + EPSILON)  # k·ln(λ)
        - expected_rates  # -λ
        - jsp.special.gammaln(observed_counts + 1.0)  # -ln(k!)
    )


class ChannelData(eqx.Module):
    """
    Container for experimental data in a single analysis channel.

    Represents a orthogonal event category (e.g., lepton flavor + jet multiplicity)
    with binned distributions of a discriminating observable.

    Attributes
    ----------
    name : str
        Channel identifier (e.g., "muon_jet5")
    observed_counts : jnp.ndarray
        Observed event counts in histogram bins
    templates : Dict[str, jnp.ndarray]
        MC templates for signal/background processes
    bin_edges : jnp.ndarray
        Bin boundaries for the observable

    Notes
    -----
    - Static fields (name) are treated as constants by Equinox
    - Templates should include all relevant processes (signal + backgrounds)
    - Bin edges are stored for visualization/rebinning purposes
    """

    name: str = eqx.field(static=True)  # Treated as constant by JAX
    observed_counts: jnp.ndarray
    templates: Dict[str, jnp.ndarray]
    bin_edges: jnp.ndarray

    def __init__(
        self,
        name: str,
        observed_counts: jnp.ndarray,
        templates: Dict[str, jnp.ndarray],
        bin_edges: jnp.ndarray,
    ) -> None:
        """
        Initialize an analysis channel container.

        Parameters
        ----------
        name : str
            Unique channel identifier
        observed_counts : jnp.ndarray
            Observed data counts
        templates : Dict[str, jnp.ndarray]
            Process templates (normalized to expected yields)
        bin_edges : jnp.ndarray
            Histogram bin boundaries
        """
        # Validate inputs
        if not templates:
            raise ValueError("Channel requires at least one template")
        if len(observed_counts.shape) != 1:
            raise ValueError("Observed counts must be 1D array")

        # Consistency check: all templates should match data shape
        for proc, template in templates.items():
            if template.shape != observed_counts.shape:
                raise ValueError(
                    f"Template '{proc}' shape {template.shape} "
                    f"doesn't match data {observed_counts.shape}"
                )

        self.name = name
        self.observed_counts = observed_counts
        self.templates = templates
        self.bin_edges = bin_edges


class AllBackgroundsModelScalar(eqx.Module):
    """
    Statistical model for discovery p-value test.

    Implements a two-parameter model:
        μ (signal strength) and κ_tt (ttbar normalization)
    with all other backgrounds fixed.

    The expected rate in bin i is:
        λ_i = μ·S_i + κ_tt·B_i^{tt} + Σ_{other} B_i^{other}

    where:
        S_i = signal template
        B_i^{tt} = ttbar template
        B_i^{other} = fixed background templates

    Attributes
    ----------
    channels : List[ChannelData]
        Analysis channels included in the model

    Notes
    -----
    - Inherits from eqx.Module for JAX-compatible object-oriented programming
    - Uses static_field for channels to prevent tracing during JIT compilation
    - Implements the likelihood interface required by the 'relaxed' inference library
    """

    channels: List[ChannelData]

    def __init__(self, channels: List[ChannelData]) -> None:
        """
        Initialize statistical model with analysis channels.

        Parameters
        ----------
        channels : List[ChannelData]
            Analysis channels to include in the fit
        """
        if not channels:
            raise ValueError("Model requires at least one channel")
        self.channels = channels

    def expected_rates(
        self,
        parameters: Dict[str, jnp.ndarray],
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Compute expected event rates for all channels.

        Parameters
        ----------
        parameters : Dict[str, jnp.ndarray]
            Model parameters:
                - "mu": signal strength modifier (μ)
                - "scale_ttbar": ttbar normalization factor (κ_tt)

        Returns
        -------
        main_expectations : List[jnp.ndarray]
            Expected event rates per bin for each channel
        aux_expectations : List[jnp.ndarray]
            Placeholder for auxiliary measurements (unused in this model)

        Notes
        -----
        - The signal template is scaled by μ (signal strength)
        - Only ttbar background has a free normalization (κ_tt)
        - Other backgrounds remain fixed at nominal values
        - Returns empty auxiliary constraints as this model doesn't
          implement systematics
        """
        signal_strength = parameters["mu"]
        ttbar_scale = parameters["scale_ttbar"]

        main_expectations = []  # Expected rates per channel
        aux_expectations = []  # Auxiliary constraints (empty)

        for channel in self.channels:
            # Start with zero expected events
            total_expected = jnp.zeros_like(channel.observed_counts)

            # Sum contributions from all processes
            for process_name, template in channel.templates.items():
                if process_name == "signal":
                    # Scale signal by μ parameter
                    total_expected += signal_strength * template
                elif process_name == "ttbar_semilep":
                    # Scale ttbar by κ_tt parameter
                    total_expected += ttbar_scale * template
                else:
                    # Fixed backgrounds (no scaling)
                    total_expected += template

            main_expectations.append(total_expected)
            # No auxiliary measurements in this model
            aux_expectations.append(jnp.array([]))

        return main_expectations, aux_expectations

    def logpdf(
        self,
        data: Tuple[List[jnp.ndarray], List[jnp.ndarray]],
        pars: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute total log-likelihood for the observed data.

        Parameters
        ----------
        data : Tuple[List[jnp.ndarray], List[jnp.ndarray]]
            Experimental data:
                [0]: List of observed counts per channel
                [1]: Unused (auxiliary measurements placeholder)
        parameters : Dict[str, jnp.ndarray]
            Current parameter values

        Returns
        -------
        jnp.ndarray
            Scalar total log-likelihood

        Notes
        -----
        - The function signature is fixed by the 'relaxed' library
        - The likelihood is the product of Poisson probabilities per bin
        - Total log-likelihood is the sum over all bins and channels
        - This function is differentiable w.r.t. parameters
          (enables gradient-based inference)
        - This function is differentiable w.r.t. parameters
          (enables gradient-based inference)
        """
        observed_counts_per_channel, _ = data
        expected_rates_per_channel, _ = self.expected_rates(pars)

        total_log_likelihood = 0.0
        # Iterate over channels
        for i, (observed, expected) in enumerate(
            zip(observed_counts_per_channel, expected_rates_per_channel)
        ):
            # Compute Poisson log-likelihood for each bin in the channel
            bin_log_likelihoods = poisson_log_likelihood(observed, expected)
            # Sum over bins (marginalize over bins)
            channel_log_likelihood = jnp.sum(bin_log_likelihoods)
            total_log_likelihood += channel_log_likelihood

        return total_log_likelihood


def build_channel_data_scalar(
    histogram_dictionary: Dict[str, Dict[str, Dict[str, Any]]],
    channel_configurations: List[Any],
) -> Tuple[List[ChannelData], List[jnp.ndarray]]:
    """
    Construct ChannelData objects from nested histogram dictionary.

    Parameters
    ----------
    histogram_dictionary : Dict[str, Dict[str, Dict[str, Any]]]
        Nested histogram structure:
            Level 1: Process names (e.g., "signal", "ttbar")
            Level 2: Systematic variations (e.g., "nominal", "scale_up")
            Level 3: Channel names
            Level 4: Observable names → (counts, bin_edges) or counts
    channel_configurations : List[Any]
        Channel configuration objects with attributes:
            - name: Channel identifier
            - fit_observable: Key for target observable
            - use_in_discovery: Flag to include channel

    Returns
    -------
    channel_data_list : List[ChannelData]
        Constructed channel data containers
    observed_counts_list : List[jnp.ndarray]
        Observed counts for each channel

    Notes
    -----
    - Only uses "nominal" systematic variation
    - Automatically creates zero templates for missing required processes
    - Skips channels not marked for discovery use
    - Converts all inputs to JAX arrays for compatibility
    """
    channel_data_list = []
    observed_counts_list = []

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
                logger.warning(
                    f"Missing data for {channel_name}/{observable_key}"
                )
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
                    variations.get("nominal", {})
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
        observed_counts_list.append(observed_counts)

    return channel_data_list, observed_counts_list


def compute_discovery_pvalue(
    histogram_dictionary: Dict[str, Dict[str, Dict[str, Any]]],
    channel_configurations: List[Any],
    parameters: Dict[str, float],
    signal_strength_test_value: float = 0.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
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
    histogram_dictionary : Dict[str, Dict[str, Dict[str, Any]]]
        Nested histogram structure (see build_channel_data_scalar)
    channel_configurations : List[Any]
        Channel configuration objects
    initial_parameters : Dict[str, float], optional
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
    - Uses the "relaxed" library for automatic differentiation-based inference
    - Implements the q₀ test statistic from arXiv:1007.1727
    - The p-value is computed using the asymptotic approximation:
        p = 1 - Φ(√q₀)
      where Φ is the standard normal CDF
    """
    # =====================================================================
    # Step 1: Prepare data and model
    # =====================================================================
    channels, observed_counts = build_channel_data_scalar(
        histogram_dictionary, channel_configurations
    )

    # Handle case with no valid channels
    if not channels:
        logger.error("Discovery calculation aborted: no valid channels")
        return jnp.array(0.0), {}

    # Initialize statistical model
    statistical_model = AllBackgroundsModelScalar(channels)

    # Package data: (main observations, auxiliary constraints)
    experimental_data = (observed_counts, [])  # No auxiliary measurements

    # =====================================================================
    # Step 2: Perform hypothesis test
    # =====================================================================
    # Use the 'relaxed' library to compute the profile likelihood ratio
    p_value, mle_parameters = relaxed.infer.hypotest(
        test_poi=signal_strength_test_value,  # Parameter of interest (μ)
        data=experimental_data,  # Observed data
        model=statistical_model,  # Statistical model
        init_pars=parameters,  # Starting point for optimization
        return_mle_pars=True,  # Return fitted nuisance parameters
        test_stat="q0",  # Discovery test statistic
    )
    return p_value, mle_parameters
