#!/usr/bin/env python3

"""
ZprimeAnalysis framework for applying object and event-level systematic corrections
on NanoAOD ROOT files and producing histograms of observables like mtt. Supports both
correctionlib-based and function-based corrections.
"""
import logging
import sys
import warnings

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from analysis.diff import DifferentiableAnalysis
from analysis.nondiff import NonDiffAnalysis
from user.configuration import config as ZprimeConfig
from utils.datasets import ConfigurableDatasetManager
from utils.logging import setup_logging, log_banner
from utils.schema import Config, load_config_with_restricted_cli
from utils.metadata_extractor import NanoAODMetadataGenerator
from utils.skimming import process_workitems_with_skimming

# -----------------------------
# Logging Configuration
# -----------------------------
setup_logging()

logger = logging.getLogger("AnalysisDriver")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

# -----------------------------
# Main Driver
# -----------------------------
def main():
    """
    Main driver function for running the Zprime analysis framework.
    Loads configuration, runs preprocessing, and dispatches analysis over datasets.
    """
    cli_args = sys.argv[1:]
    full_config = load_config_with_restricted_cli(ZprimeConfig, cli_args)
    config = Config(**full_config)  # Pydantic validation
    logger.info(f"Luminosity: {config.general.lumi}")
    dataset_manager = ( ConfigurableDatasetManager(config.datasets)
                       if config.datasets
                       else None
                    )

    logger.info(log_banner("metadata and workitems extraction"))
    # Generate metadata and fileset from NanoAODs
    generator = NanoAODMetadataGenerator(dataset_manager=dataset_manager)
    generator.run(generate_metadata=config.general.run_metadata_generation)
    fileset = generator.fileset
    workitems = generator.workitems
    if not workitems:
        logger.error("No workitems available. Please ensure metadata generation completed successfully.")
        sys.exit(1)

    logger.info(log_banner("SKIMMING AND PROCESSING"))
    logger.info(f"Processing {len(workitems)} workitems")

    # Process workitems with dask-awkward
    processed_datasets = process_workitems_with_skimming(workitems, config, fileset, generator.nanoaods_summary)


    analysis_mode = config.general.analysis
    if analysis_mode == "skip":
        logger.info(log_banner("Skim-Only Mode: Skimming Complete"))
        logger.info("✅ Skimming completed successfully. Analysis skipped as requested.")
        logger.info(f"Skimmed files are available in the configured output directories.")
        return
    elif analysis_mode == "nondiff":
        logger.info(log_banner("Running Non-Differentiable Analysis"))
        nondiff_analysis = NonDiffAnalysis(config, processed_datasets)
        nondiff_analysis.run_analysis_chain()
    elif analysis_mode == "diff":
        logger.info(log_banner("Running Differentiable Analysis"))
        diff_analysis = DifferentiableAnalysis(config, processed_datasets)
        diff_analysis.run_analysis_optimisation()
    else:  # "both"
        logger.info(log_banner("Running both Non-Differentiable and Differentiable Analysis"))
        # Non-differentiable analysis
        logger.info("Running Non-Differentiable Analysis")
        nondiff_analysis = NonDiffAnalysis(config, processed_datasets)
        nondiff_analysis.run_analysis_chain()
        # Differentiable analysis
        logger.info("Running Differentiable Analysis")
        diff_analysis = DifferentiableAnalysis(config, processed_datasets)
        diff_analysis.run_analysis_optimisation()


if __name__ == "__main__":
    main()
