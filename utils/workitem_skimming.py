"""
Workitem-based preprocessing workflow using dask.bag for failure handling.

This module implements a new preprocessing approach that processes workitems directly
using dask.bag, providing robust failure handling and retry mechanisms. It serves as
an alternative to the current fileset-based skimming approach.

The workflow follows the pattern:
1. Process workitems with dask.bag.from_sequence()
2. Apply skimming analysis to each workitem
3. Handle failures and collect successful results
4. Retry failed workitems until completion
5. Save outputs using the same patterns as existing code
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from collections import defaultdict

import awkward as ak
import cloudpickle
import dask.bag
import hist
import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.processor.executor import WorkItem
from tabulate import tabulate

from utils.schema import SkimmingConfig
from utils.tools import get_function_arguments

logger = logging.getLogger(__name__)


def default_histogram() -> hist.Hist:
    """
    Create a default histogram for tracking processing success/failure.

    This histogram serves as a dummy placeholder to track whether workitems
    were processed successfully. The actual analysis histograms are created
    separately during the analysis phase.

    Returns
    -------
    hist.Hist
        A simple histogram with regular binning for tracking purposes
    """
    return hist.Hist.new.Regular(10, 0, 1000).Weight()


def workitem_analysis(workitem: WorkItem, config: SkimmingConfig,
                     configuration: Any, file_counters: Dict[str, int],
                     part_counters: Dict[str, int], is_mc: bool = True) -> Dict[str, Any]:
    """
    Process a single workitem for skimming analysis.

    This function is equivalent to the `my_analysis` function in the reference
    implementation. It handles I/O, applies skimming selections, and saves
    output files on successful processing.

    Parameters
    ----------
    workitem : WorkItem
        The coffea WorkItem containing file metadata and entry ranges
    config : SkimmingConfig
        Skimming configuration with selection functions and output settings
    configuration : Any
        Main analysis configuration object containing branch selections
    file_counters : Dict[str, int]
        Pre-computed mapping of file keys to file numbers
    part_counters : Dict[str, int]
        Pre-computed mapping of part keys (including entry ranges) to part numbers
    is_mc : bool, default True
        Whether the workitem represents Monte Carlo data

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'hist': Dummy histogram for success tracking
        - 'failed_items': Set of failed workitems (empty on success)
        - 'processed_events': Number of events processed
        - 'output_files': List of created output files
    """
    dummy_hist = default_histogram()

    try:
        # Extract workitem metadata
        filename = workitem.filename
        treename = workitem.treename
        entry_start = workitem.entrystart
        entry_stop = workitem.entrystop
        dataset = workitem.dataset

        logger.debug(f"Processing workitem: {filename}[{entry_start}:{entry_stop}]")

        # Load events using NanoEventsFactory
        events = NanoEventsFactory.from_root(
            {filename: treename},
            entry_start=entry_start,
            entry_stop=entry_stop,
            schemaclass=NanoAODSchema,
        ).events()

        total_events = len(events)
        logger.debug(f"Loaded {total_events} events from workitem")

        # Apply skimming selection using the provided function
        selection_func = config.selection_function
        selection_use = config.selection_use

        # Get function arguments using existing utility
        selection_args = get_function_arguments(
            selection_use, events, function_name=selection_func.__name__
        )
        packed_selection = selection_func(*selection_args)

        # Apply final selection mask
        selection_names = packed_selection.names
        if selection_names:
            final_selection = selection_names[-1]
            mask = packed_selection.all(final_selection)
        else:
            # No selection applied, keep all events
            mask = slice(None)

        filtered_events = events[mask]
        processed_events = len(filtered_events)
        logger.debug(f"After selection: {processed_events} events remain")

        # Fill dummy histogram with some dummy values for tracking
        if processed_events > 0:
            # Use a simple observable for the dummy histogram
            dummy_values = [500.0] * min(processed_events, 100)  # Limit for performance
            dummy_hist.fill(dummy_values)

        output_files = []
        if processed_events > 0:
            output_file = _create_output_file_path(workitem, config, file_counters, part_counters)
            _save_workitem_output(filtered_events, output_file, config, configuration, is_mc)
            output_files.append(str(output_file))
            logger.debug(f"Saved {processed_events} events to {output_file}")

        return {
            "hist": dummy_hist,
            "failed_items": set(),
            "processed_events": processed_events,
            "output_files": output_files
        }

    except Exception as e:
        logger.error(f"Failed to process workitem {workitem.filename}: {e}")
        return {
            "hist": default_histogram(),
            "failed_items": {workitem},  # Track failure
            "processed_events": 0,
            "output_files": []
        }


def reduce_results(result_a: Dict[str, Any], result_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine partial results from workitem processing.

    This function is equivalent to the `reduce` function in the reference
    implementation. It combines histograms, failed items, and other metrics
    from parallel processing.

    Parameters
    ----------
    result_a : Dict[str, Any]
        First result dictionary
    result_b : Dict[str, Any]
        Second result dictionary

    Returns
    -------
    Dict[str, Any]
        Combined result dictionary
    """
    return {
        "hist": result_a["hist"] + result_b["hist"],
        "failed_items": result_a["failed_items"] | result_b["failed_items"],
        "processed_events": result_a["processed_events"] + result_b["processed_events"],
        "output_files": result_a["output_files"] + result_b["output_files"]
    }


def _create_output_file_path(workitem: WorkItem, config: SkimmingConfig,
                           file_counters: Dict[str, int], part_counters: Dict[str, int]) -> Path:
    """
    Create output file path following the existing pattern with entry-range-based counters.

    Uses the same output structure as the current skimming code:
    {output_dir}/{dataset}/file__{file_idx}/part_{chunk}.root

    Parameters
    ----------
    workitem : WorkItem
        The workitem being processed
    config : SkimmingConfig
        Skimming configuration with output directory
    file_counters : Dict[str, int]
        Pre-computed mapping of file keys to file numbers
    part_counters : Dict[str, int]
        Pre-computed mapping of part keys (including entry ranges) to part numbers

    Returns
    -------
    Path
        Full path to the output file
    """
    dataset = workitem.dataset

    # Create keys that include entry ranges for proper differentiation
    file_key = f"{dataset}::{workitem.filename}"
    part_key = f"{file_key}::{workitem.entrystart}_{workitem.entrystop}"

    # Get pre-computed file and part numbers
    file_number = file_counters[file_key]
    part_number = part_counters[part_key]

    # Create output directory structure
    base_dir = Path(config.output_dir)
    dataset_dir = base_dir / dataset / f"file__{file_number}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename with entry-range-based part number
    output_filename = f"part_{part_number}.root"
    return dataset_dir / output_filename


def _save_workitem_output(events: Any, output_file: Path, config: SkimmingConfig,
                         configuration: Any, is_mc: bool) -> None:
    """
    Save filtered events to output ROOT file.

    This function handles the actual I/O of saving skimmed events to disk,
    using the same branch selection logic as the existing skimming code.

    Parameters
    ----------
    events : Any
        Filtered events to save
    output_file : Path
        Output file path
    config : SkimmingConfig
        Skimming configuration
    configuration : Any
        Main analysis configuration with branch selections
    is_mc : bool
        Whether this is Monte Carlo data
    """
    # Build branches to keep using existing logic
    branches_to_keep = _build_branches_to_keep(configuration, is_mc)

    # Create output file
    with uproot.recreate(str(output_file)) as output_root:
        # Prepare data for writing
        output_data = {}

        # Extract branches following the existing pattern
        for obj, obj_branches in branches_to_keep.items():
            if obj == "event":
                # Event-level branches
                for branch in obj_branches:
                    if hasattr(events, branch):
                        output_data[branch] = getattr(events, branch)
            else:
                # Object collection branches
                if hasattr(events, obj):
                    obj_collection = getattr(events, obj)
                    for branch in obj_branches:
                        if hasattr(obj_collection, branch):
                            output_data[f"{obj}_{branch}"] = getattr(obj_collection, branch)

        # Create and populate output tree
        if output_data:
            output_tree = output_root.mktree(config.tree_name,
                                           {k: v.type for k, v in output_data.items()})
            output_tree.extend(output_data)


def _build_branches_to_keep(configuration: Any, is_mc: bool) -> Dict[str, List[str]]:
    """
    Build dictionary of branches to keep based on configuration.

    This replicates the logic from the existing skimming code to determine
    which branches should be saved in the output files.

    Parameters
    ----------
    configuration : Any
        Main analysis configuration
    is_mc : bool
        Whether this is Monte Carlo data

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping object names to lists of branch names
    """
    branches = configuration.preprocess.branches
    mc_branches = configuration.preprocess.mc_branches

    filtered = {}
    for obj, obj_branches in branches.items():
        if not is_mc:
            # For data, exclude MC-only branches
            filtered[obj] = [
                br for br in obj_branches
                if br not in mc_branches.get(obj, [])
            ]
        else:
            # For MC, keep all branches
            filtered[obj] = obj_branches

    return filtered


class WorkitemSkimmingManager:
    """
    Manager for workitem-based skimming using dask.bag processing.

    This class orchestrates the new preprocessing workflow that processes
    workitems directly using dask.bag, providing robust failure handling
    and retry mechanisms.

    Attributes
    ----------
    config : SkimmingConfig
        Skimming configuration with selection functions and output settings
    """

    def __init__(self, config: SkimmingConfig):
        """
        Initialize the workitem skimming manager.

        Parameters
        ----------
        config : SkimmingConfig
            Skimming configuration with selection functions and output settings
        """
        self.config = config
        logger.info("Initialized workitem-based skimming manager")

    def process_workitems(self, workitems: List[WorkItem], configuration: Any,
                         split_every: int = 4, max_retries: int = 3) -> Dict[str, Any]:
        """
        Process a list of workitems using dask.bag with failure handling.

        This is the main entry point that implements the dask.bag workflow
        with retry logic for failed workitems.

        Parameters
        ----------
        workitems : List[WorkItem]
            List of workitems to process
        configuration : Any
            Main analysis configuration object
        split_every : int, default 4
            Split parameter for dask.bag.fold operation
        max_retries : int, default 3
            Maximum number of retry attempts for failed workitems

        Returns
        -------
        Dict[str, Any]
            Final combined results with histograms and processing statistics
        """
        logger.info(f"Starting workitem processing with {len(workitems)} workitems")

        # Pre-compute file and part counters for all workitems
        file_counters, part_counters = self._compute_counters(workitems)

        # Initialize accumulator for successful results
        full_result = {
            "hist": default_histogram(),
            "failed_items": set(),
            "processed_events": 0,
            "output_files": []
        }

        # Process workitems with retry logic
        remaining_workitems = workitems.copy()
        retry_count = 0

        while remaining_workitems and retry_count < max_retries:
            logger.info(f"Processing {len(remaining_workitems)} workitems (attempt {retry_count + 1})")

            # Create dask bag from remaining workitems
            bag = dask.bag.from_sequence(remaining_workitems)

            # Map analysis function over workitems
            futures = bag.map(lambda wi: workitem_analysis(
                wi, self.config, configuration, file_counters, part_counters,
                is_mc=self._is_monte_carlo(wi.dataset)
            ))

            # Reduce results using fold operation
            task = futures.fold(reduce_results, split_every=split_every)

            # Compute results
            (result,) = dask.compute(task)

            # Update remaining workitems to failed ones
            remaining_workitems = list(result["failed_items"])

            # Accumulate successful results
            if result["processed_events"] > 0:
                full_result["hist"] += result["hist"]
                full_result["processed_events"] += result["processed_events"]
                full_result["output_files"].extend(result["output_files"])

            # Log progress
            failed_count = len(remaining_workitems)
            successful_count = len(workitems) - failed_count
            logger.info(f"Completed attempt {retry_count + 1}: "
                       f"{successful_count} successful, {failed_count} failed")

            retry_count += 1

        # Final logging
        if remaining_workitems:
            logger.warning(f"Failed to process {len(remaining_workitems)} workitems "
                          f"after {max_retries} attempts")
            full_result["failed_items"] = set(remaining_workitems)
        else:
            logger.info("Successfully processed all workitems")

        # Create summary statistics by dataset
        self._log_processing_summary(workitems, full_result["output_files"])

        return full_result

    def discover_workitem_outputs(self, workitems: List[WorkItem]) -> List[str]:
        """
        Discover existing output files from previous workitem processing.

        This method scans for output files that would be created by the
        workitem processing, allowing for resumption of interrupted workflows.

        Parameters
        ----------
        workitems : List[WorkItem]
            List of workitems to check for existing outputs

        Returns
        -------
        List[str]
            List of existing output file paths with tree names
        """
        output_files = []

        # Use the same counter computation as processing
        file_counters, part_counters = self._compute_counters(workitems)

        for workitem in workitems:
            expected_output = _create_output_file_path(workitem, self.config, file_counters, part_counters)

            if expected_output.exists():
                # Add tree name for compatibility with existing code
                file_with_tree = f"{expected_output}:{self.config.tree_name}"
                output_files.append(file_with_tree)

        logger.info(f"Discovered {len(output_files)} existing workitem output files")
        return output_files

    def _log_processing_summary(self, workitems: List[WorkItem], output_files: List[str]) -> None:
        """
        Log a summary table of processing results by dataset.

        Parameters
        ----------
        workitems : List[WorkItem]
            Original list of workitems processed
        output_files : List[str]
            List of output files created
        """
        # Collect statistics by dataset
        dataset_stats = defaultdict(lambda: {"events_processed": 0, "files_written": 0})

        # Count events processed by reading from output files
        for output_file in output_files:
            try:
                # Extract dataset from file path
                # Path format: {output_dir}/{dataset}/file__{N}/part_{M}.root
                path_parts = Path(output_file).parts
                if len(path_parts) >= 3:
                    dataset = path_parts[-3]  # Get dataset from path

                    # Read the file to count events
                    with uproot.open(output_file) as f:
                        if self.config.tree_name in f:
                            tree = f[self.config.tree_name]
                            num_events = tree.num_entries
                            dataset_stats[dataset]["events_processed"] += num_events
                            dataset_stats[dataset]["files_written"] += 1

            except Exception as e:
                logger.debug(f"Could not read events from {output_file}: {e}")
                # Still count the file even if we can't read events
                try:
                    path_parts = Path(output_file).parts
                    if len(path_parts) >= 3:
                        dataset = path_parts[-3]
                        dataset_stats[dataset]["files_written"] += 1
                except:
                    pass

        # Create summary table
        if dataset_stats:
            table_data = []
            total_events = 0
            total_files = 0

            for dataset, stats in sorted(dataset_stats.items()):
                events = stats["events_processed"]
                files = stats["files_written"]
                table_data.append([dataset, f"{events:,}", files])
                total_events += events
                total_files += files

            # Add totals row
            table_data.append(["TOTAL", f"{total_events:,}", total_files])

            # Create and log table
            headers = ["Dataset", "Events Saved", "Files Written"]
            table = tabulate(table_data, headers=headers, tablefmt="grid")

            logger.info("Processing Summary:")
            logger.info(f"\n{table}")
        else:
            logger.info("No output files were created during processing")

    def _compute_counters(self, workitems: List[WorkItem]) -> tuple[Dict[str, int], Dict[str, int]]:
        """
        Pre-compute file and part counters for all workitems.

        This ensures consistent numbering across all workers by computing
        the counters once before parallel processing begins.

        Parameters
        ----------
        workitems : List[WorkItem]
            List of all workitems to process

        Returns
        -------
        tuple[Dict[str, int], Dict[str, int]]
            File counters and part counters dictionaries
        """
        file_counters = {}
        part_counters = {}

        # Track unique files per dataset for sequential file numbering
        dataset_file_counts = {}

        for workitem in workitems:
            dataset = workitem.dataset
            file_key = f"{dataset}::{workitem.filename}"
            part_key = f"{file_key}::{workitem.entrystart}_{workitem.entrystop}"

            # Assign file number if not already assigned
            if file_key not in file_counters:
                if dataset not in dataset_file_counts:
                    dataset_file_counts[dataset] = 0
                file_counters[file_key] = dataset_file_counts[dataset]
                dataset_file_counts[dataset] += 1

            # Assign part number if not already assigned
            if part_key not in part_counters:
                # Count existing parts for this file
                existing_parts = [k for k in part_counters.keys() if k.startswith(f"{file_key}::")]
                part_counters[part_key] = len(existing_parts)

        logger.debug(f"Computed counters for {len(file_counters)} files and {len(part_counters)} parts")
        return file_counters, part_counters

    def _is_monte_carlo(self, dataset: str) -> bool:
        """
        Determine if a dataset represents Monte Carlo data.

        Parameters
        ----------
        dataset : str
            Dataset name

        Returns
        -------
        bool
            True if dataset is Monte Carlo, False if it's data
        """
        # Simple heuristic: data datasets typically contain "data" in the name
        return "data" not in dataset.lower()


def process_workitems_with_skimming(workitems: List[WorkItem], config: Any,
                                   fileset: Optional[Dict[str, Any]] = None,
                                   nanoaods_summary: Optional[Dict[str, Any]] = None,
                                   cache_dir: str = "/tmp/gradients_analysis/") -> Dict[str, Any]:
    """
    Process workitems using the workitem-based skimming approach with event merging and caching.

    This function serves as the main entry point for the workitem-based preprocessing workflow.
    It processes workitems (if skimming is enabled) and then discovers, merges, and caches events
    from the saved files for analysis. Events from multiple output files per dataset are
    automatically merged into a single NanoEvents object for improved performance and memory efficiency.

    Parameters
    ----------
    workitems : List[WorkItem]
        List of workitems to process, typically from NanoAODMetadataGenerator.workitems
    config : Any
        Main analysis configuration object containing skimming and preprocessing settings
    fileset : Optional[Dict[str, Any]], default None
        Fileset containing metadata including cross-sections for normalization
    nanoaods_summary : Optional[Dict[str, Any]], default None
        NanoAODs summary containing event counts per dataset for nevts metadata
    cache_dir : str, default "/tmp/gradients_analysis/"
        Directory for caching merged events. Cached files use the pattern:
        {cache_dir}/{dataset}__{hash}.pkl where hash is based on input file paths

    Returns
    -------
    Dict[str, List[Tuple[Any, Dict[str, Any]]]]
        Dictionary mapping dataset names to lists containing a single (events, metadata) tuple.
        Events are merged NanoEvents objects from all output files for the dataset.
        Each metadata dictionary contains dataset, process, variation, and xsec information.
    """
    logger.info(f"Starting workitem-based preprocessing with {len(workitems)} workitems")

    # Create workitem skimming manager
    skimming_manager = WorkitemSkimmingManager(config.preprocess.skimming)

    # Group workitems by dataset
    workitems_by_dataset = {}
    for workitem in workitems:
        dataset = workitem.dataset
        if dataset not in workitems_by_dataset:
            workitems_by_dataset[dataset] = []
        workitems_by_dataset[dataset].append(workitem)

    # Process workitems if skimming is enabled
    if config.general.run_skimming:
        logger.info("Running skimming - processing workitems")
        results = skimming_manager.process_workitems(workitems, config)
        logger.info(f"Skimming completed - processed {results['processed_events']:,} events")

    # Always discover and read from saved files
    logger.info("Discovering and reading from saved files")
    processed_datasets = {}

    for dataset, dataset_workitems in workitems_by_dataset.items():
        # Skip datasets not explicitly requested in config
        if hasattr(config.general, 'processes') and config.general.processes:
            process_name = dataset.split('__')[0] if '__' in dataset else dataset
            if process_name not in config.general.processes:
                logger.info(f"Skipping {dataset} (process {process_name} not in requested)")
                continue

        # Discover output files for this dataset
        output_files = skimming_manager.discover_workitem_outputs(dataset_workitems)

        if output_files:
            # Create metadata for compatibility with existing analysis code
            metadata = {
                "dataset": dataset,
                "process": dataset.split('__')[0] if '__' in dataset else dataset,
                "variation": dataset.split('__')[1] if '__' in dataset else "nominal",
            }

            # Add cross-section metadata from fileset if available
            if fileset and dataset in fileset:
                xsec = fileset[dataset].get('metadata', {}).get('xsec', 1.0)
                metadata['xsec'] = xsec
            else:
                metadata['xsec'] = 1.0
                if fileset:
                    logger.warning(f"Cross-section not found for dataset {dataset}, using default value 1.0")

            # Add nevts from NanoAODs summary if available
            # The analysis code expects 'nevts' field for normalization
            nevts = 0
            if nanoaods_summary:
                # Parse dataset to get process and variation
                process_name = dataset.split('__')[0] if '__' in dataset else dataset
                variation = dataset.split('__')[1] if '__' in dataset else "nominal"

                if process_name in nanoaods_summary:
                    if variation in nanoaods_summary[process_name]:
                        nevts = nanoaods_summary[process_name][variation].get('nevts_total', 0)

            metadata['nevts'] = nevts
            if nevts == 0:
                logger.warning(f"Could not find nevts for dataset {dataset}, using 0")

            # Create cache key for the merged dataset
            # Use sorted file paths to ensure consistent cache key
            sorted_files = sorted(output_files)
            cache_input = f"{dataset}::{':'.join(sorted_files)}"
            cache_key = hashlib.md5(cache_input.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{dataset}__{cache_key}.pkl")

            # Check if we should read from cache
            if config.general.read_from_cache and os.path.exists(cache_file):
                logger.info(f"Reading cached merged events for dataset {dataset}")
                try:
                    with open(cache_file, "rb") as f:
                        merged_events = cloudpickle.load(f)
                    logger.info(f"Loaded {len(merged_events)} cached events for dataset {dataset}")
                    processed_datasets[dataset] = [(merged_events, metadata.copy())]
                    continue  # Skip to next dataset
                except Exception as e:
                    logger.error(f"Failed to load cached events for {dataset}: {e}")
                    # Fall back to loading from files

            # Load and merge events from all discovered files
            all_events = []
            total_events_loaded = 0

            for file_path in output_files:
                try:
                    # Load events using NanoEventsFactory (same as old function)
                    events = NanoEventsFactory.from_root(
                        file_path, schemaclass=NanoAODSchema, mode="eager"
                    ).events()
                    events = ak.materialize(events)  # Ensure data is loaded into memory
                    all_events.append(events)
                    total_events_loaded += len(events)
                    logger.debug(f"Loaded {len(events)} events from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load events from {file_path}: {e}")
                    continue

            # Merge all events into a single array if we have any events
            if all_events:
                try:
                    if len(all_events) == 1:
                        # Single file, no need to concatenate
                        merged_events = all_events[0]
                    else:
                        # Multiple files, concatenate them
                        merged_events = ak.concatenate(all_events, axis=0)

                    logger.info(f"Merged {len(output_files)} files into {len(merged_events)} total events for dataset {dataset}")

                    # Cache the merged events if not reading from cache
                    #if not config.general.read_from_cache:
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, "wb") as f:
                            cloudpickle.dump(merged_events, f)
                        logger.info(f"Cached merged events for dataset {dataset}")
                    except Exception as e:
                        logger.warning(f"Failed to cache events for {dataset}: {e}")

                    processed_datasets[dataset] = [(merged_events, metadata.copy())]

                except Exception as e:
                    logger.error(f"Failed to merge events for dataset {dataset}: {e}")
                    # Fallback to individual events if merging fails
                    processed_events = []
                    for i, events in enumerate(all_events):
                        processed_events.append((events, metadata.copy()))
                        logger.debug(f"Added individual events chunk {i} as fallback")
                    processed_datasets[dataset] = processed_events
        else:
            logger.warning(f"No output files found for dataset {dataset}")
            processed_datasets[dataset] = []

    return processed_datasets
