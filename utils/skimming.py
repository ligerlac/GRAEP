"""
Centralized skimming manager with configurable selections and dual-mode support.

This module provides a unified interface for skimming that supports both:
1. NanoAOD/DAK mode with PackedSelection functions
2. Pure uproot mode with string-based cuts

The skimming logic uses the same functor pattern as other selections in the codebase.
"""

import glob
import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import awkward as ak
import cloudpickle
import dask_awkward as dak
import numpy as np
import uproot
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from tqdm import tqdm

from utils.schema import SkimmingConfig
from utils.tools import get_function_arguments

logger = logging.getLogger(__name__)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

# Fixed output pattern for simplicity
SKIMMING_OUTPUT_PATTERN = "part_{chunk}.root"  # chunk number will be filled in


class ConfigurableSkimmingManager:
    """
    Centralized skimming with config-driven selections and paths.

    This class handles both NanoAOD (with PackedSelection) and uproot (with cut strings)
    preprocessing modes, using configuration to define selections.
    """

    def __init__(self, config: SkimmingConfig):
        """
        Initialize the skimming manager with configuration.

        Parameters
        ----------
        config : SkimmingConfig
            Configuration containing selection functions and parameters
        """
        self.config = config
        logger.info("Initialized configurable skimming manager")

        # Validate configuration
        if not config.nanoaod_selection and not config.uproot_cut_string:
            raise ValueError("Either nanoaod_selection or uproot_cut_string must be provided")


    def get_uproot_cut_string(self) -> Optional[str]:
        """
        Get the uproot cut string.

        Returns
        -------
        str or None
            Cut string for uproot mode, or None if not configured
        """
        return self.config.uproot_cut_string

    def apply_nanoaod_selection(
        self,
        events: ak.Array,
        function_args_getter: callable
    ) -> ak.Array:
        """
        Apply NanoAOD selection using the configured functor.

        Parameters
        ----------
        events : ak.Arrayƒ√
            NanoAOD events array
        function_args_getter : callable
            Function to get arguments for the selection function
            (typically from analysis base class)

        Returns
        -------
        ak.Array
            Boolean mask for selected events
        """
        if not self.config.nanoaod_selection:
            raise ValueError("No NanoAOD selection configured")

        # Get the selection function and its arguments
        selection_func = self.config.nanoaod_selection.function
        selection_use = self.config.nanoaod_selection.use

        # Get function arguments (this delegates to the analysis class method)
        selection_args = function_args_getter(
            selection_use,
            events,
            function_name=selection_func.__name__
        )

        # Apply the selection function
        packed_selection = selection_func(*selection_args)

        if not isinstance(packed_selection, PackedSelection):
            raise ValueError(f"Selection function must return PackedSelection, got {type(packed_selection)}")

        # Get the final selection mask
        # Assume the last selection name is the final one, or use 'all' if available
        selection_names = packed_selection.names
        if selection_names:
            final_selection = selection_names[-1]
            mask = ak.Array(packed_selection.all(final_selection))
        else:
            # Fallback: create a mask that selects all events
            mask = ak.ones_like(events.run, dtype=bool)

        return mask

    def get_chunk_size(self) -> int:
        """
        Get the configured chunk size for processing.

        Returns
        -------
        int
            Number of events to process per chunk
        """
        return self.config.chunk_size

    def get_tree_name(self) -> str:
        """
        Get the configured tree name.

        Returns
        -------
        str
            ROOT tree name
        """
        return self.config.tree_name

    def get_output_dir(self) -> str:
        """
        Get the configured output directory.

        Returns
        -------
        str
            Output directory path
        """
        return self.config.output_dir

    def get_dataset_output_dir(self, dataset: str, file_idx: int) -> Path:
        """
        Get the output directory for a specific dataset and file index.

        Parameters
        ----------
        dataset : str
            Dataset name
        file_idx : int
            File index

        Returns
        -------
        Path
            Output directory path following: {output_dir}/{dataset}/file__{idx}/
        """
        base_dir = Path(self.config.output_dir)
        return base_dir / dataset / f"file__{file_idx}"

    def get_glob_pattern(self, dataset: str, file_idx: int) -> str:
        """
        Get a glob pattern for finding output files for a specific dataset and file index.

        Parameters
        ----------
        dataset : str
            Dataset name
        file_idx : int
            File index

        Returns
        -------
        str
            Glob pattern for finding output files
        """
        # Build the pattern based on the expected directory structure
        # Structure: {output_dir}/{dataset}/file__{idx}/part_*.root
        return f"{self.config.output_dir}/{dataset}/file__{file_idx}/part_*.root"

    def supports_nanoaod_mode(self) -> bool:
        """
        Check if NanoAOD mode is supported.

        Returns
        -------
        bool
            True if NanoAOD selection is configured
        """
        return self.config.nanoaod_selection is not None

    def supports_uproot_mode(self) -> bool:
        """
        Check if uproot mode is supported.

        Returns
        -------
        bool
            True if uproot cut string is configured
        """
        return self.config.uproot_cut_string is not None


    def discover_skimmed_files(self, dataset: str, file_idx: int) -> list[str]:
        """
        Discover existing skimmed files for a specific dataset and file index.

        Parameters
        ----------
        dataset : str
            Dataset name to search for
        file_idx : int
            File index to search for

        Returns
        -------
        list[str]
            List of skimmed file paths with tree names
        """
        glob_pattern = self.get_glob_pattern(dataset, file_idx)
        logger.debug(f"Searching for skimmed files with pattern: {glob_pattern}")
        found_files = glob.glob(glob_pattern)

        # Add tree name for uproot
        files_with_tree = [f"{file_path}:{self.config.tree_name}" for file_path in found_files]

        logger.info(f"Discovered {len(files_with_tree)} skimmed files for {dataset}/file__{file_idx}")
        return files_with_tree

    def validate_user_skimmed_files(self, user_files: dict[str, list[str]]) -> bool:
        """
        Validate that user-provided skimmed files follow the expected structure.

        Parameters
        ----------
        user_files : dict[str, list[str]]
            Dictionary mapping dataset names to lists of file paths

        Returns
        -------
        bool
            True if files are valid and accessible
        """
        for dataset, files in user_files.items():
            for file_path in files:
                # Remove tree name if present
                clean_path = file_path.split(':')[0] if ':' in file_path else file_path

                if not os.path.exists(clean_path):
                    logger.error(f"Skimmed file not found: {clean_path}")
                    return False

                # Check if it's a ROOT file
                if not clean_path.endswith('.root'):
                    logger.warning(f"File {clean_path} does not appear to be a ROOT file")

        logger.info(f"Validated {sum(len(files) for files in user_files.values())} user-provided skimmed files")
        return True

    def build_branches_to_keep(self, config, mode="uproot", is_mc=False):
        """
        Build list or dict of branches to keep for preprocessing.

        Parameters
        ----------
        config : Config
            Configuration object with a preprocess block.
        mode : str
            'uproot' returns a flat list; 'dask' returns a dict.
        is_mc : bool
            Whether input files are Monte Carlo.

        Returns
        -------
        dict or list
            Branches to retain depending on mode.
        """
        branches = config.preprocess.branches
        mc_branches = config.preprocess.mc_branches
        filtered = {}

        for obj, obj_branches in branches.items():
            if not is_mc:
                filtered[obj] = [
                    br for br in obj_branches if br not in mc_branches.get(obj, [])
                ]
            else:
                filtered[obj] = obj_branches

        if mode == "dask":
            return filtered

        if mode == "uproot":
            flat = []
            for obj, brs in filtered.items():
                flat.extend(
                    brs if obj == "event" else [f"{obj}_{br}" for br in brs]
                )
            return flat

        raise ValueError("Invalid mode: use 'dask' or 'uproot'.")

    def skim(
        self,
        input_path: str,
        tree: str,
        dataset: str,
        file_idx: int,
        configuration,
        processor: str = "uproot",
        is_mc: bool = True,
    ) -> Union[int, bool]:
        """
        Skim input ROOT file using the specified processor.

        Parameters
        ----------
        input_path : str
            Path to the input ROOT file.
        tree : str
            Name of the TTree inside the file.
        dataset : str
            Dataset name for organizing output files.
        file_idx : int
            File index for organizing output files.
        configuration : object
            Configuration object containing branch selection and other settings.
        processor : str
            Processor to use: "uproot" or "dask"
        is_mc : bool
            Whether input files are Monte Carlo.

        Returns
        -------
        Union[int, bool]
            For dask: Total number of input events before filtering.
            For uproot: True if successful, False otherwise.
        """
        # Common setup - create output directory
        output_dir = self.get_dataset_output_dir(dataset, file_idx)
        os.makedirs(output_dir, exist_ok=True)

        # Get total events for logging
        with uproot.open(f"{input_path}:{tree}") as f:
            total_events = f.num_entries

        logger.info("========================================")
        logger.info(f"📂 Preprocessing file: {input_path} with {total_events:,} events")

        # Get common parameters
        step_size = self.get_chunk_size()
        output_tree_name = self.get_tree_name()

        # Dispatch to processor-specific implementation
        if processor == "dask":
            return self._skim_with_dask(
                input_path, tree, output_dir, configuration, is_mc,
                total_events, step_size, output_tree_name
            )
        elif processor == "uproot":
            return self._skim_with_uproot(
                input_path, tree, output_dir, configuration, is_mc,
                total_events, step_size, output_tree_name
            )
        else:
            raise ValueError(f"Unknown processor: {processor}. Use 'uproot' or 'dask'.")

    def _skim_with_dask(
        self,
        input_path: str,
        tree: str,
        output_dir: Path,
        configuration,
        is_mc: bool,
        total_events: int,
        step_size: int,
        output_tree_name: str,
    ) -> int:
        """
        Dask-specific skimming implementation.
        """
        branches = self.build_branches_to_keep(configuration, mode="dak", is_mc=is_mc)
        chunk_num = 0

        for start in range(0, total_events, step_size):
            stop = min(start + step_size, total_events)

            events = NanoEventsFactory.from_root(
                {input_path: tree},
                schemaclass=NanoAODSchema,
                entry_start=start,
                entry_stop=stop,
                mode="dask",
            ).events()

            # Apply configurable selection
            selection_func = self.config.nanoaod_selection["function"]
            selection_use = self.config.nanoaod_selection["use"]

            selection_args = get_function_arguments(
                selection_use, events, function_name=selection_func.__name__
            )
            packed_selection = selection_func(*selection_args)

            # Get final selection mask
            selection_names = packed_selection.names
            if selection_names:
                final_selection = selection_names[-1]
                mask = ak.Array(packed_selection.all(final_selection))
            else:
                mask = ak.ones_like(events.run, dtype=bool)

            filtered = events[mask]

            # Skip empty chunks
            if len(filtered) == 0:
                logger.debug(f"Chunk {chunk_num} is empty after selection, skipping")
                continue

            subset = {}
            for obj, obj_branches in branches.items():
                if obj == "event":
                    subset.update(
                        {
                            br: filtered[br]
                            for br in obj_branches
                            if br in filtered.fields
                        }
                    )
                elif obj in filtered.fields:
                    subset.update(
                        {
                            f"{obj}_{br}": filtered[obj][br]
                            for br in obj_branches
                            if br in filtered[obj].fields
                        }
                    )

            compact = dak.zip(subset, depth_limit=1)

            # Write chunk to file
            output_file = output_dir / SKIMMING_OUTPUT_PATTERN.format(chunk=chunk_num)
            uproot.dask_write(
                compact, destination=str(output_file), compute=True, tree_name=output_tree_name
            )
            chunk_num += 1

        logger.info(f"💾 Wrote {chunk_num} skimmed chunks to: {output_dir}")
        return total_events

    def _skim_with_uproot(
        self,
        input_path: str,
        tree: str,
        output_dir: Path,
        configuration,
        is_mc: bool,
        total_events: int,
        step_size: int,
        output_tree_name: str,
    ) -> bool:
        """
        Uproot-specific skimming implementation.
        """
        cut_str = self.get_uproot_cut_string()
        branches = self.build_branches_to_keep(configuration, mode="uproot", is_mc=is_mc)

        iterable = uproot.iterate(
            f"{input_path}:{tree}",
            branches,
            step_size=step_size,
            cut=cut_str,
            library="ak",
            num_workers=1,
        )

        n_chunks = (total_events + step_size - 1) // step_size
        print('\n')
        pbar = tqdm(iterable, total=n_chunks, desc="Processing events", unit="chunk")

        chunk_num = 0
        files_created = 0

        for arrays in pbar:
            # Skip empty chunks
            if len(arrays) == 0:
                logger.debug(f"Chunk {chunk_num} is empty after selection, skipping")
                continue

            # Create output file for this chunk
            output_file = output_dir / SKIMMING_OUTPUT_PATTERN.format(chunk=chunk_num)

            with uproot.recreate(str(output_file)) as output:
                # Get branch types from first chunk
                branch_types = {}
                for branch in arrays.fields:
                    if isinstance(arrays[branch], ak.Array):
                        branch_types[branch] = arrays[branch].type
                    else:
                        branch_types[branch] = np.dtype(arrays[branch].dtype)

                # Create output tree
                output_tree = output.mktree(output_tree_name, branch_types)

                # Write data
                filtered_data = {branch: arrays[branch] for branch in arrays.fields}
                output_tree.extend(filtered_data)


            files_created += 1
            chunk_num += 1

        pbar.close()
        print('\n')

        if files_created > 0:
            logger.info(f"💾 Wrote {files_created} skimmed chunks to: {output_dir}")
            return True
        else:
            logger.info(f"💾 No events passed selection for {input_path}")
            return False


def create_default_skimming_config(output_dir: str = "skimmed/") -> SkimmingConfig:
    """
    Create a default skimming configuration that matches current hardcoded behavior.

    This provides backward compatibility by replicating the current hardcoded
    selections in configurable form.

    Parameters
    ----------
    output_dir : str
        Base output directory for skimmed files

    Returns
    -------
    SkimmingConfig
        Default skimming configuration
    """
    from user.cuts import default_skim_selection  # Import here to avoid circular imports

    # Default uproot cut string matching current hardcoded behavior
    default_uproot_cut = "HLT_TkMu50*(PuppiMET_pt>50)"

    return SkimmingConfig(
        nanoaod_selection={
            "function": default_skim_selection,
            "use": [("Muon", None), ("Jet", None), ("PuppiMET", None), ("HLT", None)]
        },
        uproot_cut_string=default_uproot_cut,
        output_dir=output_dir,
        chunk_size=100_000,
        tree_name="Events"
    )


def process_fileset_with_skimming(config, fileset, cache_dir="/tmp/gradients_analysis/"):
    """
    Process entire fileset with skimming and return processed events for analysis.

    This is the main entry point for skimming that handles the complete workflow:
    - Loop over datasets in fileset
    - Run skimming if enabled
    - Discover and load skimmed files
    - Handle caching of processed events
    - Return events ready for analysis

    Parameters
    ----------
    config : Config
        Analysis configuration
    fileset : dict
        Dictionary mapping dataset names to file and metadata
    cache_dir : str, optional
        Directory for caching processed events

    Returns
    -------
    dict
        Dictionary mapping dataset names to list of (events, metadata) tuples
    """
    processed_datasets = {}

    # Loop over datasets in the fileset
    for dataset, content in fileset.items():
        metadata = content["metadata"]
        metadata["dataset"] = dataset
        process_name = metadata["process"]

        # Skip datasets not explicitly requested in config
        if (req := config.general.processes) and process_name not in req:
            logger.info(f"Skipping {dataset} (process {process_name} not in requested)")
            continue

        processed_events = []

        # Create skimming manager - always available
        skimming_manager = ConfigurableSkimmingManager(config.preprocess.skimming)

        # Log which mode is being used
        if skimming_manager.supports_nanoaod_mode():
            logger.debug(f"Using NanoAOD/DAK skimming mode for {dataset}")
        elif skimming_manager.supports_uproot_mode():
            logger.debug(f"Using uproot skimming mode for {dataset}")

        logger.info(f"🚀 Processing dataset: {dataset}")

        # Loop over ROOT files associated with the dataset
        for idx, (file_path, tree) in enumerate(content["files"].items()):
            # Honour file limit if set in configuration
            if config.general.max_files != -1 and idx >= config.general.max_files:
                logger.info(f"Reached max files limit ({config.general.max_files})")
                break

            # Run skimming if enabled
            if config.general.run_skimming:
                logger.info(f"🔍 Skimming input file: {file_path}")
                skimming_manager.skim(
                    input_path=file_path,
                    tree=tree,
                    dataset=dataset,
                    file_idx=idx,
                    configuration=config,
                    is_mc=(dataset != "data")
                )

            # Discover skimmed files using skimming manager
            skimmed_files = skimming_manager.discover_skimmed_files(dataset, idx)

            # Process each skimmed file
            tree_name = skimming_manager.get_tree_name()
            for skimmed_file in skimmed_files:
                cache_key = hashlib.md5(skimmed_file.encode()).hexdigest()
                cache_file = os.path.join(cache_dir, f"{dataset}__{cache_key}.pkl")

                # Handle caching: process and cache, read from cache, or skip
                if config.general.read_from_cache and os.path.exists(cache_file):
                    logger.info(f"Reading cached events for {skimmed_file}")
                    with open(cache_file, "rb") as f:
                        events = cloudpickle.load(f)
                else:
                    logger.info(f"Processing {skimmed_file}")
                    events = NanoEventsFactory.from_root(
                        skimmed_file, schemaclass=NanoAODSchema,
                        delayed=False,
                    ).events()
                    print(len(events), "MOO")

                    # Cache the events if not reading from cache
                    if not config.general.read_from_cache:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, "wb") as f:
                            cloudpickle.dump(events, f)

                processed_events.append((events, metadata.copy()))

        processed_datasets[dataset] = processed_events

    return processed_datasets
