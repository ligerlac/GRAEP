import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tabulate import tabulate

from utils.datasets import ConfigurableDatasetManager, create_default_dataset_config

# Configure module-level logger
logger = logging.getLogger(__name__)


def construct_fileset(
    max_files_per_sample: int,
    preprocessor: str = "uproot",
    json_path: Optional[Union[str, Path]] = None,
    dataset_manager: Optional[ConfigurableDatasetManager] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a structured fileset mapping for physics analyses including
    file paths and metadata.

    This function reads dataset definitions from a JSON file and constructs a nested
    dictionary where each key is "<process>__<variation>" and values contain:
      - files: a mapping of file or glob patterns to ROOT TTrees
      - metadata: information on event counts, cross-sections, etc.

    Parameters
    ----------
    max_files_per_sample : int
        Maximum number of files to include for each sample. Use -1 to include all files.
    preprocessor : str, optional
        Type of file access to prepare. Supported values:
        - "uproot": use glob patterns for directory-level access
        - other: list each file individually
        Default is "uproot".
    json_path : str or Path, optional
        Path to the JSON configuration file specifying samples, variations, and file lists.
        If None, uses dataset_manager.config.metadata_output_dir.
    dataset_manager : ConfigurableDatasetManager, optional
        Dataset manager containing all configuration for cross-sections, paths, etc.
        If None, creates a default dataset manager.

    Returns
    -------
    fileset : dict
        Nested mapping where each key "<process>__<variation>" maps to:
        - files (dict): {file_path_or_pattern: tree_name}
        - metadata (dict): {
            "process": str,
            "variation": str,
            "nevts": int,
            "nevts_wt": float,
            "xsec": float,
        }

    Raises
    ------
    ValueError
        If max_files_per_sample is invalid.
    FileNotFoundError
        If the JSON configuration file does not exist.
    KeyError
        If a process is not found in the dataset manager configuration.
    JSONDecodeError
        If the JSON file cannot be parsed.
    """
    # Validate inputs
    if max_files_per_sample < -1:
        raise ValueError(
            f"max_files_per_sample must be -1 or non-negative; "
            f"got {max_files_per_sample}"
        )

    # Use provided dataset manager or create default
    if dataset_manager is None:
        dataset_config = create_default_dataset_config()
        dataset_manager = ConfigurableDatasetManager(dataset_config)

    # Use provided json_path or get from dataset manager config
    if json_path is None:
        json_path = dataset_manager.config.metadata_output_dir

    json_file = Path(json_path) / "nanoaods.json"
    if not json_file.is_file():
        raise FileNotFoundError(f"Dataset JSON file not found: {json_file}")

    # Load dataset definitions
    with json_file.open("r") as f:
        dataset_info = json.load(f)

    fileset: Dict[str, Dict[str, Any]] = {}

    # Iterate over each process and its systematic variations
    for process_name, variations in dataset_info.items():
        # Validate that process is configured in dataset manager
        if not dataset_manager.validate_process(process_name):
            raise KeyError(f"Process '{process_name}' not found in dataset manager configuration")

        for variation_name, info in variations.items():
            # Extract raw file entries
            raw_entries = info.get("files", [])

            # Limit number of files if requested
            if max_files_per_sample != -1:
                raw_entries = raw_entries[:max_files_per_sample]

            # Compute total event counts
            total_events = sum(entry.get("nevts", 0) for entry in raw_entries)
            total_weighted = sum(
                entry.get("nevts_wt", 0.0) for entry in raw_entries
            )

            # Get cross-section and tree name from dataset manager
            xsec = dataset_manager.get_cross_section(process_name)
            tree_name = dataset_manager.get_tree_name(process_name)

            # Prepare metadata dict
            metadata = {
                "process": process_name,
                "variation": variation_name,
                "nevts": total_events,
                "nevts_wt": total_weighted,
                "xsec": xsec,
            }

            # Explicit file listings for other preprocessors
            file_map = {entry.get("path", ""): tree_name for entry in raw_entries}

            key = f"{process_name}__{variation_name}"
            fileset[key] = {"files": file_map, "metadata": metadata}

            logger.debug(
                f"Added fileset entry: {key} with {len(file_map)} files"
            )

    logger.info(f"Constructed fileset with {len(fileset)} entries.")

    # --- Add summary table ---
    summary_data = []
    headers = ["Key", "Process", "Variation", "# Files", "Cross-section"]
    for key, content in fileset.items():
        process = content["metadata"]["process"]
        variation = content["metadata"]["variation"]
        num_files = len(content["files"])
        xsec = content["metadata"]["xsec"]
        summary_data.append([key, process, variation, num_files, f"{xsec:.2f} pb"])

    # Sort by key for consistent output
    summary_data.sort(key=lambda x: x[0])

    logger.info(
        "Fileset Summary:\n"
        + tabulate(summary_data, headers=headers, tablefmt="grid")
    )

    return fileset


# End of module
