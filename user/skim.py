"""
Skimming configuration and selection functions for the Z-prime ttbar analysis.

This module contains all skimming-related configuration including:
- Dataset definitions with cross-sections and paths
- Skimming selection functions
- Skimming configuration parameters
"""

from user.cuts import default_skim_selection

# ==============================================================================
#  Dataset Configuration
# ==============================================================================

datasets_config = [
    {
        "name": "signal",
        "directory": "datasets/signal/m2000_w20/",
        "cross_section": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "ttbar_semilep",
        "directory": "datasets/ttbar_semilep/",
        "cross_section": 831.76 * 0.438,  # 364.35
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "ttbar_had",
        "directory": "datasets/ttbar_had/",
        "cross_section": 831.76 * 0.457,  # 380.11
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "ttbar_lep",
        "directory": "datasets/ttbar_lep/",
        "cross_section": 831.76 * 0.105,  # 87.33
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "wjets",
        "directory": "datasets/wjets/",
        "cross_section": 61526.7,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "data",
        "directory": "datasets/data/",
        "cross_section": 1.0,
        "file_pattern": "*.txt",
        "tree_name": "Events",
        "weight_branch": "genWeight"
    }
]

# ==============================================================================
#  Dataset Manager Configuration
# ==============================================================================

dataset_manager_config = {
    "datasets": datasets_config,
    "metadata_output_dir": "output_jax_nn_standalone/skimmed/nanoaods_jsons/"
}

# ==============================================================================
#  Skimming Configuration
# ==============================================================================

skimming_config = {
    "nanoaod_selection": {
        "function": default_skim_selection,
        "use": [("Muon", None), ("Jet", None), ("PuppiMET", None), ("HLT", None)]
    },
    "uproot_cut_string": "HLT_TkMu50*(PuppiMET_pt>50)",
    "output_dir": None,  # Will default to {general.output_dir}/skimmed/
    "chunk_size": 100_000,
    "tree_name": "Events",
}
