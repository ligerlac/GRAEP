# Differentiable Z' → tt̄ Analysis Framework

This project is a framework for High-Energy Physics (HEP) analysis that leverages automatic differentiation to optimise analysis selections for maximal statistical significance. It is built on top of the scientific Python ecosystem, including `coffea`, `awkward-array`, and `uproot` for data handling, and `JAX` for gradient-based optimisation.

The example analysis implemented here searches for a Z' boson decaying to a top-antitop quark pair (tt̄).

## Quick Start

This section guides you through running the default analysis configuration provided in the repository.

### 1. Prerequisites

#### Environment Setup

Before running, you must set up the Python environment and install the required dependencies. The recommended method is to use Pixi.

##### Using Pixi

1. Install [Pixi](https://pixi.sh/latest/installation/).
2. Run

```
pixi run start
```

from the top level of the repository.
This will launch a JupyterLab interface with the environment described the by `pixi.toml` Pixi manifest and instantiated in the `pixi.lock` lock file activated.

To interactively use the environment in your local machine's shell run

```
pixi shell
```

from the top level of the repository.

##### Using `conda`

The `environment.yml` file contains all the necessary packages. Create and activate the conda environment with the following commands:

```bash
conda env create -f environment.yml
conda activate zprime_diff_analysis
```

##### Using `pip`

We also provide a `requirements.txt` file, you can still leverage `conda` for environment management:

```bash
# Create a new environment with Python 3.10 (or adjust version as needed)
conda create -n zprime_diff_analysis python=3.10

# Activate the environment
conda activate zprime_diff_analysis

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```
Alternatively, you can use Python’s built-in virtual environment module:

```bash
# Create a virtual environment in a folder named .venv
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Data Skimming

The analysis expects skimmed data files. If you do not have them, you can generate them by running the skimming step. This will download the necessary data from the CERN Open Data Portal and skim it according to the configuration.

```bash
# This command runs only the skimming step to produce skimmed files
python analysis.py general.run_skimming=True general.analysis=skip

# Or run skimming and then analysis in one command
python analysis.py general.run_skimming=True
```

The skimming system provides three modes:

1. **Skim-only mode**: `general.analysis=skip` - Only performs skimming, no analysis
2. **Skim-and-analyse mode**: `general.run_skimming=True` - Skims data then runs analysis
3. **Analysis-only mode**: `general.run_skimming=False` - Uses existing skimmed files for analysis

### 2. Run the Differentiable Analysis

Once the skimmed data is available, you can run the main analysis with a single command:

```bash
python analysis.py
```

### 3. What is Happening?

The default configuration (`user/configuration.py`) is set up to perform a differentiable analysis. The command above will:
1.  **MVA Pre-training**: First, it trains a small, JAX-based neural network to distinguish between `W+jets` and `ttbar` background events. The trained model parameters are saved to disk.
2.  **Differentiable Optimisation**: It then runs the main analysis optimisation loop. The goal is to find the selection cuts that maximise the statistical significance of the Z' signal. At each step, it calculates the gradient of the significance with respect to the cut thresholds (e.g., `met_threshold`, `btag_threshold`) and uses the `optax` optimiser to update them.
3.  **Outputs**: The analysis will produce plots in the `outputs/` directory showing the evolution of the parameters and significance during optimisation, along with the final histograms. The final optimised significance will be printed to the console.

## For Users: What You Need to Know

**This framework separates user-configurable code from framework code:**

- **`user/` directory**: **This is where you make changes for your analysis**
  - `user/configuration.py`: Main configuration file - modify this for your analysis settings
  - `user/cuts.py`: Selection functions - define your analysis regions and cuts here
  - `user/observables.py`: Physics observables - define what variables you want to compute
  - `user/systematics.py`: Systematic variations - define uncertainty sources

- **`analysis/` and `utils/` directories**: **Framework code - you typically don't need to modify these**
  - These contain the analysis infrastructure, plotting utilities, and technical implementation

**To adapt this framework for your analysis, focus on modifying the files in the `user/` directory.** The framework will handle the rest automatically.

## Table of Contents

- [Core Concepts](#core-concepts)
  - [Key Technologies](#key-technologies)
  - [The Differentiable Workflow](#the-differentiable-workflow)
- [How to Implement an Analysis](#how-to-implement-an-analysis)
  - [1. The Configuration File (`user/configuration.py`)](#1-the-configuration-file-userconfigurationpy)
  - [2. Defining Analysis Logic](#2-defining-analysis-logic)
  - [3. Running the Analysis](#3-running-the-analysis)
- [Config-Driven Skimming Framework](#config-driven-skimming-framework)
  - [Dataset Configuration](#dataset-configuration)
  - [Skimming Configuration](#skimming-configuration)
  - [Selection Functions](#selection-functions)
  - [Integration with Main Configuration](#integration-with-main-configuration)
  - [Usage Examples](#usage-examples)
  - [Advanced Features](#advanced-features)
- [Configuration Reference](#configuration-reference)
  - [`general` Block](#general-block)
  - [`preprocess` Block](#preprocess-block)
  - [`datasets` Block](#datasets-block)
  - [`skimming` Block](#skimming-block)
  - [`jax` Block](#jax-block)
  - [`mva` Block](#mva-block)
  - [`channels` Block](#channels-block)
  - [`corrections` and `systematics` Blocks](#corrections-and-systematics-blocks)
  - [Other Top-Level Blocks](#other-top-level-blocks)
- [Under the Hood: The Differentiable Engine](#under-the-hood-the-differentiable-engine)
- [The Differentiable Statistical Model in JAX](#the-differentiable-statistical-model-in-jax)
- [Multi-Variate Analysis (MVA) Integration](#multi-variate-analysis-mva-integration)
- [Extending the Analysis](#extending-the-analysis)
- [Non-Differentiable Analysis](#non-differentiable-analysis)
- [Directory Structure](#directory-structure)
- [[Developer] Building the Documentation](#building-the-documentation)

## Core Concepts

The central idea is to treat the entire analysis chain—from event selection to statistical significance—as a single, differentiable function. The inputs to this function are not just the data, but also the analysis parameters themselves (e.g., selection cut thresholds). By calculating the gradient of the significance with respect to these parameters, we can use optimizers like `optax` to iteratively update them and find the optimal set of cuts.

### Key Technologies
*   **`coffea` & `awkward-array`**: For handling complex, jagged data structures typical in HEP.
*   **`JAX`**: For just-in-time (JIT) compilation and automatic differentiation of Python/NumPy code.
*   **`relaxed`**: A JAX-based library for differentiable statistical models, providing a differentiable approximation of the profile likelihood ratio.
*   **`optax`**: A library of gradient-based optimizers for JAX.

### The Differentiable Workflow

The analysis is orchestrated by the `DifferentiableAnalysis` class in `analysis/diff.py`. The workflow proceeds as follows:

1.  **Preprocessing**: Raw NanoAOD files are skimmed to keep only necessary branches and apply a baseline selection. This is a one-time, non-differentiable step to reduce data volume.
2.  **MVA Pre-training (Optional)**: If configured, a Machine Learning model (e.g., a neural network) is trained on pre-selected data to serve as a powerful discriminator. The weights of this model can themselves become optimizable parameters.
3.  **Event Processing**: For each event, object corrections and systematic variations are applied.
4.  **Differentiable Histogramming**: Instead of making hard cuts, we apply "soft" selections using sigmoid functions. This results in a per-event weight. Histograms are filled using a Kernel Density Estimation (KDE) approach, which is smooth and differentiable.
5.  **Statistical Significance**: The `relaxed` library is used to construct a statistical model from the histograms and compute an asymptotic significance (a differentiable quantity).
6.  **Gradient Calculation**: `JAX` computes the gradient of the significance with respect to all optimizable parameters (cut thresholds, MVA weights, etc.).
7.  **Parameter optimisation**: The `optax` optimizer takes a step in the direction of the gradient to update the parameters, aiming to maximize significance. Steps 4-7 are repeated for a set number of iterations.

---

## How to Implement an Analysis

Implementing a new analysis or modifying the existing one primarily involves three steps:
1.  Modifying the central configuration file.
2.  Defining the analysis logic (observables and selections) in Python functions.
3.  Running the analysis workflow.

### 1. The Configuration File (`user/configuration.py`)

This file is the central hub for defining your entire analysis. The `config` dictionary controls every aspect of the workflow.

**Key Sections:**

*   `general`: Global settings like integrated luminosity, which analysis steps to run (`run_preprocessing`, `run_mva_training`), and file paths.
*   `preprocess`: Defines the branches to keep from the input NanoAOD files.
*   `good_object_masks`: Defines baseline "good" object criteria (e.g., muon pT > 55 GeV) that are applied before any channel-specific logic. This is useful for creating a common object collection for all analysis channels.
*   `baseline_selection`: A hard, non-differentiable selection applied to all events early on.
*   `channels`: Defines the different analysis regions (e.g., signal region, control regions). For each channel, you specify:
    *   `name`: A unique name for the channel.
    *   `selection`: The selection function to apply for this channel.
    *   `observables`: A list of variables to be histogrammed.
    *   `fit_observable`: The specific observable used for the final statistical fit.
*   `ghost_observables`: A powerful feature for computing derived quantities (e.g., ST, ΔR) once and attaching them to the event record. These can then be used by any downstream function.
*   `corrections` & `systematics`: Define object and event-level corrections and systematic uncertainties. The framework supports both `correctionlib` and custom Python functions.
*   `mva`: Configure MVA models. You can define the architecture, features, and training parameters for a JAX or TensorFlow/Keras network.
*   `jax`: **This is the core of the differentiable analysis.**
    *   `params`: A dictionary of all optimizable parameters and their initial values (e.g., `'met_threshold': 50.0`).
    *   `soft_selection`: Points to the Python function that implements your differentiable selection logic.
    *   `param_updates`: Defines clamping functions to keep parameters within physical bounds during optimisation (e.g., `jnp.clip(x + d, 0.0, 3.0)`).
    *   `learning_rates`: Allows you to set custom learning rates for different parameters.

### 2. Defining Analysis Logic

The `config` file points to Python functions that contain the actual physics logic. These typically live in `user/`.

#### Observables (`user/observables.py`)

An observable function takes `awkward-array` collections as input and returns a flat array of the computed values.

**Example: `get_mtt`**
```python
def get_mtt(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
) -> ak.Array:
    # ... logic to calculate four-vectors and sum them ...
    p4tot = p4mu + p4fj + p4j + p4met
    return p4tot.mass
```

#### Selections (`user/cuts.py`)

There are two types of selection functions:

1.  **Standard Selections**: Used for non-differentiable analysis or initial hard cuts. They take `awkward` arrays and should return a `coffea.analysis_tools.PackedSelection` object.

2.  **Differentiable "Soft" Selections**: This is where the magic happens. Instead of returning a boolean mask, this function returns a continuous, per-event **weight** between 0 and 1. This is achieved by replacing hard cuts like `met.pt > 50` with a sigmoid function `jax.nn.sigmoid((met.pt - 50) / scale)`.

**Example: `Zprime_softcuts_jax_workshop`**

This function takes JAX-backed awkward arrays and a `params` dictionary (containing the optimizable parameters defined in the config).

```python
def Zprime_softcuts_jax_workshop(
    muons: ak.Array,
    jets: ak.Array,
    met: ak.Array,
    jet_mass: ak.Array,
    nn,
    params: dict
) -> jnp.ndarray:
    # ...

    # A differentiable cut on MET
    met_cut_weight = jax.nn.sigmoid(
        (ak.to_jax(met) - params["met_threshold"]) / 25.0
    )

    # A differentiable cut on a b-tagging score
    btag_cut_weight = jax.nn.sigmoid(
        (soft_b_counts - params["btag_threshold"]) * 10.0
    )

    # ... other cuts

    # Combine all weights multiplicatively (like a logical AND)
    selection_weight = jnp.prod(jnp.stack([met_cut_weight, btag_cut_weight, ...]))
    return selection_weight
```

### 3. Running the Analysis

With the configuration and functions in place, you can run the analysis using a top-level script.

#### Example `run.py` script
A typical script would:
1.  Load the base configuration from `user/configuration.py`.
2.  Optionally, override configuration settings from the command line.
3.  Construct the fileset of data samples.
4.  Instantiate the `DifferentiableAnalysis` class from `analysis/diff.py`.
5.  Call the main `run_analysis_optimisation` method.

```python
# In a hypothetical run.py
import sys
from analysis.diff import DifferentiableAnalysis
from user.configuration import config
from utils.schema import load_config_with_restricted_cli
from utils.input_files import construct_fileset

if __name__ == "__main__":
    # Load base config and override with CLI args
    cfg = load_config_with_restricted_cli(config, sys.argv[1:])

    fileset = construct_fileset(n_files_max_per_sample=cfg.general.max_files)
    analysis = DifferentiableAnalysis(cfg)
    final_histograms, final_significance = analysis.run_analysis_optimisation(fileset)

    print(f"optimisation complete! Final significance: {final_significance:.3f}")
```

#### Overriding Configuration from the Command Line

You can override certain configuration options directly from the command line using a dot-list format. This is useful for quick tests and batch submissions without modifying the main configuration file.

**Example:**
```python
python run.py general.max_files=10 general.run_systematics=False
```

**Important Limitations:**
For safety and to prevent breaking the analysis logic, only a restricted set of configuration keys can be overridden from the command line. This is because the main Python configuration file contains complex objects like functions and lambdas, which cannot be expressed as simple command-line arguments.

The allowed top-level keys for CLI overrides are:
*   `general`
*   `preprocess`
*   `statistics`

Attempting to override other keys (e.g., `jax.params`) will result in an error. To change these, you must edit the `user/configuration.py` file directly.

## Config-Driven Skimming Framework

The framework provides a configuration-driven approach to data skimming. This section explains how to configure and use the skimming system for your analysis.

### Core Components

The skimming framework has three main parts:

1. **Dataset Configuration**: Define where your data lives and its properties
2. **Skimming Configuration**: Define how to filter and process your data
3. **Selection Functions**: Define the physics logic for event selection

---

### Dataset Configuration

Dataset configuration centralizes information about your data samples, including file locations, cross-sections, and metadata.

#### Basic Dataset Setup

Create a dataset configuration in `user/skim.py`:

```python
# user/skim.py
datasets_config = [
    {
        "name": "signal",
        "directory": "datasets/signal/",
        "cross_section": 1.0,  # pb
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "ttbar_semilep",
        "directory": "datasets/ttbar_semilep/",
        "cross_section": 831.76,  # pb
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "wjets",
        "directory": "datasets/wjets/",
        "cross_section": 61526.7,  # pb
        "tree_name": "Events",
        "weight_branch": "genWeight"
    },
    {
        "name": "data",
        "directory": "datasets/data/",
        "cross_section": 1.0,  # Not used for data
        "tree_name": "Events",
        "weight_branch": "genWeight"  # Not used for data
    }
]
```

#### Advanced Dataset Configuration

For more complex setups, you can specify additional properties:

```python
datasets_config = [
    {
        "name": "zprime_2000",
        "directory": "/eos/cms/store/user/myuser/zprime_M2000/",
        "cross_section": 0.123,
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "metadata": {
            "mass": 2000,
            "campaign": "RunIISummer20UL16",
            "generator": "MadGraph"
        }
    },
    {
        "name": "ttbar_powheg",
        "directory": "/eos/cms/store/user/myuser/ttbar_powheg/",
        "cross_section": 831.76,
        "tree_name": "Events",
        "weight_branch": "genWeight",
        "metadata": {
            "generator": "Powheg+Pythia8",
            "tune": "CP5"
        }
    }
]
```

#### Using Different Storage Systems

The framework supports various file storage patterns:

```python
# Local files
{
    "name": "local_sample",
    "directory": "/home/user/data/sample/",
    "cross_section": 100.0
}

# EOS storage
{
    "name": "eos_sample",
    "directory": "/eos/cms/store/user/username/sample/",
    "cross_section": 50.0
}

# XRootD URLs (will be auto-detected)
{
    "name": "remote_sample",
    "directory": "root://cms-xrd-global.cern.ch//store/mc/sample/",
    "cross_section": 25.0
}
```

---

### Skimming Configuration

Skimming configuration defines how events are filtered and processed. The framework supports two modes:

- **NanoAOD Mode**: Uses Python functions with `PackedSelection` (more flexible)
- **Uproot Mode**: Uses string-based cuts (faster for simple selections)

#### Basic Skimming Setup

```python
# user/skim.py
from user.cuts import default_skim_selection

skimming_config = {
    # NanoAOD mode selection
    "nanoaod_selection": {
        "function": default_skim_selection,
        "use": [("Muon", None), ("Jet", None), ("PuppiMET", None), ("HLT", None)]
    },

    # Uproot mode selection (alternative/complementary)
    "uproot_cut_string": "HLT_TkMu50*(PuppiMET_pt>50)*(nMuon>=1)",

    # Output settings
    "output_pattern": "part{idx}.root",
    "chunk_size": 100_000,
    "tree_name": "Events",
    "weight_branch": "genWeight"
}
```

#### Skimming Configuration Options

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `nanoaod_selection` | `dict` | Python function-based selection | See below |
| `uproot_cut_string` | `str` | String-based cut for uproot | `"pt>50*(eta<2.4)"` |
| `output_dir` | `str` | Output directory (auto-set if None) | `"skimmed_data/"` |
| `output_pattern` | `str` | Filename pattern with `{idx}` placeholder | `"skim_{idx}.root"` |
| `chunk_size` | `int` | Events per processing chunk | `50000` |
| `tree_name` | `str` | ROOT tree name | `"Events"` |
| `weight_branch` | `str` | Event weight branch name | `"genWeight"` |

---

### Selection Functions

Selection functions define the physics logic for event filtering. They use the same functor pattern as the main analysis.

#### Basic Selection Function

```python
# user/cuts.py
import awkward as ak
from coffea.analysis_tools import PackedSelection

def default_skim_selection(muons, jets, met, hlt):
    """
    Basic skimming selection for Z' → tt analysis.

    Parameters
    ----------
    muons : ak.Array
        Muon collection
    jets : ak.Array
        Jet collection
    met : ak.Array
        MET collection
    hlt : ak.Array
        HLT trigger collection

    Returns
    -------
    PackedSelection
        Selection object with named cuts
    """
    selection = PackedSelection()

    # Muon selection: exactly one tight muon
    mu_sel = (
        (muons.pt > 55)
        & (abs(muons.eta) < 2.4)
        & muons.tightId
        & (muons.miniIsoId > 1)
    )
    muon_count = ak.sum(mu_sel, axis=1)
    selection.add("single_muon", muon_count == 1)

    # MET selection
    selection.add("met", met.pt > 50)

    # HLT selection
    selection.add("trigger", hlt.TkMu50)

    # Jet selection: at least 4 jets
    jet_sel = (jets.pt > 30) & (abs(jets.eta) < 2.4) & (jets.jetId >= 4)
    jet_count = ak.sum(jet_sel, axis=1)
    selection.add("jets", jet_count >= 4)

    return selection
```
---

### 4. Integration with Main Configuration

Once you've defined your datasets and skimming configuration, integrate them with your main analysis configuration:

```python
# user/configuration.py
from user.skim import datasets_config, skimming_config

# ... other configuration blocks ...

config = {
    "general": {
        "lumi": 16400,
        "analysis": "diff",
        "run_preprocessing": True,  # Enable skimming
        "output_dir": "outputs/my_analysis/",
        # preprocessed_dir will automatically default to output_dir/skimmed/
    },

    "preprocess": {
        "branches": {
            "Muon": ["pt", "eta", "phi", "mass", "miniIsoId", "tightId", "charge"],
            "Jet": ["btagDeepB", "jetId", "pt", "eta", "phi", "mass"],
            "PuppiMET": ["pt", "phi"],
            "HLT": ["TkMu50"],
            "event": ["genWeight", "run", "luminosityBlock"],
        },
        "mc_branches": {
            "event": ["genWeight"],
        },
        "skimming": skimming_config  # Add your skimming config
    },

    "datasets": datasets_config,  # Add your dataset config

    # ... rest of your configuration ...
}
```

---

### 5. Running with Config-Driven Skimming

#### Basic Usage

You can simply use the framework without any changes:

```python
# This automatically uses your configured datasets and skimming
python run.py
```

---

## Configuration Reference

The analysis is controlled by a central configuration dictionary, typically defined in `user/configuration.py`.
The structure of this configuration is validated against a Pydantic schema in `utils/schema.py`.

Below is a comprehensive reference for all available options, grouped by their top-level key.

---

### `general` Block

Global settings that control the overall workflow of the analysis.

| Parameter            | Type         | Default                     | Description                                                |
|----------------------|--------------|-----------------------------|------------------------------------------------------------|
| `lumi`              | `float`      | *Required*                  | Integrated luminosity in inverse picobarns (/pb).         |
| `weights_branch`    | `str`        | *Required*                  | Branch name containing event weights (e.g. `genWeight`).  |
| `lumifile`          | `str`        | *Required*                  | Path to the JSON file containing certified good luminosity sections (Golden JSON). |
| `analysis`          | `str`        | `"nondiff"`                 | Analysis mode: `"nondiff"`, `"diff"`, or `"both"`.        |
| `max_files`         | `int`        | `-1`                        | Max number of files per dataset. `-1` = unlimited.        |
| `run_preprocessing` | `bool`       | `False`                     | Run NanoAOD skimming and filtering.                       |
| `run_histogramming` | `bool`       | `True`                      | Run histogramming for non-differentiable analysis.        |
| `run_statistics`    | `bool`       | `True`                      | Run statistical analysis step (e.g. `cabinetry` fit).     |
| `run_systematics`   | `bool`       | `True`                      | Process systematic variations for non-differentiable analysis. |
| `run_plots_only`    | `bool`       | `False`                     | Generate plots from cached results only.                  |
| `run_mva_training`  | `bool`       | `False`                     | Run MVA model pre-training.                               |
| `read_from_cache`   | `bool`       | `True`                      | Read preprocessed data from cache if available.           |
| `output_dir`        | `str`        | `"output/"`                 | Root directory for all analysis outputs.                  |
| `preprocessor`      | `str`        | `"uproot"`                  | Preprocessing engine: `"uproot"` or `"dask"`.            |
| `preprocessed_dir`  | `str`        | `None`                      | Directory with pre-processed (skimmed) files.            |
| `cache_dir`         | `str`        | `"/tmp/gradients_analysis/"`| Cache directory for differentiable analysis.             |
| `processes`         | `list[str]`  | `None`                      | Limit analysis to specific processes.                     |
| `channels`          | `list[str]`  | `None`                      | Limit analysis to specific channels.                      |

---

### `preprocess` Block

Settings for the initial data skimming and filtering step.

| Parameter        | Type       | Default     | Description                                         |
|------------------|------------|-------------|-----------------------------------------------------|
| `branches`       | `dict`     | *Required*  | Mapping of collection names to branch lists.        |
| `ignore_missing` | `bool`     | `False`     | Ignore missing branches if `True`.                  |
| `mc_branches`    | `dict`     | *Required*  | Additional branches for MC samples.                 |
| `skimming`       | `dict`     | `None`      | Skimming configuration (see `skimming` block below). |

---

### `datasets` Block

List of dataset configurations defining data sample properties.

| Parameter        | Type       | Default     | Description                                         |
|------------------|------------|-------------|-----------------------------------------------------|
| `name`          | `str`      | *Required*  | Unique dataset identifier.                          |
| `directory`     | `str`      | *Required*  | Path to dataset files.                             |
| `cross_section` | `float`    | *Required*  | Cross-section in picobarns (pb).                  |
| `tree_name`     | `str`      | `"Events"`  | ROOT tree name.                                    |
| `weight_branch` | `str`      | `"genWeight"` | Event weight branch name.                        |
| `metadata`      | `dict`     | `{}`        | Additional dataset metadata.                       |

---

### `skimming` Block

Configuration for the data skimming step (part of `preprocess` block).

| Parameter            | Type       | Default           | Description                                    |
|----------------------|------------|-------------------|------------------------------------------------|
| `nanoaod_selection`  | `dict`     | `None`           | NanoAOD mode selection configuration.         |
| &nbsp;&nbsp;↳ `function` | `Callable` | *Required*   | Selection function to apply.                   |
| &nbsp;&nbsp;↳ `use`      | `list[str]` | *Required*  | Input variables for the function.              |
| `uproot_cut_string`  | `str`      | `None`           | Uproot mode cut string.                       |
| `output_dir`         | `str`      | `None`           | Output directory (auto-set if None).         |
| `output_pattern`     | `str`      | `"part{idx}.root"` | Output filename pattern.                    |
| `chunk_size`         | `int`      | `100000`         | Events per processing chunk.                  |
| `tree_name`          | `str`      | `"Events"`       | ROOT tree name.                               |
| `weight_branch`      | `str`      | `"genWeight"`    | Event weight branch name.                     |

---

### `jax` Block

Configuration for the differentiable analysis workflow.

| Parameter                | Type         | Default     | Description                                                        |
|--------------------------|--------------|-------------|--------------------------------------------------------------------|
| `soft_selection`         | `dict`      | *Required*  | Differentiable selection function.                                 |
| &nbsp;&nbsp;↳ `function` | `Callable`  | *Required*  | Selection function to apply.                                       |
| &nbsp;&nbsp;↳ `use`      | `list[str]` | *Required*  | Input variables passed to `function`.                              |
| `params`                 | `dict`      | *Required*  | Optimizable parameters (e.g. `{'met_threshold': 50.0}`).          |
| `optimize`               | `bool`      | `True`      | Run gradient-based optimisation if `True`.                         |
| `learning_rate`          | `float`     | `0.01`      | Default optimizer learning rate.                                   |
| `max_iterations`         | `int`       | `50`        | Number of optimisation steps.                                      |
| `param_updates`          | `dict`      | `{}`        | Parameter-specific clamping functions.                             |
| &nbsp;&nbsp;↳ `param_name` | `Callable`| -           | `(old_value, delta) -> new_value` function.                        |
| `learning_rates`         | `dict`      | `None`      | Parameter-specific learning rates.                                 |
| `explicit_optimisation`  | `bool`      | `False`     | Use manual optimisation loop if `True`.                            |

---

### `mva` Block

List of MVA model configurations.

| Parameter               | Type            | Default     | Description                                                       |
|-------------------------|-----------------|-------------|-------------------------------------------------------------------|
| `name`                  | `str`          | *Required*  | Unique model name.                                                |
| `framework`             | `str`          | *Required*  | `"jax"` or `"keras"`.                                            |
| `learning_rate`         | `float`       | `0.01`      | Learning rate for pre-training.                                  |
| `grad_optimisation`     | `dict`        | `{}`        | MVA optimisation settings.                                       |
| &nbsp;&nbsp;↳ `optimise`    | `bool`    | `False`     | Include MVA in global optimisation.                              |
| &nbsp;&nbsp;↳ `learning_rate` | `float` | `0.001`    | Learning rate for MVA in optimisation.                           |
| `layers`                | `list[dict]` | *Required*  | Network architecture layers.                                     |
| &nbsp;&nbsp;↳ `ndim`    | `int`        | *Required*  | Number of nodes.                                                 |
| &nbsp;&nbsp;↳ `activation` | `str`     | *Required*  | Activation function.                                             |
| &nbsp;&nbsp;↳ `weights` | `str`        | *Required*  | Name for weights parameter.                                      |
| &nbsp;&nbsp;↳ `bias`    | `str`        | *Required*  | Name for bias parameter.                                         |
| `loss`                  | `Callable` or `str` | *Required* | Loss function (callable for JAX, string for Keras).             |
| `features`              | `list[dict]` | *Required*  | Input features for the model.                                   |
| &nbsp;&nbsp;↳ `name`    | `str`        | *Required*  | Feature name.                                                    |
| &nbsp;&nbsp;↳ `function`| `Callable`   | *Required*  | Function to compute the feature.                                 |
| &nbsp;&nbsp;↳ `use`     | `list[str]` | *Required*  | Input dependencies for the feature.                              |
| `classes`               | `list`       | *Required*  | Target classes (e.g. `["wjets", {"ttbar": [...]}]`).            |
| `balance_strategy`      | `str`       | `"undersample"` | `"none"`, `"undersample"`, `"oversample"`, `"class_weight"`.  |
| `random_state`          | `int`       | `42`        | Random seed for reproducibility.                                 |
| `epochs`                | `int`       | `1000`      | Pre-training epochs.                                             |
| `batch_size`            | `int`       | `32`        | Training batch size.                                             |
| `validation_split`      | `float`     | `0.2`       | Fraction for validation.                                         |
| `log_interval`          | `int`       | `100`      | Log frequency during training.                                   |

---

### `channels` Block

List of analysis channels or regions.

| Parameter            | Type          | Default     | Description                                                      |
|----------------------|---------------|-------------|------------------------------------------------------------------|
| `name`              | `str`         | *Required*  | Channel name (e.g. `"signal_region"`).                          |
| `observables`       | `list[dict]`  | *Required*  | Observables to histogram.                                       |
| &nbsp;&nbsp;↳ `name`      | `str`    | *Required*  | Observable name.                                                |
| &nbsp;&nbsp;↳ `binning`   | `tuple`  | *Required*  | Histogram binning.                                              |
| &nbsp;&nbsp;↳ `function`  | `Callable` | *Required* | Function to compute observable.                                 |
| &nbsp;&nbsp;↳ `use`       | `list[str]` | *Required* | Inputs to the observable function.                              |
| `fit_observable`    | `str`         | *Required*  | Observable used in statistical fit.                             |
| `selection`         | `dict`       | `None`      | Channel selection function.                                     |
| &nbsp;&nbsp;↳ `function` | `Callable` | *Required* | Selection function.                                             |
| &nbsp;&nbsp;↳ `use`      | `list[str]` | *Required* | Inputs to selection function.                                   |
| `use_in_diff`       | `bool`       | `False`     | Include in differentiable analysis.                             |

---

### `corrections` and `systematics` Blocks

| Parameter            | Type       | Default        | Description                                           |
|----------------------|------------|----------------|-------------------------------------------------------|
| `name`              | `str`     | *Required*     | Correction or systematic name.                       |
| `type`              | `str`     | *Required*     | `"object"` or `"event"`.                             |
| `op`                | `str`     | `"mult"`       | Operation: `"mult"` or `"add"`.                      |
| `target`           | `tuple` or `list` | `None` | Object/variable to modify (e.g. `("Jet", "pt")`).   |
| `use`              | `list`    | `[]`          | Inputs required for the variation function.          |
| **Corrections Only** |           |                |                                                       |
| &nbsp;&nbsp;↳ `file`        | `str`     | *Required*     | Path to correction file.                              |
| &nbsp;&nbsp;↳ `key`         | `str`     | `None`         | Key within the file.                                  |
| &nbsp;&nbsp;↳ `use_correctionlib` | `bool` | `True` | Use `correctionlib`.                                 |
| &nbsp;&nbsp;↳ `transform`   | `Callable` | `None`        | Transform arguments before evaluation.               |
| &nbsp;&nbsp;↳ `up_and_down_idx` | `list[str]` | `["up", "down"]` | Variation labels in file.                     |
| **Systematics Only** |           |                |                                                       |
| &nbsp;&nbsp;↳ `up_function`    | `Callable` | `None` | Function for "up" variation.                        |
| &nbsp;&nbsp;↳ `down_function`  | `Callable` | `None` | Function for "down" variation.                      |
| &nbsp;&nbsp;↳ `symmetrise`     | `bool`    | `False`| Auto-generate "down" from "up" (not implemented).   |

---

## Under the Hood: The Differentiable Engine

The core of the differentiable workflow is the `_run_traced_analysis_chain` method in `analysis/diff.py`. This function is what `JAX` traces and differentiates.

```python
def _run_traced_analysis_chain(
    self,
    params: dict[str, Any],
    processed_data_events: dict,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    # ...
    # 1. Collect histograms for all processes using the current `params`
    histograms_by_process = self._collect_histograms(...)

    # 2. Calculate significance from these histograms
    significance, mle_params = self._calculate_significance(histograms_by_process, params["fit"])
    # ...
    return significance, mle_params
```

The optimisation loop in `run_analysis_optimisation` then does the following:

```python
# Define the objective function to be *maximized* (significance)
# Note: Optimizers typically *minimize*, so we would differentiate the *negative* significance.
# The `relaxed` library handles this internally.
def objective(params):
    return self._run_traced_analysis_chain(params, processed_data)

# Get the gradient of the objective function w.r.t. the parameters
gradients = jax.grad(objective)(all_parameters)

# Use an optimizer to update the parameters
# solver = OptaxSolver(fun=objective, opt=tx, ...)
# new_parameters, state = solver.update(parameters, state)
```

This loop continues until the significance converges or a maximum number of iterations is reached.

## The Differentiable Statistical Model in JAX

A key innovation of this framework is its end-to-end differentiable statistical model. This model is constructed in JAX, allowing the entire analysis—from event selection to statistical inference—to be differentiated.

### 1. Differentiable Histograms

Instead of standard histograms with hard bin counts, the analysis produces "soft" histograms using a Kernel Density Estimation (KDE) approach. For each event, instead of adding `1` to a single bin, a Gaussian kernel is placed at the event's observable value. The histogram's bin contents are then the integral of all event kernels over that bin's range. This process is smooth and differentiable with respect to both the event weights (from soft selections) and the observable values themselves.

### 2. Building the Statistical Model

The statistical model is built upon these differentiable histograms. The goal is to construct a likelihood function `L(data | params)` that can be maximized to find the best-fit parameters.

The core of the model is the prediction for the expected number of events (`expected_yields`) in each histogram bin for each analysis channel. This is a function of the model parameters. In the current implementation (`utils/jax_stats.py`), the model is simplified and includes two global scalar parameters:
*   `mu` ($\mu$): The signal strength parameter. $\mu=0$ corresponds to the background-only hypothesis, and $\mu=1$ corresponds to the nominal signal hypothesis.
*   `norm_ttbar_semilep` ($\kappa_{t\bar{t}}$): A normalisation factor for the `ttbar_semilep` background process, applied uniformly across all channels.

For a single channel, the expected yield $\lambda_i$ in bin $i$ is the sum of all signal and background processes:

$\lambda_i(\mu, \kappa_{t\bar{t}}) = \mu \cdot \lambda_{\text{sig}, i} + \kappa_{t\bar{t}} \cdot \lambda_{t\bar{t}, i} + \sum_{b \in \text{other bkg}} \lambda_{b, i}$

Note that in this simplified model, systematic uncertainties are not incorporated via nuisance parameters in the differentiable fit. The framework can produce systematically-varied histograms, but they are used in the non-differentiable analysis path.

### 3. Likelihood and Hypothesis Testing with `relaxed`

With the model for `expected_yields` defined, the final step is to perform statistical inference. This is where the `relaxed` library is used.

1.  **Likelihood Construction**: `relaxed` takes the JAX model of expected yields and the observed data histograms to construct the full likelihood function. For this simplified model, the likelihood is a product of Poisson probability mass functions over all bins:

    $L(\text{data} | \mu, \kappa_{t\bar{t}}) = \prod_{c \in \text{channels}} \prod_{i \in \text{bins}} \text{Pois}(N_{\text{obs}, c, i} | \lambda_{c, i}(\mu, \kappa_{t\bar{t}}))$

    Since the model in `utils/jax_stats.py` does not include systematic uncertainties as nuisance parameters, there are no constraint terms in the likelihood.

2.  **Differentiable Hypothesis Test**: The main goal is to test for the presence of a signal. This is done using a profile likelihood ratio test statistic, $q_0$. `relaxed.infer.hypotest` performs this test. It finds the values of the model parameters (in this case, $\kappa_{t\bar{t}}$) that maximize the likelihood for a given $\mu$ (profiling) and then computes $q_0$. The significance is then $Z = \sqrt{q_0}$.

This means we can compute the gradient of the final significance `Z` with respect to any parameter in the chain, including the selection cut thresholds (`met_threshold`, etc.) and even the weights of a JAX-based MVA. This gradient is what drives the optimisation.

## Multi-Variate Analysis (MVA) Integration

The framework includes support for integrating Machine Learning models (MVAs) into the analysis, with implementations for both JAX and TensorFlow/Keras. This allows for the use of non-linear discriminants that can be pre-trained and, in the case of JAX models, optimized *in-situ* with the rest of the analysis.

The core logic is handled by the `JAXNetwork` and `TFNetwork` classes in `utils/mva.py`.

### MVA Implementations

#### `JAXNetwork`
This is a neural network implementation written purely in JAX, providing deep integration with the differentiable analysis workflow.
*   **Explicit Parameter Management**: Unlike frameworks that encapsulate model weights, the `JAXNetwork` manages its weights and biases in a simple Python dictionary. This transparency is key to its integration. Parameter names follow a convention (e.g., `__NN_my_model_W1`) that allows the framework to automatically identify them.
*   **End-to-End optimisation**: When MVA optimisation is enabled in the configuration (`grad_optimisation.optimise: True`), the network's parameters are added to the global set of variables that the main optimizer tunes. This means the optimizer can simultaneously adjust the MVA's weights to improve signal/background separation *and* tune the analysis selection cuts, all to directly maximize the final statistical significance.
*   **Full Control**: The from-scratch implementation gives full control over the network's forward pass, loss function, and training loop, all within the JAX ecosystem.
*   **Configuration**: The network architecture (layers, activations) is defined in `user/configuration.py`. Activations are provided as Python `lambda` functions, allowing for custom, non-standard activation functions if needed.

#### `TFNetwork`
This class provides a wrapper around a standard `tf.keras.Sequential` model.
*   **Leverage Keras**: It allows you to use the rich and mature Keras API for building and training models.
*   **Pre-training Only**: The primary use case is to pre-train a powerful discriminator. The trained model is then used to compute a score for each event, which is used as a static input feature in the main analysis. The weights of a TF/Keras model are **not** part of the global significance optimisation.
*   **Configuration**: The architecture is defined in the configuration file, with activations specified as strings (e.g., `"relu"`, `"tanh"`).

### The MVA Workflow: Pre-training and In-situ Optimisation

The framework handles MVAs in a two-stage process: an initial, one-off pre-training phase, followed by an optional, continuous optimisation phase that happens alongside the main analysis optimisation.

#### Stage 1: Pre-training

*   **When**: This happens once at the start of the analysis if `general.run_mva_training` is `True`.
*   **Data**: The framework allows for a completely separate object definition for MVA training. In `config.good_object_masks`, you can define an `mva` key with different object selection criteria than the `analysis` key. This is useful for training on a broader, less-biased dataset.
*   **Process**:
    1.  During event processing (`_prepare_data`), two parallel sets of object collections are created: one for the main analysis and one for MVA training, each with its own "good object" masks applied.
    2.  After all files are processed, the MVA-specific data is passed to `_run_mva_training`.
    3.  The models (both JAX and Keras) are trained on this dedicated dataset.
    4.  The resulting trained model (for Keras) or parameters (for JAX) are saved to disk.

#### Stage 2: Inference and In-situ Optimisation (Fine-tuning)

*   **When**: This happens at every step of the main gradient-based optimisation loop.
*   **Inference (On-the-fly)**:
    *   The MVA *instance* (containing the forward pass logic) and its input *features* are attached to a special collection in the event record.
    *   The differentiable selection function (e.g., `Zprime_softcuts_jax_workshop`) must be designed to accept this collection and the global `params` dictionary as inputs.
    *   Inside this traced function, the MVA's `forward_pass` is called on-the-fly, using the current state of the MVA's weights from the `params` dictionary. This ensures that the entire calculation, from MVA inputs to score, is part of the JAX computation graph.
*   **In-situ Optimisation (JAX only)**:
    *   If an MVA is configured with `framework: "jax"` and `grad_optimisation.optimise: True`, its pre-trained parameters are included in the set of globally optimisable parameters.
    *   Because the MVA's forward pass is executed on-the-fly within the main traced function, its output (the MVA score) is fully differentiable with respect to its weights and biases.
    *   The gradient of the final statistical significance is therefore also calculated with respect to these MVA parameters.
    *   The `optax` optimiser updates the MVA weights at each step, effectively "fine-tuning" the MVA to directly maximise the analysis significance, alongside all other selection cuts.
    *   Keras models are used for inference only; their weights are not optimised during this stage.

### Extending and Adding a New MVA

Adding a new MVA to the analysis is a configuration-driven process:

1.  **Add to Config**: Create a new dictionary entry in the `mva` list in `user/configuration.py`. Give it a unique `name`.

2.  **Define Architecture & Framework**:
    *   Set `framework` to `"jax"` or `"keras"`.
    *   Define the `layers` list, specifying the dimensions, activation functions, and names for weights/biases for each layer.
    *   Define the `loss` function.

3.  **Define Input Features**:
    *   In the `features` list for your MVA, define each input variable.
    *   Each feature needs a `name`, a `function` to compute it (e.g., `lambda mva: mva.n_jet`), and a `use` key specifying the inputs to that function. The framework automatically computes "ghost observables" first, so you can define features that depend on them.

4.  **Use the MVA in Selection**:
    *   The framework will automatically train the MVA (if `run_mva_training` is `True`) and compute its output score for every event.
    *   This score is attached to a special object collection named after your MVA (e.g., `wjets_vs_ttbar_nn`).
    *   You can then use this score in your differentiable selection function in `user/cuts.py` just like any other variable.

5.  **Enable Gradient optimisation (JAX only)**:
    *   To make the JAX MVA's weights optimizable, set `grad_optimisation.optimise: True` in its configuration.
    *   The framework will automatically find all parameters with the `__NN` prefix and add them to the set of variables that the optimizer will tune. You can even set a custom learning rate for the MVA weights.

## Extending the Analysis

#### Adding a New Optimizable Parameter

1.  **Add to Config**: Add a new key-value pair to `config["jax"]["params"]`.
    ```python
    "params": {
        'met_threshold': 50.0,
        'btag_threshold': 0.5,
        'my_new_cut': 100.0, # <-- Add new parameter
    },
    ```
2.  **Use in Soft Selection**: Use `params["my_new_cut"]` in your soft selection function in `user/cuts.py`.
3.  **(Optional) Add a Clamp**: Add a rule for your new parameter in `config["jax"]["param_updates"]` to keep it within a sensible range.

#### Adding a New Systematic Uncertainty

1.  **Add to Config**: Add a new dictionary to the `config["systematics"]` list.
2.  **Define Logic**:
    *   If it's a simple scale factor, you can define the `up_function` and `down_function` directly in the config (e.g., `lambda: 1.05`).
    *   For more complex variations, define a function in `user/systematics.py` that takes an object collection and returns a per-object weight.
    *   For `correctionlib`-based uncertainties, ensure the `file` and `key` are specified correctly.
3.  **Specify Target**: Define the `target` (which object and variable are affected) and the `op` (how the variation is applied, e.g., `mult` or `add`).

The framework will automatically propagate these systematics through the non-differentiable analysis path to produce varied histograms for tools like `cabinetry`.

## Non-Differentiable Analysis

Alongside the differentiable path, the framework fully supports a traditional, non-differentiable analysis via the `NonDiffAnalysis` class in `analysis/nondiff.py`. This path uses standard hard cuts and fills `hist` objects, which can then be used with tools like `cabinetry` for statistical inference. You can control which analysis runs via the `analysis` key in `config.general`.

---

## Directory Structure

```
├── user/                    # USER-CONFIGURABLE MODULES - Modify these for your analysis
│   ├── __init__.py         # Package initialization
│   ├── configuration.py    # Main configuration file for the analysis
│   ├── cuts.py            # Selection logic (both hard and soft/differentiable)
│   ├── observables.py     # Physics observables and reconstruction functions
│   └── systematics.py     # Systematic variation functions
├── analysis/               # FRAMEWORK CODE - Core analysis classes and pipeline logic
│   ├── base.py            # Base class with common analysis logic (corrections, etc.)
│   ├── diff.py            # Implements the full differentiable analysis workflow
│   └── nondiff.py         # Implements a traditional, non-differentiable analysis
├── utils/                  # FRAMEWORK CODE - Supporting utility functions
│   ├── mva.py             # MVA (neural network) model definitions and training logic
│   ├── schema.py          # Pydantic schemas for validating the configuration
│   ├── plot.py            # Plotting utilities and visualization functions
│   ├── stats.py           # Statistical analysis functions
│   ├── tools.py           # General utility functions
│   ├── input_files.py     # File handling utilities
│   ├── output_files.py    # Output management utilities
│   └── ...                # Other helper utilities
├── cabinetry/
│   └── ...                # Configuration for the `cabinetry` statistical tool
├── corrections/
│   └── ...                # Correction files (e.g., from `correctionlib`)
└── README.md
```

### Key Design Principle

The framework separates **user-configurable modules** (`user/`) from **framework code** (`analysis/`, `utils/`):

- **`user/` directory**: Contains modules that users should modify for their specific analysis needs
- **`analysis/` and `utils/` directories**: Contains framework code that provides the analysis infrastructure

This separation ensures that users can focus on physics configuration while the framework handles the technical implementation details.

---

## Logical Flow of the Differentiable Analysis

Understanding the logical flow of the differentiable analysis helps users see how their configuration choices in the `user/` directory affect the overall workflow. Here's a step-by-step breakdown:

### 1. Initialization and Configuration Loading
```
user/configuration.py → Analysis Setup
```
- The analysis starts by loading your configuration from `user/configuration.py`
- This defines all analysis parameters, observables, cuts, and optimization settings
- The framework validates the configuration against the schema in `utils/schema.py`

### 2. Data Preprocessing (One-time Setup)
```
Raw NanoAOD → Preprocessing → Cached Data
```
- If `general.run_preprocessing=True`, raw NanoAOD files are skimmed
- Only branches specified in `config.preprocess.branches` are kept
- Baseline selections from `config.baseline_selection` are applied
- Results are cached for faster subsequent runs

### 3. MVA Pre-training (Optional)
```
Cached Data → Feature Extraction → Model Training → Saved Model
```
- If `general.run_mva_training=True`, neural networks are pre-trained
- Features defined in `config.mva[].features` are computed using functions from `user/observables.py`
- Models are trained to distinguish between background processes
- Trained parameters are saved and later used in the main analysis

### 4. Event Processing Loop
```
For each event batch:
  Raw Objects → Corrections → Good Objects → Ghost Observables
```
- Object corrections from `config.corrections` are applied
- "Good object" masks from `config.good_object_masks` filter objects
- "Ghost observables" from `config.ghost_observables` are computed using `user/observables.py`
- This creates an event records with all necessary variables

### 5. Differentiable Selection (The Core Loop)
```
For each optimization step:
  Events → Soft Cuts → Event Weights → Histograms → Significance → Gradients → Parameter Updates
```
- **Soft Cuts**: Your selection function from `user/cuts.py` (e.g., `Zprime_softcuts_jax_workshop`) is called
- **Event Weights**: Instead of hard cuts, sigmoid functions produce continuous weights (0-1) per event
- **Histograms**: Events are binned using Kernel Density Estimation (KDE) - smooth and differentiable
- **Significance**: Statistical model computes discovery significance using the `relaxed` library
- **Gradients**: JAX computes gradients of significance w.r.t. all parameters in `config.jax.params`
- **Updates**: Optimizer (optax) updates parameters to maximize significance

### 6. Parameter Flow Through the System
```
config.jax.params → Selection Function → Event Weights → Final Significance
     ↑                                                           ↓
Parameter Updates ←← Gradients ←← Statistical Model ←← Histograms
```
- Parameters you define in `config.jax.params` (e.g., `met_threshold: 50.0`) flow into your selection function
- Your selection function in `user/cuts.py` uses these parameters in sigmoid cuts
- The resulting event weights affect histogram shapes
- Changes in histograms affect the final statistical significance
- Gradients flow backward through this entire chain to update parameters

### 7. Multi-Channel Analysis
```
For each channel in config.channels:
  Selection → Observable Computation → Histogramming → Statistical Combination
```
- Each analysis channel (signal region, control regions) is processed
- Channel-specific selections from `config.channels[].selection` are applied
- Observables from `config.channels[].observables` are computed using `user/observables.py`
- All channels contribute to the final statistical model


### Key Insight: Your Role as a User
- **Configuration (`user/configuration.py`)**: You define what gets optimized and how
- **Observables (`user/observables.py`)**: You define what physics quantities to compute
- **Cuts (`user/cuts.py`)**: You define how events are selected (both hard and soft cuts)
- **Systematics (`user/systematics.py`)**: You define uncertainty sources

The framework handles the technical details (JAX tracing, gradient computation, optimisation) while you focus on the physics logic. Every function you write in the `user/` directory becomes part of a fully differentiable computation graph that can be optimised end-to-end.

---

## Building the Documentation

This project uses Sphinx to generate documentation from the source code's docstrings and other reStructuredText files. The documentation is hosted on Read the Docs.

### Building Locally

To build and view the documentation on your local machine, follow these steps:

1.  **Install Dependencies**:
    The documentation dependencies are listed in `docs/requirements.txt`. You can install them using `pip`. It is recommended to do this within the project's conda environment to keep dependencies organized.
    ```bash
    pip install -r docs/requirements.txt
    ```

2.  **Build the HTML pages**:
    Navigate to the `docs/` directory and use the provided `Makefile` to build the documentation.
    ```bash
    cd docs
    make html
    ```

3.  **View the Documentation**:
    The generated HTML files will be in the `docs/build/html/` directory. You can open the main page in your browser (e.g., by navigating to the file path in your browser's address bar or using a command like `open` on macOS or `xdg-open` on Linux).

### Deploying to Read the Docs

The repository is configured to automatically build and deploy the documentation to Read the Docs on every push to the main branch. This process is controlled by the `.readthedocs.yaml` file at the root of the repository.

If you have forked this repository and wish to set up your own Read the Docs deployment:

1.  **Sign up/in to Read the Docs**: Go to readthedocs.org and create an account or log in. Make sure your GitHub account is connected.
2.  **Import the Repository**: From your Read the Docs dashboard, click on "Import a Project" and select your GitHub repository fork.
3.  **Configuration**: Read the Docs will automatically detect the `.readthedocs.yaml` file and configure the build process. You should not need to change any settings in the Read the Docs project configuration page.
4.  **Trigger a Build**: The first build should trigger automatically after importing. Subsequent builds will be triggered by pushing new commits to your repository's default branch. You can monitor the build status and logs in your Read the Docs project dashboard.
```
