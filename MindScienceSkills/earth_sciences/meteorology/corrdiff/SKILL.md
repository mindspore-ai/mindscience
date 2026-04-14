---
name: corrdiff
description: corrdiff (Correction Diffusion) is a generative correction diffusion model for km-scale atmospheric downscaling developed by NVIDIA. Use this model when you need to perform weather downscaling - transforming low-resolution weather data (e.g., ERA5 at ~25km) to high-resolution predictions (e.g., ~2km), capturing stochastic variations and extreme weather events.
license: NVIDIA AI Enterprise
metadata:
    skill-author: MindSpore Science Team
---

# CorrDiff (Correction Diffusion)

## Overview

CorrDiff (Correction Diffusion) is a generative correction diffusion model for km-scale atmospheric downscaling developed by NVIDIA. It addresses the challenge of improving weather hazard predictions without expensive simulations by using a cost-effective stochastic downscaling approach.

### Key Technical Details:
- **Architecture**: Two-step approach combining a UNet-based regression model with a diffusion model
- **Purpose**: Weather downscaling - transforming low-resolution weather data (e.g., ERA5 at ~25km) to high-resolution predictions (e.g., ~2km)
- **Paper**: [Residual Diffusion Modeling for Km-scale Atmospheric Downscaling](https://arxiv.org/abs/2309.15214)
- **Foundation**: Based on "Elucidating the design space of diffusion-based generative models" (EDM)

### Model Components:
1. **Regression Model (Mean Predictor)**: A deterministic UNet that provides baseline high-resolution predictions
2. **Diffusion Model**: A residual diffusion model that learns to correct the regression predictions, capturing stochastic variations and extreme weather events

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping you determine whether the model suits their task requirements.

- **Scenario 1**: Weather Downscaling - Upscale coarse-resolution weather data (ERA5, GEFS) to high-resolution (HRRR-level) predictions
- **Scenario 2**: Extreme Weather Prediction - Capturing intense rainfall, typhoon dynamics, and other weather extremes
- **Scenario 3**: Multi-variate Weather Modeling - Modeling relationships between multiple weather variables
- **Scenario 4**: Ensemble Forecasting - Generating multiple plausible high-resolution weather scenarios
- **Scenario 5**: Regional Weather Modeling - Custom datasets for specific geographic regions (e.g., Taiwan, Continental US)

### Example Datasets Supported:
- **HRRR-Mini**: Simplified US dataset for learning/educational purposes
- **GEFS-HRRR**: Full continental US dataset
- **Taiwan (CWB)**: High-resolution Taiwan weather data
- **Custom datasets**: User-defined datasets for specific regions

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure you can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | NetCDF4 (.nc files) |
| Data Structure | Paired low-resolution (input) and high-resolution (output) data |
| Data Size | At least 50,000 samples recommended (more is better) |
| Data Source | Custom datasets or pre-downloaded from NGC |

#### Dataset Components

```
Dataset must include:
├── Input data (low-resolution)
├── Output data (high-resolution target)
├── Invariant variables (static fields like topography, land-sea mask)
├── Time coordinates
├── Spatial coordinates (lat/lon)
└── Statistics file (JSON) for normalization
```

#### Data Preprocessing

You need to preprocess data according to the following steps:

- **Preprocessing Step 1**: Prepare NetCDF4 files with paired low-res and high-res data
- **Preprocessing Step 2**: Create statistics JSON file for normalization (mean, std for each channel)
- **Preprocessing Step 3**: Ensure spatial coordinates (lat/lon) are properly defined
- **Preprocessing Step 4**: Verify data shape matches expected input/output channel configuration

#### Dataset Implementation (Custom Datasets)

Custom datasets must inherit from `DownscalingDataset` and implement:
- `longitude()`, `latitude()`: Coordinate arrays
- `input_channels()`, `output_channels()`: Channel metadata
- `time()`: Time values
- `image_shape()`: Spatial dimensions (height, width)
- `__len__()`, `__getitem__()`: Data access

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping you quickly set up the development environment.

#### Dependency Installation

```bash
# Install core dependencies
pip install netCDF4>=1.7.2 hydra-core>=1.2.0 omegaconf>=2.3.0
pip install wandb>=0.13.7 nvtx>=0.2.8 dask>=2025.3.0
pip install xskillscore>=0.0.26 cftime>=1.6.2 opencv-python>=4.11.0.86
pip install numba>=0.61.0 scipy>=1.15.1 typer>=0.16.0 psutil>=6.0.0

# Install PhysicsNeMo core
pip install torch>=2.5.0 torchvision>=0.19.0 timm>=1.0.22
pip install einops>=0.8.1 h5py>=3.15.1
```

#### Environment Requirements

| Requirement | Specification |
|-------------|---------------|
| Python Version | Python 3.11 - 3.13 |
| Hardware | NVIDIA GPU with CUDA support (A100 recommended) |
| Memory | 64GB+ RAM recommended; GPU memory varies by batch size |
| Disk Space | Depends on dataset size (full GEFS-HRRR is several hundred GB) |

#### Installation Steps

1. **Step 1**: Clone the PhysicsNeMo repository - `git clone https://github.com/NVIDIA/physicsnemo.git`
2. **Step 2**: Install PhysicsNeMo - `pip install -e .`
3. **Step 3**: Install CorrDiff-specific dependencies - `cd examples/weather/corrdiff && pip install -r requirements.txt`
4. **Step 4**: Download dataset (e.g., HRRR-Mini from NGC)
5. **Step 5**: Copy example files for inference - `cp -r examples/weather/corrdiff /path/to/your/corrdiff`

---

### 3. Usage Limitations and Notes

This module lists the model's usage limitations, important notes, and license agreements to help you use the model in compliance.

#### Model Limitations

| Limitation Type | Description |
|----------------|--------------|
| Functional Limitations | Max ×16 downscaling for pure spatial super-resolution; ×11 when inferring new output variables |
| Performance Limitations | Full inference requires significant GPU memory; may need to reduce batch size |
| Scale Limitations | Patch size must exceed auto-correlation distance for patch-based models |

#### Notes

1. **CorrDiff-Mini is for educational purposes only** - predictions should NOT be used for real applications
2. **Pre-trained Checkpoints**: Available via NVIDIA AI Enterprise; may not be compatible with current implementation
3. **Memory Management**: Reduce `batch_size_per_gpu` if encountering OOM errors
4. **Inference Modes**:
   - `regression`: Only deterministic baseline
   - `diffusion`: Only residual correction
   - `all`: Regression + diffusion (recommended)
5. **Sampling Options**: Use stochastic sampling for diverse outputs, deterministic for reproducible results

---

### 4. Model Invocation Guide

This module provides code examples for initializing and running inference with the model. You must follow the steps in Section 2 to aquire the required scripts to run the code examples.

#### Basic Inference Command

```bash
python generate.py --config-name="config_generate_hrrr_mini.yaml" \
  ++generation.io.res_ckpt_filename=/path/to/diffusion/model \
  ++generation.io.reg_ckpt_filename=/path/to/regression/model
```

#### Configuration File Example (config_generate_hrrr_mini.yaml)

```yaml
defaults:
    - dataset: hrrr_mini
    - generation: non_patched

dataset:
    data_path: ./data/hrrr_mini/hrrr_mini_train.nc
    stats_path: ./data/hrrr_mini/stats.json

generation:
    num_ensembles: 2
    seed_batch_size: 1
    times:
        - 2020-02-02T00:00:00
    io:
        res_ckpt_filename: <diffusion_checkpoint.mdlus>
        reg_ckpt_filename: <regression_checkpoint.mdlus>
    inference_mode: all  # Options: "all", "regression", "diffusion"
    hr_mean_conditioning: true
```

#### Key Generation Parameters

```yaml
generation:
    num_ensembles: 2          # Number of samples per input
    seed_batch_size: 1        # Batch size for inference
    inference_mode: all      # "all", "regression", or "diffusion"
    hr_mean_conditioning: true  # Use regression output for conditioning
    
perf:
    use_fp16: false           # Use half-precision
    use_torch_compile: false  # Use torch.compile for speedup
    profile_mode: false       # Enable NVTX profiling
```

#### Sampling Configuration

```yaml
# Stochastic sampling (recommended for diverse outputs)
sampler:
    type: stochastic
    num_steps: 18

# Deterministic sampling (for reproducible results)
sampler:
    type: deterministic
    num_steps: 9
    solver: euler  # or "heun"
```

#### Reading NetCDF Output

```python
import netCDF4 as nc

# Open the output file
f = nc.Dataset("output.nc", "r")

# Access different groups
input_data = f["input"]      # Low-resolution input
truth_data = f["truth"]       # Ground truth (if available)
prediction = f["prediction"]  # Model predictions

# Access specific variables
var = prediction["temperature_2m"][:]
```

#### Visualization

```python
# Plot single sample
python inference/plot_single_sample.py --file <output.nc> --output-dir ./plots

# Plot multiple samples
python inference/plot_multiple_samples.py --file <output.nc> --output-dir ./plots
```

---

## Reference Resources

### Primary Resources

| Resource | Link |
|----------|------|
| CorrDiff Paper | [arxiv.org/abs/2309.15214](https://arxiv.org/abs/2309.15214) |
| EDM Paper | [Elucidating the design space of diffusion-based generative models](https://openreview.net/pdf?id=k7FuTOWMOc7) |
| GitHub Repository | [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo/blob/main/examples/weather/corrdiff) |
| Documentation | [docs.nvidia.com/physicsnemo](https://docs.nvidia.com/physicsnemo/index.html#core) |

### Dataset Downloads

| Dataset | Link |
|---------|------|
| HRRR-Mini | [NGC - modulus_datasets-hrrr_mini](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets-hrrr_mini) |
| Taiwan (CWB) | [NGC - modulus_datasets_cwa](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets_cwa) |
| GEFS-HRRR | [NGC - CorrDiff model](https://build.nvidia.com/nvidia/corrdiff/modelcard) |

### Additional Tools

- **Earth2Studio**: [github.com/NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio) - For visualization and workflow integration
- **Hydra**: [hydra.cc](https://hydra.cc/docs/intro/) - Configuration management
- **Weights & Biases**: [wandb.ai](https://wandb.ai) - Experiment tracking