---
name: leadformer
description: leadformer is a deep learning model for high-resolution intelligent forecasting of Arctic sea ice leads (linear fracture zones). It uses a Transformer-based encoder-decoder architecture to predict lead morphology (length, width, orientation) at 2km resolution across the pan-Arctic region. Use this model when you need to forecast sea ice leads for Arctic navigation, climate research, or ocean-atmosphere exchange studies.
license: MIT
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend

---

# LeadFormer

## Overview

LeadFormer is a Transformer-based deep learning model designed for high-resolution intelligent forecasting of Arctic sea ice leads. Sea ice leads are linear fracture zones formed under the influence of waves, wind, and ocean currents. Their morphological characteristics (length, width, orientation) reflect the intensity of substance and energy exchange between the ocean and the atmosphere.

The model adopts an encoder-decoder framework:
- **Encoding stage**: Compresses and deepens features through overlapping block embedding and a four-level Transformer block structure
- **Decoding stage**: Gradually reconstructs spatial dimensions via MLP (Multi-Layer Perceptron) and upsampling operations
- **Core innovation**: Fuses global modeling capability of Transformers with local perception characteristics of CNNs, making it suitable for high-precision image processing tasks

This skill provides inference capabilities for LeadFormer on Huawei Ascend NPUs using MindSpore.

---

## When to Use

### Hardware Requirements

This model requires Ascend hardware. Before running, please verify that your device is Ascend:

```python
import subprocess

def check_npu_device():
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Ascend not detected. This model requires Ascend hardware.")
    except FileNotFoundError:
        raise RuntimeError("npu-smi command not found. Please ensure Ascend driver is installed.")

check_npu_device()
```

- **Arctic sea ice lead forecasting**: Predict the location, size, and orientation of sea ice leads for navigational route planning
- **Climate research**: Study ocean-atmosphere heat and moisture exchange based on lead morphology predictions
- **Environmental monitoring**: Monitor Arctic sea ice changes and seasonal/interannual variability
- **High-resolution ice condition forecasting**: Generate 2km resolution pan-Arctic lead predictions

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | Polar region images (format specified in config)            |
| Data Size   | Training: 728 samples; Testing: 44 samples (as per reference) |
| Data Source | Not open-source; users must prepare their own polar region imagery |

#### Data Acquisition Methods

1. **Satellite imagery**: Obtain polar region satellite images from sources like MODIS, Sentinel, or other Earth observation satellites
2. **Numerical model output**: Use Arctic numerical model data as input
3. **Custom data**: Prepare high-resolution polar region images matching the 2km resolution requirement

#### Data Preprocessing

- Ensure images cover the pan-Arctic region
- Match the spatial resolution to 2km per pixel
- Configure `data_path` in the configuration file to point to your dataset
- Set `model_checkpoint` in the config for inference mode

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend stack (reference)

| Component | Version     |
| --------- | ----------- |
| Hardware  | Ascend 64G  |
| MindSpore | 2.5.0       |
| Python    | 3.10 (recommended for MindSpore) |

#### Environment Setup

```bash
# Install MindSpore 2.5.0 (CPU or NPU version as needed)
pip install mindspore==2.5.0

# Install other dependencies (check requirements.txt in the repository)
# The repository should contain a requirements.txt with additional dependencies
```

#### Clone LeadFormer Repository

```bash
# Clone the LeadFormer repository (check official source for exact URL)
# The main script is main.py with configuration files in ./configs/
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Hardware    | Ascend NPU (64G recommended)                   |
| Memory      | At least 32GB RAM recommended                  |
| Disk Space  | At least 10GB for model checkpoints and outputs |

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Prepare your polar region image dataset (2km resolution)   |
| 2    | Set `data_path` in `./configs/2km_ice_config.yaml`         |
| 3    | Install MindSpore 2.5.0 and dependencies                    |
| 4    | For inference: set `model_checkpoint` to diffusion model path |
| 5    | Run inference: `python main.py --device_id 0 --mode test`  |

Use Linux or WSL (or Git Bash on Windows) so `bash` is available.

Optional NPU check:

```bash
python -c "import mindspore; print(mindspore.__version__); print(mindspore.get_context('device_target'))"
```

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Requires high-resolution polar region imagery; dataset not open source |
| Performance Limitations | Inference time depends on image size and hardware           |
| Scale Limitations       | Optimized for 2km resolution pan-Arctic region              |
| Input Format            | Must match the expected image format in configuration        |

#### Notes

- **Note 1**: The dataset for LeadFormer is currently not open-source. Users must prepare their own polar region imagery data.
- **Note 2**: For NPU inference, ensure Ascend CANN and MindSpore are properly installed and the NPU is visible to MindSpore.
- **Note 3**: The model uses an encoder-decoder Transformer architecture with overlapping block embedding for feature extraction.
- **Note 4**: Configuration files are located in `./configs/` directory - modify `data_path` and `model_checkpoint` as needed.

---

### 4. Model Invocation Guide

#### Running Inference

**Command line:**

```bash
python main.py --device_id 0 --mode test
```

Where:
- `--device_id`: Device ID, default is 0
- `--mode`: Running mode, use "test" for inference

**For training:**

```bash
python main.py --device_id 0 --device_target Ascend --cfg ./configs/diffusion_cfg.yaml --mode train
```

Where:
- `--device_target`: Device type, default is Ascend
- `--device_id`: Device ID, default is 0
- `--cfg`: Path to the configuration file, default is "./configs/2km_ice_config.yaml"
- `--mode`: Running mode, use "train" for training

#### Configuration

In `./configs/2km_ice_config.yaml`:

```yaml
# Set data path to your dataset
data_path: /path/to/your/data

# Set model checkpoint for inference
model_checkpoint: /path/to/diffusion_model_checkpoint
```

#### Result Display

The prediction results include:
- Training Loss (RMSE): 0.07727
- Lead Detection Prediction Accuracy (Acc): 98.90112%
- Lead Length Deviation: 0.09848%
- Lead Angle Deviation: 6.27244°
- Lead Width Deviation: 1.21519%

Output visualization shows black outlines for topography and colored bands for prediction results.

---

## Reference Resources

- **LeadFormer Model**: Transformer-based sea ice lead forecasting
- **MindSpore Documentation**: https://www.mindspore.cn/
- **Ascend NPU Documentation**: https://www.huawei.com/ascend
- **Arctic Sea Ice Research**: Sea ice leads are key indicators of ocean-atmosphere exchange

## Contributors

- **Gitee ID**: funfunplus
- **Email**: funniless@163.com