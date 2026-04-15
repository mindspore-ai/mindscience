---
name: powerflownet
description: powerflownet is a deep learning model that uses Message Passing Graph Neural Networks (MP-GNNs) for high-quality power flow approximation. Use this model when you need to predict voltage magnitude, voltage angle, and active/reactive power flows in electrical power grids.
license: MIT
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend
---

# PowerFlowNet

## Overview

PowerFlowNet transforms the power flow problem into a GNN node-regression problem by representing each electrical bus as a node and each transmission line as an edge while maintaining the network's connectivity. The model's distinctiveness lies in its innovative PowerFlowConv architecture that combines message-passing GNNs and high-order GCNs in a unique arrangement for handling trainable masked embeddings of the network graph. This approach makes PowerFlowNet remarkably scalable and effective for power flow approximation.

The model was published in the International Journal of Electrical Power & Energy Systems (2024): https://www.sciencedirect.com/science/article/pii/S0142061524003338

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

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: Power flow prediction - Suitable for predicting voltage magnitude, voltage angle, and power flows in electrical power grids
- **Scenario 2**: Grid state estimation - Suitable for real-time state estimation in power system operation
- **Scenario 3**: Grid planning and analysis - Suitable for analyzing different grid configurations and load scenarios

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | PyTorch Geometric graph data (node features, edge indices, edge attributes) |
| Data Size | IEEE 14-bus or IEEE 118-bus test cases; supports custom grid configurations |
| Data Source | Download from SurfDrive: https://surfdrive.surf.nl/files/index.php/s/Qw4RHLvI2RPBIBL |

#### Data Acquisition Methods

1. **SurfDrive Download** - Download pre-processed power grid datasets from the official SurfDrive link
2. **Custom Data Generation** - Generate power grid data using power system analysis tools (e.g., pandapower) and convert to PyTorch Geometric format

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure data is in PyTorch Geometric `Data` format with node features, edge indices, edge attributes, and target labels
- **Step 2**: Node features should include: bus index, bus type (0=slack, 1=PV, 2=PQ), voltage magnitude, voltage angle, active power demand, reactive power demand
- **Step 3**: Edge features should include: from_bus, to_bus, resistance (pu), reactance (pu)
- **Step 4**: Organize data by grid case (e.g., 14, 118v2) in the data directory

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

**Note**: The official directory name is **PoweFlowNet** (not “PowerFlowNet”).

| Component | Version |
| --------- | ---------------------------------------- |
| HDK       | 25.0.RC1 |
| CANN      | 8.3.RC1 |
| Python    | 3.11 |
| torch     | 2.1.0 |
| torch-npu | 2.1.0.post14 |

#### Clone GitCode tree and apply patch (Ascend — primary)

```bash
git clone https://gitcode.com/AI4Science/AI4Energy.git
cd AI4Energy/PoweFlowNet
git clone https://github.com/StavrosOrf/PoweFlowNet.git
cd ./PoweFlowNet
git checkout 1ebf73ba9605427fccb273a6a703b059d800dedc
git apply ../PoweFlowNet.patch
cd ..
```

#### Conda environment and Python dependencies (README summary)

```bash
conda create --name powerflownet python=3.11
conda activate powerflownet
# Install torch==2.1.0, torch_npu==2.1.0.post14, then pip install -r requirements.txt
# Clone the pytorch_geometric tag from README, apply patch, pip install -e
# DrivingSDK and other links: see README
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU (GitCode AI4Energy / PoweFlowNet README) |
| Memory | 8GB+ RAM recommended |
| Disk Space | Datasets and checkpoints per README (e.g. `dataset_generator.py`) |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Clone **AI4Energy** and **PoweFlowNet**, apply `PoweFlowNet.patch` as above |
| 2 | Python 3.11 conda env; install **torch** / **torch_npu** and **requirements.txt** per README |
| 3 | Install patched **torch_geometric** and **DrivingSDK** per README |
| 4 | Data: `python dataset_generator.py` (README) |
| 5 | Training / inference per README (e.g. `bash test/train_8p.sh`, `python3 inference.py ...`) |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Requires structured power grid data in PyTorch Geometric format; cannot process raw power system files directly |
| Performance Limitations | Performance depends on grid size and topology; may need retraining for significantly different grid configurations |
| Scale Limitations | Optimized for IEEE standard test cases (14-bus, 118-bus); large-scale grids may require model modifications |
| Input Format | Requires specific node/edge feature dimensions (see documentation) |

#### Notes

- **Note 1**: The model uses 6 input node features and 5 edge features; ensure your data matches these dimensions
- **Note 2**: Recommended model architecture is `MaskEmbdMultiMPN` for best performance
- **Note 3**: Configuration files in `./configs/` control hidden dimensions, GNN layers, and other hyperparameters

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| checkpoint | Pre-trained models available from SurfDrive: https://surfdrive.surf.nl/files/index.php/s/iunfVTGsABT5NaD |
| model architecture | MaskEmbdMultiMPN (recommended) |
| config | standard.json (hidden_dim=129, n_gnn_layers=4, K=3) |

#### Running examples (recommended path)

**Shell inference:**

```bash
cd /path/to/PoweFlowNet
python3 test.py --cfg_json ./configs/standard.json \
               --data-dir ./data/ \
               --case 118v2 \
               --model MaskEmbdMultiMPN
```

**Shell training (for reference):**

```bash
python3 train.py --cfg_json ./configs/standard.json \
                --num-epochs 2000 \
                --data-dir ./data/ \
                --batch-size 128 \
                --train_loss_fn mse_loss \
                --lr 0.001 \
                --case 118v2 \
                --model MaskEmbdMultiMPN \
                --save
```

#### Result Post-processing

- Output predictions are saved to the specified output directory
- Results include voltage magnitude, voltage angle, active power (P), and reactive power (Q) for each bus
- Use MSE loss for evaluation against true power flow solutions

---

## Reference Resources

- **GitCode AI4Energy (PoweFlowNet)**: https://gitcode.com/AI4Science/AI4Energy/tree/main/PoweFlowNet
- **Official README**: https://gitcode.com/AI4Science/AI4Energy/blob/main/PoweFlowNet/README.md
- **Additional reference**: https://github.com/StavrosOrf/PoweFlowNet
- **Paper**: https://www.sciencedirect.com/science/article/pii/S0142061524003338
- **Dataset**: https://surfdrive.surf.nl/files/index.php/s/Qw4RHLvI2RPBIBL
- **Trained Models**: https://surfdrive.surf.nl/files/index.php/s/iunfVTGsABT5NaD