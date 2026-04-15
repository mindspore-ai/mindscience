---
name: cbssolver
description: cbssolver (Convergent Born Series Solver) is a computational solver for 2D/3D acoustic wave equations based on the Convergent Born Series (CBS) method. It uses MindFlow to solve frequency domain acoustic wave equations for applications in medical ultrasound and geological exploration. Use this model when you need to solve acoustic wave equations for wavefield simulation in 2D or 3D domains.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend

---

# CBSSolver

## Overview

CBSSolver is a 2D/3D acoustic wave equation solver based on the Convergent Born Series (CBS) iterative method. The CBS method is widely recognized in engineering and academic communities due to its low memory requirements and absence of dispersion errors. This solver is implemented using MindFlow (part of MindScience) and can run on Ascend NPUs.

The solver takes velocity field and source information as input parameters and outputs the spatiotemporal distribution of the wavefield. It supports both 2D and 3D acoustic wave equation solving with parallelization across different source locations and frequency points.

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

- **Medical ultrasound imaging**: Simulate acoustic wave propagation for ultrasound imaging applications
- **Geological exploration**: Solve acoustic wave equations for seismic imaging and subsurface characterization
- **Wavefield simulation**: Perform forward modeling of acoustic wave propagation in heterogeneous media
- **Non-destructive testing**: Simulate wave propagation for material characterization

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | NumPy files (.npy) for velocity fields, CSV files for source locations and waveforms |
| Data Size | 2D: velocity_2d.npy, srclocs_2d.csv, srcwaves_2d.csv; 3D: velocity_3d.npy, srclocs_3d.csv, srcwaves_3d.csv |
| Data Source | Download from [cfd/acoustic/dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/cfd/acoustic) |

#### Data Acquisition Methods

1. **Official Dataset Download**: Download the preset input data from the MindFlow dataset repository
2. **Custom Data Preparation**: Prepare velocity field (NumPy array), source locations (CSV), and source waveforms (CSV) according to the required format

#### Data Preprocessing

- **Velocity field**: 2D or 3D NumPy array representing wave velocity distribution
- **Source locations**: CSV file with source position coordinates
- **Source waveforms**: CSV file with time-series source signals
- Place all data files in the `./dataset` directory
- Configure input files via `config_2d.yaml` (2D) or `config_3d.yaml` (3D)

---

### 2. Environment Configuration and Dependencies

#### MindSpore and MindScience Version

| Component | Version |
|-----------|---------|
| MindSpore | >= 2.4.0 |
| MindScience | == 0.8.0 |

#### Installation Steps

1. **Install MindSpore and MindScience**: Ensure the correct versions are installed in the environment
   ```bash
   pip install mindspore==2.4.0 mindscience==0.8.0
   ```

2. **Clone MindScience Repository**: Obtain the CBS solver code from MindScience
   ```bash
   git clone https://atomgit.com/mindspore-lab/mindscience.git
   # Or access directly: MindFlow/applications/cfd/acoustic
   ```

3. **Download Dataset**: Get the required input data
   ```bash
   # Download from: https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/cfd/acoustic
   # Place in ./dataset directory
   ```

#### Environment Requirements

| Requirement | Specification |
|-------------|---------------|
| Hardware | Ascend NPU with memory > 32GB |
| Python Version | Python 3.8+ |
| Disk Space | Sufficient for dataset and output files |

---

### 3. Usage

#### Running the Solver Script

Download `solve_acoustic.py` from [MindFlow/applications/cfd/acoustic](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/cfd/acoustic/solve_acoustic.py) and run:

```bash
python solve_acoustic.py --dim 2 --device_id 0 --mode GRAPH
```

**Parameters:**
- `--dim`: Dimension of space (2 or 3). Default: 2
- `--device_id`: ID of the computing card. Default: auto-select most idle card
- `--mode`: Running mode - `GRAPH` (static graph) or `PYNATIVE` (dynamic graph)

#### Running Jupyter Notebook

Use the English or Chinese Jupyter Notebook versions:
- English: [acoustic.ipynb](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/cfd/acoustic/acoustic.ipynb)
- Chinese: [acoustic_CN.ipynb](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/cfd/acoustic/acoustic_CN.ipynb)

#### Output Files

The solver generates:
- `u_star.npy`: Solution of the non-dimensionalized equation in the frequency domain
- `u_time.npy`: Dimensional final solution converted to the time domain
- `wave.gif`: Visualization animation of the time domain solution

---

### 4. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
|----------------|--------------|
| Functional Limitations | No trainable parameters - this is a solver, not a machine learning model |
| Performance Limitations | 3D solving is computationally intensive; large-scale 3D may require multi-NPU distribution |
| Scale Limitations | Number of frequency points affects memory usage; large problems should be split into batches |

#### Notes

- **Batch processing**: Due to the large number of frequency points, the problem is divided into `n_batches` batches solved sequentially along the frequency direction
- **3D multi-NPU**: For large-scale 3D models, frequency points can be manually partitioned across multiple NPU cards
- **Convergence**: The CBS method requires preprocessing and appropriate epsilon selection for convergence
- **PML boundary**: The solver supports PML (Perfectly Matched Layer) absorbing boundary conditions via `pml_size` parameter

---

## Reference Resources

- [MindFlow CBS Solver Source](https://atomgit.com/mindspore-lab/mindscience/tree/master/MindFlow/applications/cfd/acoustic)
- [Dataset Download](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/cfd/acoustic)
- [CBS Method Theory (Osnabrugge et al., 2016)](https://linkinghub.elsevier.com/retrieve/pii/S0021999116302595)
- [MindSpore Official Website](https://www.mindspore.cn/)
- [MindScience GitHub](https://atomgit.com/mindspore-lab/mindscience)

## Contributors

- **WhFanatic** (gitee: WhFanatic, email: hainingwang1995@gmail.com)
- **zhaog6** (gitee: zhaog6, email: zhaog6@lsec.cc.ac.cn)

## Citation

If this project is helpful to your research, please cite the relevant work.