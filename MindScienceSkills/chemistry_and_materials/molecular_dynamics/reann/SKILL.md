---
name: reann
description: reann (Recursively Embedded Atom Neural Network) is a PyTorch-based end-to-end multi-functional Deep Neural Network Package for Molecular, Reactive and Periodic Systems. Use this model when you need to train interatomic potentials, predict dipole moments, transition dipole moments, and polarizabilities for molecular and periodic systems.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# REANN

## Overview

REANN (Recursively Embedded Atom Neural Network) is a PyTorch-based end-to-end multi-functional Deep Neural Network Package for Molecular, Reactive and Periodic Systems. Published in Physical Review Letters (2021) and Journal of Chemical Physics (2022), REANN provides a unified framework for learning interatomic potentials and molecular properties.

The model uses a recursively embedded atom neural network architecture to capture atomic interactions with high accuracy. It supports training for multiple property types:
- **Interatomic potentials** (energy and forces)
- **Dipole moments**
- **Transition dipole moments**
- **Polarizabilities**

REANN takes advantage of PyTorch's Distributed DataParallel features for scalable parallel training. It also provides interfaces to LAMMPS for molecular dynamics simulations and ASE (Atomic Simulation Environment) as a calculator.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits their task requirements.

- **Scenario 1**: Training interatomic potentials - Suitable for constructing machine learning potential functions for molecular dynamics simulations
- **Scenario 2**: Molecular property prediction - Suitable for predicting dipole moments, transition dipole moments, and polarizabilities
- **Scenario 3**: Periodic systems - Suitable for modeling crystalline materials and surfaces with periodic boundary conditions
- **Scenario 4**: MD simulations - Suitable for running molecular dynamics simulations via LAMMPS or ASE interfaces

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | Custom "configuration" text files with atomic coordinates, energies, forces |
| Data Structure | Two directories required: "train" and "val" each containing a "configuration" file |
| Data Size | Depends on system complexity; more configurations improve model accuracy |
| Properties Supported | Energy, atomic forces, dipole moments, polarizabilities |

#### Data Format Specification

The "configuration" file format:
```
# First line: arbitrary comment
# Next three lines: lattice vectors (for periodic systems)
# Fifth line: periodic boundary conditions (pbc x y z, use 0 0 0 for non-periodic)
# Following N lines (N = number of atoms):
#   atomic_name, relative_atomic_mass, coordinates(x,y,z), atomic_force_vectors (optional)
# Final line: "abprop:" followed by target property (energy/dipole/polarizability)
```

Example for non-periodic NMA system:
```
NMA molecule
a1  0.0  0.0  0.0
a2  1.0  0.0  0.0
...
pbc  0  0  0
abprop: energy
```

#### Data Acquisition Methods

1. **GDPy** - REANN is embedded in GDPy (https://github.com/hsulab/GDPy) for configuration space search
2. **DFT Calculations** - Generate training data from density functional theory calculations
3. **Experimental Data** - Use experimental molecular structures and properties

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

The AscendHub image below uses **Python 3.11** (cp311 wheels).

#### Create container

Pick a **CANN** base image that matches your **Ascend hardware generation** (A3, 910B, etc.). Image index:

https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884

Example (**A3**, CANN **8.3.rc1**, Ubuntu 22.04, Python 3.11) — change the image tag if your machine requires a different variant:

```shell
docker run -it -u root \
  --net=host --shm-size=5g \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  -v /home:/home/ \
  --name REANN_test \
  --entrypoint=/bin/bash \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-a3-ubuntu22.04-py3.11
```

Add `--device=/dev/devmm_svm` and additional `davinci*` devices if your host exposes them. Align `--device` list with the number of NPUs.

#### Install torch and torch_npu

Install **torch 2.1.0** for aarch64/cp311 first (CPU wheel or CANN-documented build), then the NPU extension:

```shell
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.1.0/torch_npu-2.1.0.post17-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0.post17-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install protobuf==3.20
```

#### Third-party Python packages

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 25.0.rc1.1 |
| CANN      | 8.3.rc1 |
| Python    | 3.11 |
| torch     | 2.1.0 |
| torch-npu | 2.1.0.post17 |

#### Clone repository

```shell
git clone https://atomgit.com/AI4Science/REANN.git
cd REANN/reann
```

#### Test data layout (CO2 + Ni100 example)

From `REANN/reann`:

```shell
mkdir para && cd para
ln -s ../../example/co2+ni100/para/input_density ./input_density
ln -s ../../example/co2+ni100/para/input_nn ./input_nn
cd ../../data
cp -r co2+Ni100/ co2+ni100/
```

#### Training (single-node launcher)

From the `reann` directory (after data and `para/` links are ready):

```shell
python3 -m torch.distributed.run --master_addr 127.0.0.1 --master_port 12345 \
  --nproc_per_node=1 --nnodes=1 --standalone ./
```

Adjust `--nproc_per_node` / `--nnodes` for multi-GPU or multi-node jobs.

#### Cluster / hostname issues

If distributed startup fails with hostname resolution errors, add a loopback mapping in **`/etc/hosts`**, for example:

```text
127.0.0.1   hostname-qdcab.foreman.pxe
```

Replace the hostname with the name your cluster stack reports (the exact string from the error log).

#### Environment requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend (image and wheels must match generation) |
| Memory | Sufficient NPU and host RAM for training config |
| Disk | Code, `para/`, `data/`, and checkpoints |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Start CANN container from Ascend Hub; match image to hardware |
| 2 | Install **torch 2.1.0**, **torch_npu** wheel, **protobuf==3.20**, **numpy**, **opt-einsum** |
| 3 | `git clone https://atomgit.com/AI4Science/REANN.git` and `cd REANN/reann` |
| 4 | Prepare `para/` symlinks and `data/co2+ni100/` copy as above (or your own dataset layout) |
| 5 | Run **`torch.distributed.run`** command from `reann/` |
| 6 | For other workflows, use `python -m reann` or example subdirs as in the repository |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ------------|
| Periodic Systems | Requires proper lattice vector specification in configuration file |
| Data Format | Custom configuration format required; no direct PDB/XYZ support |
| Training Only | REANN is primarily a training framework; inference requires trained model |
| Fortran Dependencies | Some features require Fortran compilation (neighbor list) |

#### Notes

- **Note 1**: REANN uses embedded atom neural network (EANN) methodology with recursive embedding for improved accuracy
- **Note 2**: The model supports both periodic and non-periodic systems via the "pbc" flag
- **Note 3**: For MD simulations, use the LAMMPS interface (`reann/lammps-interface`) or ASE calculator (`reann/ASE`)
- **Note 4**: FIREANN (Field-induced REANN) extends REANN for external field responses: https://github.com/zhangylch/FIREANN
---

### 4. Model invocation guide

#### Parameter Configuration Files

REANN uses two main parameter files in the `para/` directory:

**input_nn** - Neural network and training parameters:
```bash
# Neural network architecture
nl = [64,64]          # hidden layer sizes
nblock = 1
dropout_p=[0.0,0.0,0.0,0.0]
activate = 'Relu_like'

# Training parameters
Epoch=20000
patience_epoch = 200
start_lr = 0.001
end_lr = 1e-5
batchsize_train = 128
batchsize_val = 256

# Orbital coefficient parameters
oc_nl = [64,64]
oc_loop = 3
```

**input_density** - Atomic representation parameters:
```bash
neigh_atoms = 60
cutoff = 5.0
nipsin = 2           # maximal angular momenta (s, p, d...)
atomtype = ['O', 'C', 'Ni']
nwave = 8            # number of radial Gaussian functions
```

#### Running examples

**Ascend (distributed launcher, from `REANN/reann`):**

```bash
python3 -m torch.distributed.run --master_addr 127.0.0.1 --master_port 12345 \
  --nproc_per_node=1 --nnodes=1 --standalone ./
```

**Classic entry (when applicable):**

```bash
cd /path/to/REANN/reann
python -m reann
```

**CO2 + Ni100 example folder:**

```bash
cd example/co2+ni100
# Adjust para/input_nn and para/input_density as needed
python -m reann
```

**Using ASE interface:**
```bash
# Copy reann.py to ASE calculators
cp reann/ASE/reann.py $ASE_PATH/ase/calculators/
cp reann/ASE/ase_reann.py ./

# Use in Python
from ase import Atoms
from ase.calculators.reann import REANN

atoms = Atoms(...)
calc = REANN(model_path='path/to/model.pt')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
```

**LAMMPS interface:**
```bash
# Build LAMMPS with REANN pair style
cd reann/lammps-interface
mkdir build && cd build
cmake ../cmake
make
```

---

## Reference resources

- **AtomGit (clone URL)**: https://atomgit.com/AI4Science/REANN
- **GitCode mirror**: https://gitcode.com/AI4Science/REANN
- **Ascend Hub (image detail)**: https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884
- **Upstream**: https://github.com/bjiangch/REANN
- **Manual (PDF)**: https://github.com/bjiangch/REANN/blob/main/manual/REANNPackage_manumal_v_2_0.pdf
- **FIREANN**: https://github.com/zhangylch/FIREANN
- **GDPy**: https://github.com/hsulab/GDPy

### Citations

If you use this package, please cite:

1. The original EANN model: Yaolong Zhang, Ce Hu and Bin Jiang *J. Phys. Chem. Lett.* **10**, 4962–4967 (2019).
2. The EANN model for dipole/transition dipole/polarizability: Yaolong Zhang, Sheng Ye, Jinxiao Zhang, Jun Jiang and Bin Jiang *J. Phys. Chem. B* **124**, 7284–7290 (2020).
3. The theory of REANN model: Yaolong Zhang, Junfan Xia and Bin Jiang *Phys. Rev. Lett.* **127**, 156002 (2021).
4. The details about the implementation of REANN: Yaolong Zhang, Junfan Xia and Bin Jiang *J. Chem. Phys.* **156**, 114801 (2022).