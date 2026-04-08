---
name: hpc-openfwi
description: OpenFWI deep learning framework for full-waveform inversion (FWI). Uses neural networks to invert seismic data for subsurface velocity models. Use for geophysical imaging, subsurface characterization, and seismic inversion research.
---

# HPC-OpenFWI

OpenFWI is an open-source benchmark dataset and deep learning framework for seismic Full Waveform Inversion (FWI), developed by Los Alamos National Laboratory.

## Scientific Applications

| Application | Use Case |
|------------|----------|
| **Seismic Imaging** | Subsurface velocity model reconstruction |
| **Oil & Gas** | Reservoir characterization, exploration |
| **Geophysics** | Earthquake imaging, crustal studies |
| **Carbon Storage** | CO2 storage monitoring, geological storage |
| **Mining** | Mineral exploration, subsurface mapping |

## Key Concepts

### Input/Output
| Type | Description |
|------|-------------|
| Input | Seismic shot gathers (source-receiver data) |
| Output | Velocity model (m/s) |
| Format | HDF5 (.h5), NumPy (.npy) |

### Datasets (OpenFWI Benchmark)
| Family | Description | Datasets |
|--------|-------------|----------|
| Vel | Layered velocity models | Vel-1.0, Vel-2.0, Vel-3.0 |
| Fault | Fault models | Fault-1.0, Fault-2.0, Fault-3.0 |
| Style | Geological styles | Style-A, Style-B |
| Kimberlina | Carbonate reservoirs | Kimberlina-A, Kimberlina-B |

### Model Architectures
| Model | Type | Best For |
|-------|------|----------|
| InversionNet | Encoder-Decoder | Baseline, general use |
| VelocityGAN | GAN | Complex geological structures |
| FWI-Net | Multi-scale | Multi-resolution features |
| UPFWI | Bayesian | Uncertainty quantification |

## Workflow


1. Select dataset → [references/01-datasets.md]
2. Choose model architecture → [references/02-model-architectures.md]
3. Configure training pipeline → [references/03-training-pipeline.md]
4. Run inference → [references/04-inference.md]
5. Diagnose issues → [references/error-recovery.md]

## Datasets

See [references/01-datasets.md](references/01-datasets.md) for:
- Dataset families (Vel, Fault, Style, Kimberlina)
- Data format (HDF5 structure)
- Data preprocessing and normalization
- Download and setup instructions

## Model Architectures

See [references/02-model-architectures.md](references/02-model-architectures.md) for:
- InversionNet (Encoder-Decoder)
- VelocityGAN (Generator-Discriminator)
- FWI-Net (Multi-scale wavelet)
- UPFWI (Uncertainty-aware)

## Training Pipeline

See [references/03-training-pipeline.md](references/03-training-pipeline.md) for:
- Loss functions (L1, L2, Perceptual, Adversarial)
- Optimizers (Adam, AdamW, SGD)
- Multi-GPU training (DataParallel, DDP)
- Mixed precision (AMP)
- Checkpointing and monitoring

## Inference

See [references/04-inference.md](references/04-inference.md) for:
- Single-sample and batch inference
- Uncertainty quantification (MC dropout, ensemble)
- Post-processing (denormalization, smoothing)
- Evaluation metrics (MSE, MAE, SSIM, PSNR)

## Error Recovery

See [references/error-recovery.md](references/error-recovery.md) for diagnosis of:
- CUDA out-of-memory errors
- GAN training collapse
- Poor inversion quality
- Data loading errors

## Templates

Template files in [assets/templates/](assets/templates/) serve as starting points:

| Template | Purpose |
|----------|---------|
| [`train_inversionnet.py`](assets/templates/train_inversionnet.py) | InversionNet training with AMP, checkpointing |
| [`inference.py`](assets/templates/inference.py) | Batch inference with visualization |
| [`data_loader.py`](assets/templates/data_loader.py) | Custom FWI dataset and DataLoader |
| [`openfwi_slurm.sh`](assets/templates/openfwi_slurm.sh) | SLURM multi-GPU training script |

## Skill Decision Map

```
User Requirements
├─ Dataset Selection
│  ├─ Simple layers → Vel-1.0
│  ├─ Complex layers → Vel-2.0, Vel-3.0
│  ├─ Faults → Fault-1.0, Fault-2.0, Fault-3.0
│  └─ Carbonate → Kimberlina-A, Kimberlina-B
├─ Model Architecture
│  ├─ Baseline accuracy → InversionNet
│  ├─ Detailed textures → VelocityGAN
│  └─ Uncertainty → UPFWI
├─ Training Strategy
│  ├─ Loss: Smooth → L1
│  ├─ Loss: Detailed → Perceptual
│  └─ Speed → AMP mixed precision
└─ Inference
   ├─ Single prediction → Standard inference
   └─ Uncertainty → MC dropout or ensemble
```

## Guardrails

### Data Requirements
- **Input**: Seismic data shape (n_shots, n_time, n_receivers)
- **Output**: Velocity model shape (n_z, n_x)
- **Normalization**: Min-max or Z-score standardization

### Memory Management
- **GPU Memory**: 8-16 GB for 2D models, 32+ GB for 3D
- **Batch Size**: Reduce if OOM errors occur
- **Mixed Precision**: Enable AMP for 2x memory savings

### Training Stability
- Monitor GAN discriminator/generator balance
- Use learning rate warmup
- Apply gradient clipping for exploding gradients

## Required Output

Always report:
- Dataset and data preprocessing
- Model architecture and configuration
- Training metrics (loss convergence)
- Inversion quality metrics (SSIM, MSE, MAE)
- GPU memory usage and training time
