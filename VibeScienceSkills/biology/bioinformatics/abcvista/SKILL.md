---
name: abcvista
description: Fast and accurate antibody structure prediction method capable of predicting diverse antibody conformational ensembles. Use when:(1) predicting antibody 3D structures from sequences, (2) generating conformational ensembles for antibodies, (3) adjusting MSA depth for structure prediction, (4) accelerating inference with DS4Sci_EvoformerAttention. Requires GPU with >=4GB VRAM. 
metadata: 
    skill-author: Co-authored by Yang Cao Lab and MindSpore Science Team
---

# AbCVista

Antibody structure prediction with conformational ensemble generation, based on modified AlphaFold2.

## System Requirements

- 16GB RAM
- 50GB storage
- GPU with >=4GB VRAM

## Installation

```bash
git clone https://github.com/JZongf/AbCVista.git
cd AbCVista
conda env create -f environment.yml
conda activate AbCVista
python setup.py install
python download_database.py
```

## Basic Usage

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir
```

## Key Options

| Option | Description |
|--------|-------------|
| `--fasta_dir` | Input directory containing FASTA files |
| `--output_dir` | Output directory for predicted structures |
| `--max_msa_clusters` | Number of MSA sequences to use (default: 128) |
| `--max_extra_msa` | Extra MSA sequences (default: 128) |
| `--sample_count` | Number of predicted structures per target |
| `--hdbscan_cluster` | Enable cluster-based conformation prediction |

## MSA Depth Adjustment

Lower values = faster prediction, higher values = potentially more accurate:

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir --max_msa_clusters 128 --max_extra_msa 128
```

## Prediction Quantity

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir --sample_count 40
```

Generates 40 structural predictions per target.

## Conformational Ensemble Prediction

```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir --hdbscan_cluster
```

## Inference Acceleration

Clone CUTLASS and set environment variable:
```bash
git clone https://github.com/NVIDIA/cutlass
export CUTLASS_PATH=/path/to/cutlass
```

Use `--use_deepspeed_evoformer_attention` flag:
```bash
python run_fold.py --fasta_dir /path/to/fasta_dir --output_dir /path/to/output_dir --use_deepspeed_evoformer_attention
```

## Known Issue: CUDA Version Mismatch

If you encounter `RuntimeError: The detected CUDA version (xx.x) mismatches...`:

1. Edit `/path/to/envs/AbCVista/lib/python3.9/site-packages/torch/utils/cpp_extension.py`
2. Comment out the cuda version check section (lines with `if cuda_ver != torch_cuda_version`)
3. Re-run `python setup.py install`

## License

- Source code: Apache-2.0
- Model parameters: CC BY-NC 4.0 (from Google DeepMind's AlphaFold2)
