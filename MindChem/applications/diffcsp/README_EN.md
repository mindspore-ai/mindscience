# Model Name

> DiffCSP

## Introduction

DiffCSP is a diffusion-model-based deep learning framework for crystal structure prediction. It transforms the search for stable crystal structures into a generative task: by learning distributional patterns from large-scale crystal datasets, the model generates plausible 3D atomic structures (including lattice and atomic coordinates) directly and efficiently from only the chemical composition (types and ratios of atoms). Compared to approaches that depend on extensive quantum-mechanical computations, DiffCSP leverages SE(3)-equivariant graph neural networks and incorporates periodic boundary conditions to strictly respect physical symmetries. This enables highly efficient exploration of material polymorphs and provides a powerful tool for accelerated materials discovery and design.

## Environment Requirements

1. Install `mindspore (2.3.0)`
2. Install dependencies: `pip install -r requirement.txt`

## Quick Start

1. Download the `Mindchemistry/mindchemistry` package to the current directory.
2. Download the corresponding datasets from: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/
3. Install dependencies: `pip install -r requirement.txt`
4. Training: `python train.py`
5. Inference: `python evaluate.py`
6. Evaluation: `python compute_metric.py`
7. Evaluation results are saved as a JSON file under the `metric_dir` specified in `config.yaml`.

## Code Structure

```txt
diffcsp
    │  README.md                README (Chinese)
    │  README_EN.md             README (English)
    │  config.yaml              Configuration file
    │  train.py                 Training entry
    │  evaluate.py              Inference entry
    │  compute_metric.py        Evaluation entry
    │  requirement.txt          Environment dependencies
    │  
    └─data
            data_utils.py       Dataset processing utilities
            dataset.py          Dataset reader
            crysloader.py       Dataset loader
    └─models
            cspnet.py           GNN-based denoiser module
            diffusion.py        Diffusion model module
            diff_utils.py       Utilities
            infer_utils.py      Inference utilities
            train_utils.py      Training utilities
```

## Dataset Download

Download the required dataset folders and the `dataset_prop.txt` property file from: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/

Place them under the `dataset` folder in the current path (create it manually if missing). Example structure:

```txt
diffcsp
    ...
    └─dataset
            perov_5         Perovskite dataset
            carbon_24       Carbon crystal dataset
            mp_20           MP dataset with up to 20 atoms per unit cell
            mpts_52         MP dataset with up to 52 atoms per unit cell
            dataset_prop.txt  Dataset property file
    ...
```

## Training

Download the `Mindchemistry/mindchemistry` package to the current directory.

Edit the config file to set training parameters:

- Set the training dataset (see the `dataset` field).
- Configure the denoiser model (see the `model` field).
- Set the directory and filename for saving checkpoints by editing `train.ckpt_dir` and `checkpoint.last_path`.
- Other training settings are under the `train` field.

Commands:

```bash
pip install -r requirement.txt
python train.py
```

## Inference

Set the path to the checkpoint in the config field `checkpoint.last_path`. Pretrained models can be downloaded from: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/pre-train

Edit the `test` section in the config to adjust inference parameters, especially `test.num_eval`, which determines how many samples to generate per composition and is important for the subsequent evaluation stage.

```bash
python evaluate.py
```

Generated crystals are saved to the path specified by `test.eval_save_path`.

The saved file contains a Python dictionary with the following structure:

```python
{
    'pred': [
        [crystal_A sample_1, crystal_A sample_2, crystal_A sample_3, ... crystal_A sample_num_eval],
        [crystal_B sample_1, crystal_B sample_2, crystal_B sample_3, ... crystal_B sample_num_eval]
        ...
    ],
    'gt': [
        crystal_A ground_truth,
        crystal_B ground_truth,
        ...
    ]
}
```

## Evaluation

Set the path to the generated crystal file in `test.eval_save_path` of the config.

Ensure `num_evals` is consistent with or less than the number of samples per composition used during inference. For example, if `num_evals` was set to 1 for inference, it must be 1 for evaluation; if it was set to 20 during inference, `num_evals` can be any integer from 1 to 20 for evaluation.

Set `test.metric_dir` in the config to specify where evaluation results are saved.

```bash
python compute_metric.py
```

Example of evaluation output:

```json
{"match_rate": 0.985997357992074, "rms_dist": 0.013073775170360118}
```