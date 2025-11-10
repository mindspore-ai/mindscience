# Model Name

> CrystalFlow

## Introduction

Theoretical crystal structure prediction is an important approach for finding the most stable structures of materials under given external conditions via computation. Traditional structure prediction methods rely on wide random sampling over the potential energy surface to search for the most stable structure. However, such methods require local optimization on a large number of randomly generated structures, and local optimization typically incurs substantial first-principles computational cost. This cost becomes particularly significant when simulating complex multi-element systems, posing major challenges.

In recent years, deep learning–based generative methods for crystal structure generation have gained attention for their ability to sample plausible structures more efficiently on the potential energy surface. By learning from datasets of stable or locally stable structures, these methods generate reasonable crystal structures. Compared with random sampling, they reduce the cost of local optimization and can find the most stable structures of a system with fewer samples.

Based on neural ordinary differential equations and continuous density modeling with normalizing flows, our flow-based approach is simpler, more flexible, and more efficient than diffusion-model-based generative methods. Building on a flow-model architecture, we develop CrystalFlow, a generative model for crystal structures that achieves competitive performance on benchmarks such as MP20.

## Environment Requirements

1. Install dependencies: `pip install -r requirement.txt`

## Quick Start

1. Download the `mindscience/MindChem` package to the current directory and open `MindChem/applications/crystalflow`.
2. Download the corresponding datasets from the Dataset Link: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/
3. Install dependencies: `pip install -r requirement.txt`
4. Train: `python train.py`
5. Inference: `python evaluate.py`
6. Evaluation: `python compute_metric.py`
7. Evaluation results are saved to a JSON file under the `metric_dir` specified in `config.yaml`.

## Code Structure

The main modules are under the `models` folder. `cspnet.py` contains the network layers, and `flow.py` contains the flow-model module. The `data` folder includes dataset processing modules.

```text
applications
  └── crystalflow                                      # Model name
        ├── readme.md                                  # README (Chinese)
        ├── README_EN.md                               # README (English)
        ├── config.yaml                                # Configuration file
        ├── train.py                                   # Training entry
        ├── evaluate.py                                # Inference entry
        ├── compute_metric.py                          # Evaluation entry
        ├── requirement.txt                            # Environment dependencies
        ├── data                                       # Data processing modules
        |     ├── data_utils.py                        # Utility module
        |     ├── dataset.py                           # Build dataset
        |     └── crysloader.py                        # Build data loader
        └── models
              ├── conditioning.py                      # Conditional generation utilities
              ├── cspnet.py                            # GNN-based denoiser
              ├── cspnet_condition.py                  # Conditional network layers
              ├── diff_utils.py                        # Utilities
              ├── flow.py                              # Flow-model module
              ├── flow_condition.py                    # Conditional flow model
              ├── infer_utils.py                       # Inference utilities
              ├── lattice.py                           # Lattice matrix utilities
              └── train_utils.py                       # Training utilities
```

## Dataset Download

Download the required dataset folders and the `dataset_prop.txt` property file from: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/

Place them under the `dataset` folder in the current path (create it manually if missing). Example structure:

```text
crystalflow
    ...
    └─dataset
            perov_5      Perovskite dataset
            carbon_24    Carbon crystal dataset
            mp_20        MP dataset with up to 20 atoms per unit cell
            mpts_52      MP dataset with up to 52 atoms per unit cell
            dataset_prop.txt  Dataset property file
    ...
```

## Training

Download the `mindscience/MindChem` package to the current directory and open `MindChem/applications/crystalflow`.

Edit the config file to set training parameters:

- Set the training `dataset` (see the `dataset` field).
- Configure the denoiser model (see the `model` field).
- Set the directory for saving checkpoints by editing `train.ckpt_dir` and the checkpoint filename in `checkpoint.last_path`.
- Other training settings are under the `train` field.

Commands:

```bash
pip install -r requirement.txt
python train.py
```

## Inference

Edit the `test` section in the config file to adjust inference parameters, especially `test.num_eval`, which determines how many samples to generate per composition. This is important for the subsequent evaluation stage.

```bash
python evaluate.py
```

Generated crystals are saved to the path specified by `test.eval_save_path`.

The saved file contains a Python dictionary with the following structure:

```python
{
    'pred': [
        [crystal_A sample_1, crystal_A sample_2, ..., crystal_A sample_num_eval],
        [crystal_B sample_1, crystal_B sample_2, ..., crystal_B sample_num_eval],
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

Ensure `num_evals` is consistent with or less than the number of samples per composition used during inference. For example, if `num_evals` was 1 in inference, it must be 1 in evaluation; if it was 20 during inference, `num_evals` can be set to any integer from 1 to 20 for evaluation.

Set `test.metric_dir` in the config to specify where evaluation results are saved.

```bash
python compute_metric.py
```

Example of evaluation output:

```json
{"match_rate": 0.6107671899181959, "rms_dist": 0.07492558322002925}
```