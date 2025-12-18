# DiffCSP

## Background

DiffCSP is a diffusion-based deep generative framework for crystal structure
prediction. It reformulates the search for stable crystal structures as a
generative task: by learning the distribution of large-scale crystal
datasets, the model can directly and efficiently generate plausible 3D atomic
structures (including lattice and atomic coordinates) from only the chemical
composition (atom types and ratios).

Compared with traditional structure-prediction methods that rely on extensive
quantum-mechanical calculations, DiffCSP’s key innovation lies in using a
periodic E(3)-equivariant graph neural network, explicitly incorporating
translational, rotational, and periodic symmetries. This ensures that
generated structures strictly obey physical constraints, enabling efficient
exploration of the crystal configuration space and producing high-quality
candidates at a much lower computational cost than first-principles methods.
DiffCSP thus provides a powerful tool for accelerated materials discovery and
design.

## Model Implementation

### Hardware Requirements

- Supports the `Ascend` backend. The runtime device can be specified via
  `--device_target` and defaults to `Ascend` (see `train.py`).

### Version Requirements

- Requires `MindSpore >= 2.7.0`.
- Requires `MindScience >= 0.8.0` for equivariant computations.

### Installation

- Install MindSpore: see the official guide at
  `https://www.mindspore.cn/install`
- Install MindScience: see `https://atomgit.com/mindspore-lab/mindscience`
- Install Python dependencies:
  `pip install -r requirement.txt`

### Dataset

- Download the dataset folders and the `dataset_prop.txt` property file from
  the dataset link:
  https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/
- Place them under the `dataset` folder in the current path (create it
  manually if missing).

Example directory structure:

```txt
diffcsp
    └─dataset
            perov_5        Perovskite dataset
            carbon_24      Carbon crystal dataset
            mp_20          MP dataset with up to 20 atoms per unit cell
            mpts_52        MP dataset with up to 52 atoms per unit cell
            dataset_prop.txt  Dataset property file
```

### Core Code

- The main modules are under the `models` and `data` folders:

```text
applications
  └── diffcsp
        ├── README.md                   # README (Chinese)
        ├── README_EN.md                # README (English)
        ├── config.yaml                 # Configuration file
        ├── train.py                    # Training entry
        ├── evaluate.py                 # Inference entry
        ├── compute_metric.py           # Evaluation entry
        ├── requirement.txt             # Environment dependencies
        ├── data
        |     ├── data_utils.py         # Dataset processing utilities
        |     ├── dataset.py            # Dataset reading and construction
        |     ├── dataloader.py         # DataLoader wrapper
        |     └── crysloader.py         # Raw dataset loader
        └── models
              ├── cspnet.py             # GNN-based denoiser
              ├── diffusion.py          # Diffusion model module
              ├── diff_utils.py         # Model utilities
              ├── infer_utils.py        # Inference utilities
              ├── train_utils.py        # Training utilities
              ├── graph.py              # Graph and adjacency construction
              └── loss.py               # Loss functions
```

- The main model is composed of `CSPNet` in `models/cspnet.py` and
  `CSPDiffusion` in `models/diffusion.py`:
- `CSPNet` is a periodic E(3)-equivariant denoising network that represents and denoises lattice
  and atomic coordinates.
- `CSPDiffusion` implements the forward/reverse diffusion process and crystal structure generation.
- Training uses the `Adam` optimizer and the `L2LossMask` loss
  (`models/loss.py`), and leverages `@ms.jit` to accelerate the forward pass
  and training steps.

## Running the Model

### Training

- Make sure the following preparations are completed:
    - MindSpore and all dependencies are installed.
    - The `dataset` directory is prepared as described above.
    - Training parameters are configured in `config.yaml`:
        - `dataset`: dataset name and path.
        - `train.epoch_size`: number of training epochs.
        - `model`: network configuration for the denoiser (number of layers,
          hidden dimension, number of frequencies, etc.).
        - `train.ckpt_dir` and `checkpoint.last_path`: directory and filename for
          saving checkpoints.
        - Other training settings are under `train`, `checkpoint`, etc.
- Then run the following in the `diffcsp` directory:

```bash
python train.py
```

### Inference

- Set the checkpoint path to load in the `checkpoint.last_path` field of
  `config.yaml`. Pretrained models can be downloaded from:
  https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/pre-train
- Edit the `test` section in `config.yaml` to set inference parameters,
  especially `test.num_eval`, which determines how many samples are generated
  per composition and is crucial for the subsequent evaluation stage.
- Run the following in the `diffcsp` directory:

```bash
python evaluate.py
```

Generated crystals are saved to the file specified by `test.eval_save_path`.
The file stores a Python dictionary with the following structure:

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

### Evaluation

- Set the path to the generated crystal file in the `test.eval_save_path`
  field of `config.yaml`.
- Ensure `num_evals` is consistent with or less than the number of samples
  per composition used during inference. For example:
    - If `num_evals = 1` during inference, it must also be 1 for evaluation.
    - If `num_evals = 20` during inference, `num_evals` can be any integer
      from 1 to 20 for evaluation.
- Set `test.metric_dir` in the config to specify where evaluation results are
  saved, then run in the `diffcsp` directory:

```bash
python compute_metric.py
```

Evaluation results are saved as JSON files under `metric_dir`, for example:

```json
{"match_rate": 0.985997357992074, "rms_dist": 0.013073775170360118}
```

## License

- License: `Apache License 2.0`
- License link: `http://www.apache.org/licenses/LICENSE-2.0`

## Citation

- If this project is helpful to your research, please cite, for example:
    - Jiao R, Huang W, Lin P, et al. Crystal structure prediction by joint
      equivariant diffusion\[J\]. Advances in Neural Information Processing
      Systems, 2024, 36.
