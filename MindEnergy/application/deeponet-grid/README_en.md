# DeepONet-Grid-UQ

DeepONet-Grid network for power system fault prediction

## Background Introduction

### Source of Requirements and Value Overview

This work build an efficient DeepONet that

(i) takes as inputs the trajectories collected before and during the fault and
(ii) outputs the predicted post-fault trajectories.

In addition, they also endow their method with the much-needed ability to balance efficiency with reliable/trustworthy predictions via Ucertainty Quantification.

Original Paper : [DeepONet-grid-UQ: A trustworthy deep operator framework for predicting the power grid’s post-fault trajectories](!https://www.sciencedirect.com/science/article/abs/pii/S0925231223002503)
Original Code on torch: [Github](!https://github.com/cmoyacal/DeepONet-Grid-UQ)

### Research Background and Motivation

Power systems, as critical infrastructure, are essential for the stability and reliability of modern society. However, power grids frequently face rare but severe faults and disturbances, which can lead to system instability and even trigger large-scale blackouts.

Traditional dynamic security analysis requires solving complex nonlinear differential-algebraic equations, with extremely high computational costs, making real-time analysis difficult to achieve. With the transformation of power grids, power companies urgently need the capability to perform near real-time dynamic security assessment.

Existing machine learning methods mainly focus on binary classification problems (stable/unstable), lacking quantitative prediction capabilities for post-fault trajectories. System operators and planners need to understand the trajectories of various state variables after faults to assess whether voltage or frequency will violate predefined limits and trigger protection measures such as load shedding.

## Project Structure

```bash
deeponet-grid/
├── configs/
│   └── config.yaml          # Configuration file
├── src/
│   ├── model.py             # DeepONet model definition
│   ├── data.py              # Data loading and preprocessing
│   ├── utils.py             # Utility functions
│   ├── trainer.py           # Trainer implementation
│   └── metrics.py           # Evaluation metrics
├── train.py                 # Training script
├── inference.py             # Inference script
├── requirements.txt         # Dependency list
├── README_en.md             # This file
├── README.md                # Chinese version
└── outputs/                 # Output directory (auto-created)
```

## Installation & Configuration

### Install MindSpore

 Install MindSpore framework:

```bash
pip install mindspore
```

### Configuration File

Edit the `configs/config.yaml` file to configure model parameters:

#### Model Configuration

- `branch`: Branch network configuration (processes input functions)
- `trunk`: Trunk network configuration (processes evaluation points)
- `use_bias`: Whether to use bias terms

#### Training Configuration

- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `epochs`: Number of training epochs
- `optimizer`: Optimizer type (adam, sgd, adamw)
- `loss_type`: Loss function type (nll, mse)

#### Data Configuration

- `use_synthetic`: Whether to use synthetic data
- `data_path`: Data file path

## Usage

### 1. Training with Real Data

You can use the data we provided: [dataset](https://download.mindspore.cn/mindscience/mindenergy/dataset/applications/DeepONet-grid/). Thanks to the provider: lzh9673@163.com.

Make sure the `data_path` in `confis/config.yaml` is set correctly.

```bash
python train.py
```

```bash
msrun -worker_num 8 python train.py --distributed 1
```

#### Output Files

After training, the following files are generated in the output directory:

- `best_model.ckpt`: Best model checkpoint
- `final_model.ckpt`: Final model checkpoint
- `training_history.json`: Training history
- `test_results.json`: Test set evaluation results
- `training.log`: Training log

#### Evaluation Metrics

- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination
- **Calibration Error**: Calibration error

### 2. Resume Training from Checkpoint

```bash
python train.py --resume outputs/best_model.ckpt
```

### 3. Run Evaluation Only

```bash
python train.py --eval --resume outputs/best_model.ckpt
```

### 4. Show loss log curve

Call the function `extract_log` in `src/utils.py` with input parameters (log file path and evaluation json file path.)

### 5. Run Inference Only

```bash
# Single data point inference
python inference.py --checkpoint outputs/best_model.ckpt \
                   --data_path data/test-data.npz \
                   --trajectory_prediction \
                   --data_index 0

# Dataset inference
python inference.py --checkpoint outputs/best_model.ckpt \
                   --data_path data/test-data.npz \
                   --output_dir inference_results
```

### 6. Debug Mode

Set environment variable to enable detailed logging:

```bash
export MINDSPORE_LOG_LEVEL=DEBUG
python train.py
```

## More Information

### Model Architecture

DeepONet consists of two main components:

1. **Branch Network**: Processes input function `u(x)`
2. **Trunk Network**: Processes evaluation points `y`

Output is calculated through dot product:

```
G(u)(y) = Σᵢ bᵢ(u) tᵢ(y)
```

Where `bᵢ(u)` is the branch network output and `tᵢ(y)` is the trunk network output.

### Uncertainty Quantification

The model provides uncertainty quantification capabilities:

- **Mean Prediction**: Expected value predicted by the model
- **Standard Deviation Prediction**: Uncertainty measure of predictions

Supported loss functions:

- **Negative Log Likelihood (NLL)**: For uncertainty quantification
- **Mean Squared Error (MSE)**: Standard regression loss

### Data Format

Data files should be in `.npz` format with the following fields:

- `u`: Input function values, shape `(n_samples, n_sensors)`
- `y`: Evaluation points, shape `(n_samples, n_points, n_dim)`
- `s`: True solution values, shape `(n_samples, n_points, n_output)`

## Training results

|       Name        |    Results    |
| :---------------: | :-----------: |
|     Hardware      | Atlas 800T A2 |
| MindSpore version |    >=2.5.0    |
|      Samples      |     80000     |
|  Training Steps   |     1000      |
|     Scheduler     |    Cosine     |
|    Batch Size     |     1024      |
|      Min LR       |     1e-7      |
|      Max LR       |     5e-5      |
|        MSE        |   0.003331    |
|        MAE        |   0.024189    |
|        L1         |   0.024222    |
|        L2         |   0.057664    |