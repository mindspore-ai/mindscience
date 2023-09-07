ENGLISH | [简体中文](README_CN.md)

# Contents

- [Deep Ritz Description](#deep-ritz-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Deep Ritz Description](#contents)

The Deep Ritz Method is a deep learning-based method for numerically solving variational problems, particularly the ones
that arise from partial differential equations.

In this repository, two problems, which are related to Poisson equations, are solved using the Deep Ritz method.

> [paper](https://arxiv.org/abs/1710.00211): W E, B Yu.
> The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems.
> Communications in Mathematics and Statistics 2018, 6:1-12.

## [Dataset](#contents)

The training dataset is generated randomly during runtime.
The size of dataset is controlled by parameter `body_batch` and `bdry_batch` in `config.yaml`,
and by default are 1024 and 1024.
The validation dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

If you need to download the validation dataset or checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/deep_ritz/).

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend or on GPU

Default:

```bash
python train.py
```

A full command for poisson-hole case is as follows:

```bash
python train.py \
    --layers 2 8 8 8 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints/hole \
    --load_ckpt_path ./checkpoints/hole/model_50000_float32.ckpt \
    --save_fig true \
    --figures_path ./figures \
    --save_data true \
    --save_data_path ./data/hole \
    --log_path ./logs \
    --lr 0.01 \
    --train_epoch 50000 \
    --train_epoch_pre 0 \
    --body_batch 1024 \
    --bdry_batch 1024 \
    --write_step 50 \
    --sample_step 10 \
    --step_size 5000 \
    --num_quad 40000 \
    --radius 1 \
    --penalty 500 \
    --diff 0.001 \
    --gamma 0.3 \
    --decay 0.00001 \
    --autograd true \
    --download_data deep_ritz \
    --force_download false \
    --amp_level O2 \
    --device_id 0 \
    --mode 0
```

If you want to run full command for poisson-ls case, please switch the `problem` in `config.yaml`.

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── deep_ritz
│   ├── checkpoints                     # checkpoints files
│   ├── data                            # data files
│   ├── figures                         # plot figures
│   ├── logs                            # log files
│   ├── src                             # source codes
│   │   ├── process.py                  # data process
│   │   ├── poisson_hole.py             # problem definition for hole case
│   │   ├── poisson_ls.py               # problem definition for ls case
│   │   ├── network.py                  # network architecture
│   │   └── plot.py                     # plotting and recording functions
│   ├── config.yaml                     # hyper-parameters configuration
│   ├── README.md                       # English model descriptions
│   ├── README_CN.md                    # Chinese model description
│   ├── train.py                        # python training script
│   └── eval.py                         # python evaluation script
```

### [Script Parameters](#contents)

There are two problem cases. In `config.yaml`, the case can be chosen by the parameter `--problem`.

```bash
--problem:            Problem case to be solved, poisson_hole or poisson_ls.
                      Default: poisson_hole
```

For each problem case, the parameters are as follows.

| parameter       | description                                  | default value                               |
|-----------------|----------------------------------------------|---------------------------------------------|
| layers          | neural network widths                        | 2 8 8 8 1                                   |
| save_ckpt       | whether save checkpoint or not               | true                                        |
| load_ckpt       | whether load checkpoint or not               | false                                       |
| save_ckpt_path  | checkpoint saving path                       | ./checkpoints/hole                          |
| load_ckpt_path  | checkpoint loading path                      | ./checkpoints/hole/model_50000_float32.ckpt |
| save_fig        | whether save and plot figures or not         | true                                        |
| figures_path    | figures saving path                          | ./figures                                   |
| save_data       | whether save data or not                     | true                                        |
| save_data_path  | path to save data                            | ./data/hole                                 |
| log_path        | log saving path                              | ./logs                                      |
| lr              | learning rate                                | 1e-2                                        |
| train_epoch     | number of training epochs                    | 50000                                       |
| train_epoch_pre | number of pre-training epochs                | 0                                           |
| body_batch      | sampling size for disk                       | 1024                                        |
| bdry_batch      | sampling size for surface                    | 1024                                        |
| write_step      | printing steps for loss                      | 50                                          |
| sample_step     | re-sampling steps during training            | 10                                          |
| step_size       | exponentially decay step for lr              | 5000                                        |
| num_quad        | sampling number for validation               | 40000                                       |
| radius          | disk radius                                  | 1                                           |
| penalty         | loss penalty for loss2 during training       | 500                                         |
| diff            | differential step size                       | 1e-3                                        |
| gamma           | exponentially decay rate for lr              | 0.3                                         |
| decay           | weight decay                                 | 1e-5                                        |
| autograd        | whether use auto gradient or not             | true                                        |
| download_data   | necessary dataset and/or checkpoints         | deep_ritz                                   |
| force_download  | whether download the dataset or not by force | false                                       |
| amp_level       | MindSpore auto mixed precision level         | O2                                          |
| device_id       | device id to set                             | None                                        |
| mode            | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                           |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  step: 0, total loss: 166.09909, loss: 165.08899, error: 1.0101029, interval: 29.70781683921814s, total: 29.70781683921814s
  step: 50, total loss: 5.871787, loss: 5.261068, error: 0.6107192, interval: 1.2001934051513672s, total: 30.908010244369507s
  step: 100, total loss: 0.80151683, loss: 0.43523002, error: 0.3662868, interval: 1.1730225086212158s, total: 32.08103275299072s
  step: 150, total loss: 0.5899545, loss: 0.36189145, error: 0.22806305, interval: 1.1766719818115234s, total: 33.257704734802246s
  step: 200, total loss: 0.5207778, loss: 0.3336542, error: 0.18712364, interval: 1.1791396141052246s, total: 34.43684434890747s
  step: 250, total loss: 0.5430529, loss: 0.36813667, error: 0.17491627, interval: 1.1709723472595215s, total: 35.60781669616699s
  step: 300, total loss: 0.554542, loss: 0.39627352, error: 0.1582685, interval: 1.1721374988555908s, total: 36.77995419502258s
  step: 350, total loss: 0.42904806, loss: 0.28422767, error: 0.14482038, interval: 1.167961597442627s, total: 37.94791579246521s
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint loading path `load_ckpt_path` specified
in `config.yaml` for evaluation.

- running on GPU/Ascend

  ```bash
  python eval.py
  ```

  You can view the process and results through the `log_path`, `./logs` by default.
  The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.