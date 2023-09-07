ENGLISH | [简体中文](README_CN.md)

# Contents

- [Deep Hpms Description](#deep-hpms-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Deep Hpms Description](#contents)

This work proposes a deep learning approach for discovering nonlinear partial differential equations from scattered and
potentially noisy observations in space and time. The approach uses two deep neural networks to approximate the unknown
solution and the nonlinear dynamics. The effectiveness of the approach is tested on several benchmark problems spanning
a number of scientific domains.

In this repository, two problems, which are related to Burgers equation, kdv equation, are solved using the Deep Hpms
method.

> [paper](https://www.jmlr.org/papers/volume19/18-046/18-046.pdf):
> Raissi M. Deep hidden physics models: Deep learning of nonlinear partial differential equations[J]. The Journal of
> Machine Learning Research, 2018, 19(1): 932-955.

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

- Data format: `.mat` files
- Note: Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── matlab
│   ├── burgers.mat
│   ├── burgers_sine.mat
│   ├── cylinder.mat
│   ├── cylinder_vorticity.mat
│   ├── KdV_cos.mat
│   ├── KdV_sine.mat
│   ├── KS.mat
│   ├── KS_chaotic.mat
│   └── NLS.mat
```

If you need to download the dataset or checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/deep_hpms/).

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website and the required [dataset](#dataset) above, you can start training
and evaluation as follows:

- running on Ascend or on GPU

default:

```bash
python train.py
```

A full command for burgers_different case is as follows:

```bash
python train.py \
    --problem burgers_different \
    --u_layers 2 50 50 50 50 1 \
    --pde_layers 3 100 100 1 \
    --layers 2 50 50 50 50 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/burgers_diff_final.ckpt \
    --save_fig true \
    --figures_path ./figures \
    --load_data_idn_path ./data/burgers_sine.mat \
    --load_data_sol_path ./data/burgers.mat \
    --log_path ./logs \
    --lr 1e-3 \
    --train_epoch 30001 \
    --train_epoch_lbfgs 100 \
    --print_interval 100 \
    --lb_idn 0.0 -8.0 \
    --ub_idn 10.0 8.0 \
    --lb_sol 0.0 -8.0 \
    --ub_sol 10.0 8.0 \
    --download_data deep_hpms \
    --force_download false \
    --data_type float32 \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

If you want to run other cases, please switch the `problem` in `config.yaml` or specify `--problem` in command.
This model currently doesn't support PYNATIVE MODE.
parameter.

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── deep_hpms
│   ├── checkpoints                      # checkpoints files
│   ├── data                             # data files
│   │   ├── matlab                       # matlab codes for data generation
│   │   ├── burgers.mat                  # burgers case data
│   │   └── ...                          # other data
│   ├── figures                          # plot figures
│   ├── logs                             # log files
│   ├── src                              # source codes
│   │   ├── network_burgers_different.py # burgers different case networks
│   │   ├── network_common.py            # common networks
│   │   ├── network_kdv.py               # kdv case networks
│   │   ├── plot.py                      # plotting and recording functions
│   │   └── process.py                   # data process
│   ├── config.yaml                      # hyper-parameters configuration
│   ├── README.md                        # English model descriptions
│   ├── README_CN.md                     # Chinese model description
│   ├── train.py                         # python training script
│   └── eval.py                          # python evaluation script
```

### [Script Parameters](#contents)

There are two problem cases. In `config.yaml` or command parameter, the case can be chosen by the parameter `problem`.

| parameter | description                                                  | default value     |
|-----------|--------------------------------------------------------------|-------------------|
| problem   | problem case to be solved, `burgers_different` or `kdv_same` | burgers_different |

For each problem case, the parameters are as follows.

| parameter          | description                                                                                      | default value                         |
|--------------------|--------------------------------------------------------------------------------------------------|---------------------------------------|
| u_layers           | layer widths of neural network U                                                                 | 2 50 50 50 50 1                       |
| pde_layers         | layer widths of neural network PDE                                                               | 3 100 100 1                           |
| layers             | layer widths of neural network solution                                                          | 2 50 50 50 50 1                       |
| save_ckpt          | whether save checkpoint or not                                                                   | true                                  |
| load_ckpt          | whether load checkpoint or not                                                                   | false                                 |
| save_ckpt_path     | checkpoint saving path                                                                           | ./checkpoints                         |
| load_ckpt_path     | checkpoint to be loaded                                                                          | ./checkpoints/burgers_diff_final.ckpt |
| save_fig           | whether save figures or not                                                                      | true                                  |
| figures_path       | figure saving path                                                                               | ./figures                             |
| load_data_idn_path | loading data idn path                                                                            | ./data/burgers_sine.mat               |
| load_data_sol_path | loading data sol path                                                                            | ./data/burgers.mat                    |
| log_path           | log saving path                                                                                  | ./logs                                |
| lr                 | learning rate                                                                                    | 1e-3                                  |
| train_epoch        | adam epochs                                                                                      | 30001                                 |
| train_epoch_lbfgs  | l-bfgs epochs                                                                                    | 100                                   |
| print_interval     | time and loss printing interval                                                                  | 100                                   |
| lb_idn             | idn lower bound                                                                                  | 0.0, -8.0                             |
| ub_idn             | idn upper bound                                                                                  | 10.0, 8.0                             |
| lb_sol             | sol lower bound                                                                                  | 0.0, -8.0                             |
| ub_sol             | sol upper bound                                                                                  | 10.0, 8.0                             |
| download_data      | necessary dataset and/or checkpoints                                                             | deep_hpms                             |
| force_download     | whether download the dataset or not by force                                                     | false                                 |
| amp_level          | MindSpore auto mixed precision level                                                             | O3                                    |
| device_id          | device id to set                                                                                 | None                                  |
| mode               | MindSpore Graph mode(0) or Pynative mode(1). This model currently doesn't support Pynative mode. | 0                                     |

### [Training Process](#contents)

- Running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss1:" log
  step: 0, loss: 3308.352, interval: 3.1490728855133057s, total: 3.1490728855133057s
  step: 100, loss: 1074.0432, interval: 0.4218735694885254s, total: 3.570946455001831s
  step: 200, loss: 181.29312, interval: 0.36736583709716797s, total: 3.938312292098999s
  step: 300, loss: 87.94882, interval: 0.36727356910705566s, total: 4.305585861206055s
  step: 400, loss: 33.567818, interval: 0.365675687789917s, total: 4.671261548995972s
  step: 500, loss: 15.378567, interval: 0.36209774017333984s, total: 5.0333592891693115s
  step: 600, loss: 14.30908, interval: 0.3638172149658203s, total: 5.397176504135132s
  step: 700, loss: 10.322, interval: 0.3609278202056885s, total: 5.75810432434082s
  step: 800, loss: 13.931234, interval: 0.36093950271606445s, total: 6.119043827056885s
  step: 900, loss: 5.209699, interval: 0.3612406253814697s, total: 6.4802844524383545s
  step: 1000, loss: 4.2461824, interval: 0.3610835075378418s, total: 6.841367959976196s
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint loading path `load_ckpt_path` specified in `config.yaml`
for evaluation.

- running on GPU/Ascend

  ```bash
  python eval.py
  ```

  You can view the process and results through the `log_path`, `./logs` by default.
- The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.