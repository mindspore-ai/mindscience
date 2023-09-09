ENGLISH | [简体中文](README_CN.md)

# Contents

- [DeepONet Description](#deeponet-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [DeepONet Description](#contents)

DeepONet is a new network with small generalization error, consisting of a DNN for encoding the input function space
and another for encoding the output function domain. It can learn explicit and implicit operators and has been tested
on 16 diverse applications.

In this repository, two problems, which are related to Poisson equations, are solved using the Deep Ritz method.

> [paper](https://www.nature.com/articles/s42256-021-00302-5): Lu L, Jin P, Pang G, et al. Learning nonlinear operators
> via DeepONet based on the universal approximation theorem of operators[J].
> Nature machine intelligence, 2021, 3(3): 218-229.

## [Dataset](#contents)

The training and validation dataset and pretrained checkpoint files will be downloaded automatically at the first
launch.

The dataset contains 2 cases: 1D Caputo fractional derivative and 2D fractional Laplacian.

Dataset used: [1D Caputo fractional derivative] and [2D fractional Laplacian]

- Dataset size
    - 1D Caputo fractional derivative
        - length of u vector: m = 15
        - dim of (y, alpha): d = 2
        - number of training records: 1e6
        - number of test records: 1e6
        - number of test0 records: 1e2
    - 2D fractional Laplacian
        - length of u vector: m = 225
        - dim of (x, y, alpha): d = 3
        - number of training records: 1e6
        - number of test records: 1e6
        - number of test0 records: 1e2
- Data format: `.npz` files
    - Note: Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── 1d_caputo
│   │   ├── test.npz
│   │   ├── test0.npz
│   │   └── train.npz
│   ├── 2d_fractional_laplacian
│   │   ├── test.npz
│   │   ├── test0.npz
│   │   └── train.npz
```

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

A full command for `1D Caputo fractional derivative` is as follows:

```bash
python train.py \
    --problem 1d_caputo \
    --layers_u 15 80 80 80 \
    --layers_y 2 80 80 80 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints/1d_caputo \
    --load_ckpt_path ./checkpoints/1d_caputo/model_50000_float16.ckpt \
    --save_fig true \
    --figures_path ./figures/1d_caputo \
    --save_data true \
    --load_data_path ./data/1d_caputo \
    --save_data_path ./data/1d_caputo \
    --log_path ./logs \
    --lr 1e-3 \
    --epochs 20001 \
    --batch_size 100000 \
    --print_interval 10 \
    --download_data deeponet \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

If you want to run full command for `2D fractional Laplacian` case, please switch the `problem` in `config.yaml`.

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── deeponet
│   ├── checkpoints                     # checkpoints files
│   ├── data                            # data files
│   ├── figures                         # plot figures
│   ├── logs                            # log files
│   ├── src                             # source codes
│   │   ├── network.py                  # network architecture
│   │   ├── plot.py                     # network architecture
│   │   └── process.py                  # data process
│   ├── config.yaml                     # hyper-parameters configuration
│   ├── README.md                       # English model descriptions
│   ├── README_CN.md                    # Chinese model description
│   ├── train.py                        # python training script
│   └── eval.py                         # python evaluation script
```

### [Script Parameters](#contents)

There are two problem cases. In `config.yaml` or command parameter, the case can be chosen by the parameter `problem`.

| parameter | description                                                         | default value |
|-----------|---------------------------------------------------------------------|---------------|
| problem   | problem case to be solved, `1d_caputo` or `2d_fractional_laplacian` | 1d_caputo     |

For each problem case, the parameters are as follows.

| parameter      | description                                  | default value                          |
|----------------|----------------------------------------------|----------------------------------------|
| layers_u       | neural network widths                        | 15 80 80 80                            |
| layers_y       | neural network widths                        | 2 80 80 80                             |
| save_ckpt      | whether save checkpoint or not               | true                                   |
| load_ckpt      | whether load checkpoint or not               | false                                  |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints/1d_caputo                |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/1d_caputo/1d_caputo.ckpt |
| save_fig       | whether save and plot figures or not         | true                                   |
| figures_path   | figures saving path                          | ./figures/1d_caputo                    |
| save_data      | whether save data or not                     | true                                   |
| load_data_path | path to load data                            | ./data/1d_caputo                       |
| save_data_path | path to save data                            | ./data/1d_caputo                       |
| log_path       | log saving path                              | ./logs                                 |
| lr             | learning rate                                | 1e-3                                   |
| epochs         | number of training epochs                    | 20001                                  |
| batch_size     | number of recoreds per batch                 | 10000                                  |
| print_interval | time and loss print interval                 | 10                                     |
| download_data  | necessary dataset and/or checkpoints         | deeponet                               |
| force_download | whether download the dataset or not by force | false                                  |
| amp_level      | MindSpore auto mixed precision               | O3                                     |
| device_id      | device id to set                             | None                                   |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                      |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
   # grep "loss:" log
    epoch:0, step: 0/10, loss: 0.9956, interval: 33.33840894699097s, total: 33.33840894699097s
    Epoch: 1, Training loss: 0.8623, Test loss: 0.853619, Test loss0: 1.567865, RelErr: 0.923920750617981,  RelErr0: 1.25214421749115
    epoch:1, step: 0/10, loss: 0.853, interval: 18.432061195373535s, total: 51.7704701423645s
    epoch:2, step: 0/10, loss: 0.8345, interval: 0.27780890464782715s, total: 52.04827904701233s
    epoch:3, step: 0/10, loss: 0.818, interval: 0.2761566638946533s, total: 52.32443571090698s
    epoch:4, step: 0/10, loss: 0.816, interval: 0.2772941589355469s, total: 52.60172986984253s
    epoch:5, step: 0/10, loss: 0.8013, interval: 0.278522253036499s, total: 52.88025212287903s
    epoch:6, step: 0/10, loss: 0.795, interval: 0.2778182029724121s, total: 53.15807032585144s
    epoch:7, step: 0/10, loss: 0.794, interval: 0.2756061553955078s, total: 53.43367648124695s
    epoch:8, step: 0/10, loss: 0.791, interval: 0.272977352142334s, total: 53.70665383338928s
    epoch:9, step: 0/10, loss: 0.7837, interval: 0.2748894691467285s, total: 53.98154330253601s
    epoch:10, step: 0/10, loss: 0.7754, interval: 0.2739229202270508s, total: 54.25546622276306s
  ...
  ```

  The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint loading path `load_ckpt_path` specified
in `config.yaml` for evaluation.

- running on GPU/Ascend

  ```bash
  python eval.py
  ```

  You can view the process and results through the `log_path`, `./logs` by default.
  The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.