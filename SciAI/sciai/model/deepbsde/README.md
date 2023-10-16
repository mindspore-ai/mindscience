ENGLISH | [简体中文](README_CN.md)

# Contents

- [DeepBSDE Description](#DeepBSDE-description)
- [HJB equation](#HJB-equation)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Description of Random Situation](#description-of-random-situation)

## [DeepBSDE Description](#contents)

DeepBSDE is a power of deep neural networks by developing a strategy for solving a large class of high-dimensional nonlinear PDEs using deep learning. The class of PDEs that we deal with is (nonlinear) parabolic PDEs.

[paper](https:#www.pnas.org/content/115/34/8505): Han J , Arnulf J , Weinan E . Solving high-dimensional partial differential equations using deep learning[J]. Proceedings of the National Academy of Sciences, 2018:201718942-.

## [HJB equation](#Contents)

The Hamilton-Jacobi-Bellman (HJB) equation is the continuous-time analog to the discrete deterministic dynamic programming algorithm, which has now become
the cornerstone in many areas such as economics, behavioral science, computer science, and even biology, where intelligent decision-making is the key issue.

## [Environment Requirements](#contents)

- Hardware(GPU)
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https:#www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https:#www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https:#www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on GPU

Default:

```bash
python train.py
```

Full command is as follows:

```bash
python train.py \
    --save_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/deepbsde_HJBLQ_end.ckpt \
    --log_path ./logs \
    --print_interval 100 \
    --total_time 1.0 \
    --dim 100 \
    --num_time_interval 20 \
    --y_init_range 0 1 \
    --num_hiddens 110 110 \
    --lr_values 0.01 0.01 \
    --lr_boundaries 1000 \
    --num_iterations 1001 \
    --batch_size 64 \
    --valid_size 256 \
    --sink_size 100 \
    --file_format MINDIR \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
├── src
│     ├── config.py            # config parse script
│     ├── equation.py          # equation definition and dataset helper
│     ├── eval_utils.py        # evaluation callback and evaluation utils
│     └── net.py               # DeepBSDE network structure
├── config.yaml                # config file for deepbsde
├── export.py                  # export models API entry
├── README_CN.md
├── README.md
└── train.py                   # python training script
└── eval.py                    # python evaluation script
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in `config.yaml`

- config for HBJ

| parameter         | description                                 | default value                                       |
|-------------------|---------------------------------------------|-----------------------------------------------------|
| eqn_name          | equation function name                      | HJBLQ                                               |
| save_ckpt         | whether save checkpoint or not              | true                                                |
| load_ckpt         | whether load checkpoint or not              | false                                               |
| save_ckpt_path    | checkpoint saving path                      | ./checkpoints                                       |
| load_ckpt_path    | checkpoint loading path                     | ./checkpoints/discriminator/deepbsde_HJBLQ_end.ckpt |
| log_path          | log saving path                             | ./logs                                              |
| print_interval    | interval for loss printing                  | 100                                                 |
| total_time        | the total time of equation function         | 1.0                                                 |
| dim               | hidden layer dims                           | 100                                                 |
| num_time_interval | number of interval times                    | 20                                                  |
| y_init_range      | the y_init random initialization range      | [0, 1]                                              |
| num_hiddens       | a list of hidden layer's filter number      | [110, 110]                                          |
| lr_values         | lr_values of piecewise_constant_lr          | [0.01, 0.01]                                        |
| lr_boundaries     | lr_boundaries of piecewise_constant_lr      | [1000]                                              |
| num_iterations    | number of iterations                        | 2001                                                |
| batch_size        | batch_size when training                    | 64                                                  |
| valid_size        | batch_size when evaluation                  | 256                                                 |
| sink_size         | data sink size                              | 100                                                 |
| file_format       | export model type                           | MINDIR                                              |
| amp_level         | MindSpore auto mixed precision level        | O0                                                  |
| device_id         | device id to set                            | None                                                |
| mode              | MindSpore Graph mode(0) or Pynative mode(1) | 0                                                   |

### [Training Process](#contents)

  ```bash
  python train.py
  ```

  The python command above will print the training process to the console:

  ```console
  step: 0, loss: 1225.2937, interval: 8.1262, total: 8.1262
  eval loss: 4979.3413, Y0: 0.2015
  step: 100, loss: 320.9811, interval: 11.70984, total: 19.2246
  eval loss: 1399.8747, Y0: 1.1023
  step: 200, loss: 160.01154, interval: 6.7937, total: 26.0184
  eval loss: 807.4655, Y0: 1.4009
  ...
  ```

  After training, you'll get the last checkpoint file in the `save_ckpt_path` directory, `./checkpoints` by default .

### [Evaluation Process](#contents)

  Before running the command below, please check `load_ckpt_path` used for evaluation in `config.yaml`. An example would be `./checkpoints/deepbsde_HJBLQ_end.ckpt`

  ```bash
  python eval.py
  ```

  The above python command will print the evaluation result to the console:

  ```console
  eval loss: 5.146923065185527, Y0: 4.59813117980957
  Total time running eval 5.8552136129312079 seconds
  ```

## [Description of Random Situation](#contents)

  We use random sampling in equation.py, which can be set seed to fixed randomness.
