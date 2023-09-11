ENGLISH | [简体中文](README_CN.md)

# Contents

- [DeepBSDE Description](#DeepBSDE-description)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#Model-Description)
    - [Evaluation Performance](#Evaluation-Performance)
    - [Inference Performance](#Inference-Performance)
- [Description of Random Situation](#description-of-random-situation)

# [DeepBSDE Description](#contents)

DeepBSDE is a power of deep neural networks by developing a strategy for solving a large class of high-dimensional nonlinear PDEs using deep learning. The class of PDEs that we deal with is (nonlinear) parabolic PDEs.

[paper](https:#www.pnas.org/content/115/34/8505): Han J , Arnulf J , Weinan E . Solving high-dimensional partial differential equations using deep learning[J]. Proceedings of the National Academy of Sciences, 2018:201718942-.

## [HJB equation](#Contents)

Hamilton–Jacobi–Bellman Equation which is the term curse of dimensionality was first used explicitly by Richard Bellman in the context of dynamic programming, which has now become the cornerstone in many areas such as economics, behavioral science, computer science, and even biology, where intelligent decision making is the main issue.

# [Environment Requirements](#contents)

- Hardware(GPU)
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https:#www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https:#www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https:#www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

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
    --num_iterations 2000 \
    --batch_size 64 \
    --valid_size 256 \
    --sink_size 100 \
    --file_format MINDIR \
    --epochs 1001 \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
├── config
│     └── HJBLQ_config.yaml    # default config for HJB equation.
├── src
│     ├── config.py            # config parse script.
│     ├── equation.py          # equation definition and dataset helper.
│     ├── eval_utils.py        # evaluation callback and evaluation utils.
│     └── net.py               # DeepBSDE network structure.
├── eval.py                    # evaluation API entry.
├── export.py                  # export models API entry.
├── README_CN.md
├── README.md
├── requirements.txt           # requirements of third party package.
└── train.py                   # train API entry.
```

## [Script Parameters](#contents)

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
| num_iterations    | iterations numbers                          | 2000                                                |
| batch_size        | batch_size when training                    | 64                                                  |
| valid_size        | batch_size when evaluation                  | 256                                                 |
| sink_size         | data sink size                              | 100                                                 |
| file_format       | export model type                           | MINDIR                                              |
| epochs            | number of training epochs                   | 2001                                                |
| amp_level         | MindSpore auto mixed precision level        | O0                                                  |
| device_id         | device id to set                            | None                                                |
| mode              | MindSpore Graph mode(0) or Pynative mode(1) | 0                                                   |

## [Training Process](#contents)

- Running on GPU

  ```bash
  python train.py
  ```

- The python command above will run in the background, you can view the results through the file `train.log`。

  The loss value can be achieved as follows:

  ```log
  epoch: 1 step: 100, loss is 245.3738
  epoch time: 26883.370 ms, per step time: 268.834 ms
  total step: 100, eval loss: 1179.300, Y0: 1.400, elapsed time: 34
  epcoh: 2 step: 100, loss is 149.6593
  epoch time: 3184.401 ms, per step time: 32.877 ms
  total step: 200, eval loss: 659.457, Y0: 1.693, elapsed time: 37
  ...
  ```

  After training, you'll get the last checkpoint file under the folder `log_dir` in config.

## [Evaluation Process](#contents)

- Evaluation on GPU

  Before running the command below, please check the checkpoint path used for evaluation. Such as `./log/deepbsde_HJBLQ_end.ckpt`

  ```bash
  python eval.py --config_path=./config/HJBLQ_config.yaml > eval.log  2>&1 &
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The error of evaluation is as follows:

  ```log
  eval loss: 5.146923065185527, Y0: 4.59813117980957
  ```

# [Model Description](#contents)

## [Evaluation Performance](#contents)

| Parameters                 | GPU                                                     |
|----------------------------|---------------------------------------------------------|
| Model Version              | DeepBSDE                                                |
| Resource                   | NV SMX2 V100-32G                                        |
| uploaded Date              | 7/5/2021 (month/day/year)                               |
| MindSpore Version          | 1.2.0                                                   | |
| Training Parameters        | step=2000, see `./config/HJBLQ_config.yaml` for details |
| Optimizer                  | Adam                                                    | |
| Loss                       | 2.11                                                    |
| Speed                      | 32ms/step                                               |
| Total time                 | 3 min                                                   |
| Parameters                 | 650K                                                    |
| Checkpoint for Fine tuning | 7.8M (.ckpt file)                                       |

## [Inference Performance](#contents)

| Parameters        | GPU                       |
|-------------------|---------------------------|
| Model Version     | DeepBSDE                  |
| Resource          | NV SMX2 V100-32G          |
| uploaded Date     | 7/5/2021 (month/day/year) |
| MindSpore Version | 1.2.0                     |
| outputs           | eval loss & Y0            |
| Y0                | Y0: 4.59                  |

# [Description of Random Situation](#contents)

We use random sampling in equation.py, which can be set seed to fixed randomness.
