ENGLISH | [简体中文](README_CN.md)

# Contents

- [Parareal PINN Description](#parareal-pinn-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [parareal PINN Description](#contents)

A parareal PINN decomposes a long-time problem into many independent short-time problems supervised by an
inexpensive/fast coarse-grained (CG) solver.
The CG solver is designed to provide approximate predictions of solutions at discrete time, while many fine PINNs
simultaneously correct the solution iteratively.
Consequently, compared to the original PINN approach, the proposed PPINN approach may achieve a significant speed-up for
long-time integration of PDEs.

> [paper](https://www.sciencedirect.com/science/article/pii/S0045782520304357):
> Meng X, Li Z, Zhang D, et al. PPINN: Parareal physics-informed neural network for time-dependent PDEs[J]. Computer
> Methods in Applied Mechanics and Engineering, 2020, 370: 113250.

Example details: parareal PINN code to solve the Burgers equation.

## [Dataset](#contents)

The dataset is generated randomly during runtime.
The size of dataset is controlled by parameter `n_train` in `config.yaml`, and by default is 10000.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/ppinns/).

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
mpiexec -n 8 python train.py
```

Full command:

```bash
mpiexec -n 8 python train.py \
    --t_range 0 10 \
    --nt_coarse 1001 \
    --nt_fine 200001 \
    --n_train 10000 \
    --layers 1 20 20 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --save_output true \
    --save_data_path ./data \
    --load_ckpt_path ./checkpoints/fine_solver_4_float16/result_iter_1.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --epochs 50000 \
    --lbfgs false \
    --lbfgs_epochs 50000 \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── ppinns
│   ├── checkpoints                                 # checkpoints files
│   ├── data                                        # data files
│   ├── figures                                     # plot figures
│   ├── logs                                        # log files
│   ├── src                                         # source codes
│   │   ├── coarsesolver.py                         # architecture of coarse solver
│   │   ├── dataset.py                              # to create datasets
│   │   ├── finesolver.py                           # architecture of fine solver
│   │   ├── model.py                                # architecture of the model
│   │   └── net.py                                  # architecture of the net
│   ├── config.yaml                                 # hyper-parameters configuration
│   ├── README.md                                   # English model descriptions
│   ├── README_CN.md                                # Chinese model description
│   ├── train.py                                    # python training script
│   └── eval.py                                     # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                      | default value                                          |
|----------------|--------------------------------------------------|--------------------------------------------------------|
| t_range        | range of time                                    | 0 10                                                   |
| nt_coarse      | number of parts to split into for coarse solvers | 1001                                                   |
| nt_fine        | number of parts to split into for fine solvers   | 200001                                                 |
| n_train        | number of training datasets to create            | 10000                                                  |
| layers         | layer structure                                  | 1 20 20 1                                              |
| save_ckpt      | whether save checkpoint or not                   | true                                                   |
| save_fig       | whether save and plot figures or not             | true                                                   |
| load_ckpt      | whether load checkpoint or not                   | false                                                  |
| save_ckpt_path | checkpoint saving path                           | ./checkpoints                                          |
| save_output    | whether save training outputs or not             | true                                                   |
| save_data_path | path to save output data                         | ./data                                                 |
| load_ckpt_path | checkpoint loading path                          | ./checkpoints/fine_solver_4_float16/result_iter_1.ckpt |
| figures_path   | figures saving path                              | ./figures                                              |
| log_path       | log saving path                                  | ./logs                                                 |
| print_interval | time and loss print interval                     | 10                                                     |
| epochs         | number of epochs                                 | 50000                                                  |
| lbfgs          | whether use the lbfgs optimizer                  | false                                                  |
| lbfgs_epochs   | number of lbfgs optimizer epochs                 | 50000                                                  |
| amp_level      | MindSpore auto mixed precision level             | O3                                                     |
| device_id      | device id to set                                 | None                                                   |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)      | 0                                                      |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  mpiexec -n 8 python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  Fine solver for chunk#:2
  Fine solver for chunk#:3
  Fine solver for chunk#:1
  Fine solver for chunk#:4
  Fine solver for chunk#:5
  Fine solver for chunk#:6
  Fine solver for chunk#:7
  step: 0, loss: 35.401, interval: 4.902739763259888s, total: 4.902739763259888s
  step: 0, loss: 3.0025306, interval: 4.05675196647644s, total: 4.05675196647644s
  step: 10, loss: 31.377882, interval: 0.057642224568323s, total: 4.960381987828211s
  step: 10, loss: 2.124565, interval: 0.05868268013559839s, total: 4.11543464661203839s
  step: 20, loss: 28.006842, interval: 0.0367842587620457s, total: 4.9971662465902567s
  step: 20, loss: 1.7020686, interval: 0.03674263405928059s, total: 4.15217728067131898s
  step: 30, loss: 25.339191, interval: 0.0367320495820349s, total: 5.0338982961722916s
  step: 30, loss: 1.5006942, interval: 0.0364089623498762s, total: 4.18858624302119518s
  step: 40, loss: 23.387045, interval: 0.0379562045760954s, total: 5.071854500748387s
  step: 40, loss: 1.3562441, interval: 0.03872304529386204s, total: 4.22730928831505722s
  step: 50, loss: 22.027771, interval: 0.03325230498620495s, total: 5.10510680573459195s
  step: 50, loss: 1.2255007, interval: 0.03410502934802956s, total: 4.26141431766308678s
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
  mpiexec -n 8 python eval.py
  ```

  You can view the process and results through the `log_path`, `./logs` by default.
  The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.
