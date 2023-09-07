ENGLISH | [简体中文](README_CN.md)

# Contents

- [Deep Galerkin Method](#deep-galerkin-method)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Deep Galerkin Method](#contents)

Deep Galerkin Method(DGM) uses a deep neural network instead of a linear combination of basis functions,
which acts as a reduced-form solution to PDEs in the widely-used Galerkin method.
The DGM algorithm can accurately solve a class of free boundary PDEs,
showing the capabilities of neural networks in solving PDEs.

This repository illustrates an example of advection equation.
A 4-layer dense neural network with `tanh` activation function is implemented to solve the equation.
The [animation](figures/animation.gif) illustrates learning process of the equation.

> [paper](https://arxiv.org/abs/1708.07469):
> Sirignano J, Spiliopoulos K. DGM: A deep learning algorithm for solving partial differential equations[J].
> Journal of computational physics, 2018, 375: 1339-1364.

## [Dataset](#contents)

The training dataset is generated at each epoch.
The dataset size in domain is controlled by `batch_size` in `config.yaml`, and by default is 256.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/dgm/).

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

It outputs several useful information:
<br>
1- Neural Network solution for the given equation  <br>
2- Loss function value (for the differential operator, boundary condition, etc.) <br>
3- Layer by Layer mean activation value (during training) for the neural network <br>
<br>

The folder [checkpoints](checkpoints) contains pre-trained networks, we can load them into the networks
by switch on the `load_ckpt` and specified the `load_ckpt_path` in `config.yaml`.
After, we can continue training by running:

```bash
python train.py
```

A full command running `train.py` is given below.

```bash
python train.py \
    --layers 1 10 10 10 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_fig true \
    --save_anim true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_iter_2000_float32.ckpt \
    --log_path ./logs \
    --figures_path ./figures \
    --print_interval 20 \
    --lr 0.01 \
    --epochs 2001 \
    --batch_size 256 \
    --download_data dgm \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

If we want to generate animation, showing the learning process,
we have to switch on the `save_fig` and `save_anim` configurations in `./config.yaml` and make sure the script
is running with `PYNATIVE` mode. The running mode can be tuned by changing the parameter when calling `init_project`.

```python
init_project(ms.PYNATIVE_MODE)  # PYNATIVE mode
init_project(ms.GRAPH_MODE)  # GRAPH mode
```

Besides, the package `ImageMagick` should be available in the environment. If not, it can be installed with following
command.

```bash
sudo yum install ImageMagick
```

## [Script Description](#contents)

## [Script and Sample Code](#contents)

File structures are as follows:

```text
├── dgm
│   ├── checkpoints                         # the folder storing checkpoint files
│   ├── data                                # data files
│   ├── figures                             # the folder storing figures and animations
│   ├── logs                                # logs folder
│   ├── src                                 # source codes
│   │   ├── advection.py                    # definition for advection
│   │   ├── plot.py                         # plotting functions
│   │   └── network.py                      # loss function and a trainer class
│   ├── config.yaml                         # configurations
│   ├── README.md                           # English model descriptions
│   ├── README_CN.md                        # Chinese model description
│   ├── train.py                            # main function
│   └── eval.py                             # load a checkpoint and evaluate the trained net
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                    | default value                              |
|----------------|------------------------------------------------|--------------------------------------------|
| layers         | neural network layer definition                | 1 10 10 10 1                               |
| save_ckpt      | whether save checkpoint or not                 | true                                       |
| load_ckpt      | whether load checkpoint or not                 | false                                      |
| save_fig       | whether save and plot figures or not           | true                                       |
| save_anim      | whether generate animations of training or not | true                                       |
| save_ckpt_path | checkpoint saving path                         | ./checkpoints                              |
| load_ckpt_path | checkpoint loading path                        | ./checkpoints/model_iter_2000_float32.ckpt |
| figures_path   | figures saving path                            | ./figures                                  |
| log_path       | log saving path                                | ./logs                                     |
| print_interval | time and loss print interval                   | 20                                         |
| lr             | learning rate                                  | 0.01                                       |
| epochs         | number of epochs                               | 2001                                       |
| batch_size     | size of training dataset                       | 256                                        |
| download_data  | necessary dataset and/or checkpoints           | dgm                                        |
| force_download | whether download the dataset or not by force   | false                                      |
| amp_level      | MindSpore auto mixed precision level           | O0                                         |
| device_id      | device id to set                               | None                                       |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)    | 0                                          |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # cat DGM_[your_time].log
  ...
  step: 320, total loss: 47.46501, loss_domain: 47.353966, loss_ic: 0.11104686, interval: 3.9606783390045166s, total: 105.76785326004028s
  step: 340, total loss: 42.394558, loss_domain: 42.363266, loss_ic: 0.031290982, interval: 3.5947396755218506s, total: 109.36259293556213s
  step: 360, total loss: 40.42996, loss_domain: 40.41625, loss_ic: 0.013711907, interval: 3.7165727615356445s, total: 113.07916569709778s
  step: 380, total loss: 33.631718, loss_domain: 33.61124, loss_ic: 0.020477751, interval: 3.7209231853485107s, total: 116.80008888244629s
  step: 400, total loss: 25.643202, loss_domain: 25.643173, loss_ic: 2.944071e-05, interval: 3.7022390365600586s, total: 120.50232791900635s
  step: 420, total loss: 29.891747, loss_domain: 29.88154, loss_ic: 0.010205832, interval: 3.708503007888794s, total: 124.21083092689514s
  step: 440, total loss: 28.60983, loss_domain: 28.609776, loss_ic: 5.462918e-05, interval: 3.9715089797973633s, total: 128.1823399066925s
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

We can evaluate the trained network by running the following command.
Before running the command, please check the checkpoint loading path `load_ckpt_path` specified
in `config.yaml` for evaluation.

- running on GPU/Ascend

  ```bash
  python eval.py
  ```

  You can view the process and results through the `log_path`, `./logs` by default.
  The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.