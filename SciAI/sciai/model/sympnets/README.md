ENGLISH | [简体中文](README_CN.md)

# Contents

- [Sympnets Description](#sympnets-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Sympnets Description](#contents)

New symplectic networks (SympNets) are proposed for identifying Hamiltonian systems from data.
Two classes of SympNets are defined, LA-SympNets, which is composed of linear and activation modules,
and G-SympNets, which is composed of gradient modules.
SympNets can approximate arbitrary symplectic maps and generalize well,
outperforming baseline models (e.g. Hamiltonian Neural Networks), and are much faster in training and prediction.
An extended version of SympNets is developed to learn the dynamics from irregularly sampled data.

> [paper](https://www.sciencedirect.com/science/article/pii/S0893608020303063):
> Jin P, Zhang Z, Zhu A, et al. SympNets:Intrinsic structure-preserving symplectic networks for identifying Hamiltonian
> systems[J]. Neural Networks, 2020, 132:166-179.

Example details:
there are three cases: pendulum, double pendulum and three body.

## [Dataset](#contents)

The dataset is generated during runtime.
The size of dataset can be configured in function `init_data` in each problem case.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/sympnets/).

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

Full command:

```bash
python train.py \
    --problem pendulum \
    --layers 2 50 50 50 50 3 \
    --save_ckpt true \
    --save_data true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_pendulum_LASympNet_iter50000.ckpt \
    --save_data_path ./data \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 1000 \
    --lr 1e-3 \
    --batch_size null \
    --epochs 50000 \
    --net_type LA \
    --la_layers 3 \
    --la_sublayers 2 \
    --g_layers 5 \
    --g_width 30 \
    --activation sigmoid \
    --h_layers 4 \
    --h_width 30 \
    --h_activation tanh \
    --download_data sympnets \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── sympnets
│   ├── checkpoints                                 # checkpoints files
│   ├── data                                        # data files
│   ├── figures                                     # plot figures
│   ├── logs                                        # log files
│   ├── src                                         # source codes
│   │   ├── cases                                   # codes for different cases
│   │   │   ├── double_pendulum.py                  # definition for the double pendulum problem
│   │   │   ├── pendulum.py                         # definition for the pendulum problem
│   │   │   ├── problem.py                          # common patterns for all problems
│   │   │   └── three_body.py                       # definition for the three-body problem
│   │   ├── nn                                      # codes for neural networks
│   │   │   ├── fnn.py                              # fully connected neural networks
│   │   │   ├── hnn.py                              # Hamiltonian neural networks
│   │   │   ├── module.py                           # standard module format
│   │   │   └── symnets.py                          # symplectic modules
│   │   ├── brain.py                                # Runner based on mindspore
│   │   ├── data.py                                 # data process
│   │   ├── stormer_verlet.py                       # Stormer-Verlet scheme
│   │   └── utils.py                                # methods for some common patterns
│   ├── config.yaml                                 # hyper-parameters configuration
│   ├── README.md                                   # English model descriptions
│   ├── README_CN.md                                # Chinese model description
│   ├── train.py                                    # python training script
│   └── eval.py                                     # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                  | default value                                         |
|----------------|----------------------------------------------|-------------------------------------------------------|
| problem        | which problem to solve                       | pendulum                                              |
| layers         | neural network widths                        | 2 50 50 50 50 3                                       |
| save_ckpt      | whether save checkpoint or not               | true                                                  |
| save_data      | whether save data or not                     | true                                                  |
| save_fig       | whether save and plot figures or not         | true                                                  |
| load_ckpt      | whether load checkpoint or not               | false                                                 |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                                         |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/model_pendulum_LASympNet_iter50000.ckpt |
| save_data_path | path to save data                            | ./data                                                |
| figures_path   | figures saving path                          | ./figures                                             |
| log_path       | log saving path                              | ./logs                                                |
| print_interval | time and loss print interval                 | 1000                                                  |
| lr             | learning rate                                | 1e-3                                                  |
| batch_size     | batch size                                   | null                                                  |
| epochs         | number of epochs                             | 50000                                                 |
| net_type       | neural network type                          | LA                                                    |
| la_layers      | LA neural network layer number               | 3                                                     |
| la_sublayers   | LA neural network sublayer number            | 2                                                     |
| g_layers       | G neural network layer number                | 5                                                     |
| g_width        | G neural network layer width                 | 30                                                    |
| activation     | neural network activation function           | sigmoid                                               |
| h_layers       | H neural network layer number                | 4                                                     |
| h_width        | H neural network layer width                 | 30                                                    |
| h_activation   | H neural network activation function         | tanh                                                  |
| download_data  | necessary dataset and/or checkpoints         | sympnets                                              |
| force_download | whether download the dataset or not by force | false                                                 |
| amp_level      | MindSpore auto mixed precision level         | O3                                                    |
| device_id      | device id to set                             | None                                                  |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                                     |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  step: 0, loss: 0.006594808, interval: 1.4325690269470215s, total: 1.4325690269470215s
  step: 1000, loss: 3.4384914e-06, interval: 4.685465097427368s, total: 6.11803412437439s
  step: 2000, loss: 3.2273747e-06, interval: 3.522109031677246s, total: 9.640143156051636s
  step: 3000, loss: 3.0768356e-06, interval: 3.4496490955352783s, total: 13.089792251586914s
  step: 4000, loss: 2.8382028e-06, interval: 3.485715389251709s, total: 16.575507640838623s
  step: 5000, loss: 2.4878047e-06, interval: 3.4817137718200684s, total: 20.05722141265869s
  step: 6000, loss: 2.0460955e-06, interval: 3.4582290649414062s, total: 23.515450477600098s
  step: 7000, loss: 1.9280903e-06, interval: 3.470597505569458s, total: 26.986047983169556s
  step: 8000, loss: 1.2088091e-06, interval: 3.4948606491088867s, total: 30.480908632278442s
  step: 9000, loss: 9.309894e-07, interval: 3.5296313762664795s, total: 34.01054000854492s
  step: 10000, loss: 6.1760164e-07, interval: 3.5044443607330322s, total: 37.514984369277954s
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint path used for evaluation.

- running on GPU/Ascend

  ```bash
  python eval.py
   ```

  You can view the results through the log file.
  The result pictures are saved in ```figures_path```, by default is [`./figures`](./figures).