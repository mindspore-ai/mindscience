ENGLISH | [简体中文](README_CN.md)

# Contents

- [Navier-Stokes Flow Nets Description](#navier-stokes-flow-nets-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Navier-Stokes Flow Nets Description](#contents)

"Navier-Stokes flow nets(NSFNets)" solves the vorticity-velocity(VV) and velocity-pressure(VP) formulations of the
Navier-Stokes equations by training neural networks.
The training process of the model is unsupervised learning, which means no labeled data are fed into the net during
training.
The loss are defined by the residual of equations governing VP and VV formulation, and also the boundary conditions.
The results obtained by NSFNet performs better than CFD solvers in the ill-posed or inverse problems.

> [paper](https://www.sciencedirect.com/science/article/pii/S0021999120307257): Xiaowei Jin, Shengze Cai, Hui Li, George
> Em Karniadakis, NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible
> Navier-Stokes equations, Journal of Computational Physics, Volume 426, 2021, 109951, ISSN 0021-9991.

Example details: the code to simulate Kovasznay flow (2-dimension without time).

## [Dataset](#contents)

The dataset is generated randomly during runtime.
The size of dataset is controlled by parameter `n_train` for domain and `n_bound` for boundary in `config.yaml`,
and by default are 2601 and 100, respectively.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/nsf_nets/).

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
    --layers 2 50 50 50 50 3 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final_float32.ckpt \
    --log_path ./logs \
    --print_interval 10 \
    --n_train 2601 \
    --n_bound 100 \
    --lr 1e-3 1e-4 1e-5 1e-6 \
    --epochs 5000 5000 50000 50000 \
    --download_data nsf_nets \
    --force_download false \
    --amp_level O2 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── nsf_nets
│   ├── checkpoints                                 # checkpoints files
│   ├── data                                        # data files
│   ├── figures                                     # plot figures
│   ├── logs                                        # log files
│   ├── src                                         # source codes
│   │   ├──network.py                               # network architecture
│   │   └──dataset.py                               # to generate datasets
│   ├── config.yaml                                 # hyper-parameters configuration
│   ├── README.md                                   # English model descriptions
│   ├── README_CN.md                                # Chinese model description
│   ├── train.py                                    # python training script
│   └── eval.py                                     # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in `train.py` are as follows:

| parameter      | description                                  | default value                          |
|----------------|----------------------------------------------|----------------------------------------|
| layers         | layer structure                              | 2 50 50 50 50 3                        |
| save_ckpt      | whether save checkpoint or not               | true                                   |
| load_ckpt      | whether load checkpoint or not               | false                                  |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                          |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/model_final_float32.ckpt |
| log_path       | log saving path                              | ./logs                                 |
| print_interval | time and loss print interval                 | 10                                     |
| n_train        | sampling numbers inside domain               | 2601                                   |
| n_bound        | sampling numbers on boundary                 | 100                                    |
| lr             | learning rate                                | 1e-3 1e-4 1e-5 1e-6                    |
| epochs         | number of epochs                             | 5000 5000 50000 50000                  |
| download_data  | necessary dataset and/or checkpoints         | nsf_nets                               |
| force_download | whether download the dataset or not by force | false                                  |
| amp_level      | MindSpore auto mixed precision level         | O2                                     |
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
  step: 0, loss: 3.0267472, interval: 10.88703203201294s, total: 10.88703203201294s
  step: 10, loss: 1.9014359, interval: 0.2849254608154297s, total: 11.1719574928283697s
  step: 20, loss: 0.9572897, interval: 0.24947023391723633s, total: 11.42142772674560603s
  step: 30, loss: 0.6608443, interval: 0.24956488609313965s, total: 11.67099261283874568s
  step: 40, loss: 0.61762005, interval: 0.2589101791381836s, total: 11.92990279197692928s
  step: 50, loss: 0.61856925, interval: 0.2607557773590088s, total: 12.19065856933593808s
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