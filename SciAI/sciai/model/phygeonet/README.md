ENGLISH | [简体中文](README_CN.md)

# Contents

- [PhyGeoNet Description](#phygeonet-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [PhyGeoNet Description](#contents)

Proposing a novel physics-constrained CNN learning architecture that aims to learn solutions of parametric
PDEs on irregular domains without any labeled data, the model has been assessed by solving a number of steady-state PDEs
on irregular domains, including heat equations, Navier-Stokes equations, and Poisson equations with parameterized
boundary conditions, varying geometries, and spatially-varying source fields.

> [paper](https://www.sciencedirect.com/science/article/pii/S0021999120308536): Gao H, Sun L, Wang J X. PhyGeoNet:
> Physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state PDEs on
> irregular domain[J]. Journal of Computational Physics, 2021, 428: 110079.

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

Dataset used: OpenFOAM boundary data

- Data format: FoamFile
- Data example:  
  Geometry-Adaptive
  :-----:

<p align="center">
    <img align = 'center' height="200" src="figures/mesh.png?raw=true">
</p>

- Note: Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── case0
│   ├── TemplateCase
│   │   ├── 0
│   │   ├── 30
│   │   ├── 60
            ...
```

If you need to download the dataset and checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/phygeonet/).

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

Default:

```bash
python train.py
```

Full command:

```bash
python train.py \
    --save_ckpt true \
    --load_ckpt false \
    --save_fig true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final.ckpt \
    --log_path ./logs \
    --figures_path ./figures \
    --load_data_path ./data/case0 \
    --save_data_path ./data/case0 \
    --lr 1e-3 \
    --epochs 1501 \
    --batch_size 1 \
    --download_data phygeonet \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── phygeonet
│   ├── checkpoints                  # checkpoints files
│   ├── data                         # data files
│   ├── figures                      # plot figures
│   ├── logs                         # log files
│   ├── src                          # source codes
│   │   ├── dataset.py               # dataset classes
│   │   ├── foam_ops.py              # openfoam operation functions
│   │   ├── network.py               # network architecture
│   │   ├── plot.py                  # plotting results
│   │   ├── py_mesh.py               # mesh visualization
│   │   └── process.py               # data process
│   ├── config.yaml                  # hyper-parameters configuration
│   ├── README.md                    # English model descriptions
│   ├── README_CN.md                 # Chinese model description
│   ├── requirements.txt             # library requirements for this model
│   ├── train.py                     # python training script
│   └── eval.py                      # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                  | default value                  |
|----------------|----------------------------------------------|--------------------------------|
| save_ckpt      | whether save checkpoint or not               | true                           |
| save_fig       | whether to save and plot figures             | true                           |
| load_ckpt      | whether load checkpoint or not               | false                          |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                  |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/model_final.ckpt |
| figures_path   | figures saving path                          | ./figures                      |
| log_path       | log saving path                              | ./logs                         |
| load_data_path | path to load data                            | ./data/case0                   |
| save_data_path | path to save data                            | ./data/case0                   |
| lr             | learning rate                                | 1e-3                           |
| epochs         | number of epochs                             | 1501                           |
| batch_size     | training batch size                          | 1                              |
| download_data  | necessary dataset and/or checkpoints         | phygeonet                      |
| force_download | whether download the dataset or not by force | false                          |
| amp_level      | MindSpore auto mixed precision level         | O0                             |
| device_id      | device id to set                             | None                           |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                              |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  step: 0, loss: 43195.062, interval: 27.112539768218994s, total: 27.112539768218994s
  m_res Loss:43195.062, e_v Loss:0.9002858788030269
  step: 1, loss: 33180.41, interval: 13.531551599502563s, total: 40.64409136772156s
  m_res Loss:33180.41, e_v Loss:0.9047985325837214
  step: 2, loss: 26929.658, interval: 13.711315870285034s, total: 54.35540723800659s
  m_res Loss:26929.658, e_v Loss:0.8957896098853404
  step: 3, loss: 25561.246, interval: 13.139239072799683s, total: 67.49464631080627s
  m_res Loss:25561.246, e_v Loss:0.873105561166751
  step: 4, loss: 22892.932, interval: 13.660631895065308s, total: 81.15527820587158s
  m_res Loss:22892.932, e_v Loss:0.8364832182670862
  step: 5, loss: 20156.662, interval: 13.264018535614014s, total: 94.4192967414856s
  m_res Loss:20156.662, e_v Loss:0.8054152573595449
  step: 6, loss: 18716.941, interval: 13.6557936668396s, total: 108.0750904083252s
  m_res Loss:18716.941, e_v Loss:0.7725231596138828
  step: 7, loss: 17887.295, interval: 13.549290895462036s, total: 121.62438130378723s
  m_res Loss:17887.295, e_v Loss:0.7317578269591556
  step: 8, loss: 16525.012, interval: 13.712275266647339s, total: 135.33665657043457s
  m_res Loss:16525.012, e_v Loss:0.6834196610680607
  step: 9, loss: 15021.678, interval: 13.676922798156738s, total: 149.0135793685913s
  m_res Loss:15021.678, e_v Loss:0.6367062855788159
  step: 10, loss: 14102.34, interval: 13.010124444961548s, total: 162.02370381355286s
  m_res Loss:14102.34, e_v Loss:0.5960404029235946
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