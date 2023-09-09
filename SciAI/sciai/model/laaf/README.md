ENGLISH | [简体中文](README_CN.md)

# Contents

- [LAAF Description](#laaf-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [LAAF Description](#contents)

Proposing layer-wise and neuron-wise locally adaptive activation functions for deep and physics-informed neural
networks, these python codes optimize scalable parameters with a stochastic gradient descent variant. The method
accelerates convergence, reduces training cost, and avoids local minima.

> [paper](https://doi.org/10.1016/j.jcp.2019.109136):  A.D. Jagtap, K.Kawaguchi, G.E.Karniadakis, Adaptive activation
> functions accelerate convergence in deep and physics-informed neural networks, Journal of Computational Physics, 404
> (2020) 109136.

Example details:

```bash
f = 0.2*np.sin(6*x) if x < 0 else 0.1*x*np.cos(18*x) + 1
```

## [Dataset](#contents)

The dataset is generated randomly during runtime.
The size of dataset is controlled by parameter `num_grid` in `config.yaml`, and by default is 300.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/laaf/).

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
    --layers 1 50 50 50 50 1 \
    --save_ckpt true \
    --load_ckpt false \
    --save_fig true \
    --save_data true \
    --save_ckpt_path ./checkpoints \
    --figures_path ./figures \
    --load_ckpt_path ./checkpoints/model_15001.ckpt \
    --save_data_path ./data \
    --log_path ./logs \
    --lr 2e-4 \
    --epochs 15001 \
    --num_grid 300 \
    --sol_epochs 2000 8000 15000 \
    --download_data laaf \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── laaf
│   ├── checkpoints          # checkpoints files
│   ├── data                 # data files
│   ├── figures              # plot figures
│   ├── logs                 # log files
│   ├── src                  # source codes
│   │   ├── network.py       # network architecture
│   │   ├── plot.py          # plotting results
│   │   └── process.py       # data process
│   ├── config.yaml          # hyper-parameters configuration
│   ├── README.md            # English model descriptions
│   ├── README_CN.md         # Chinese model description
│   ├── train.py             # python training script
│   └── eval.py              # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                  | default value                  |
|----------------|----------------------------------------------|--------------------------------|
| layers         | neural network layer definition              | 1 50 50 50 50 1                |
| save_ckpt      | whether save checkpoint or not               | true                           |
| load_ckpt      | whether load checkpoint or not               | false                          |
| save_fig       | whether save and plot figures or not         | true                           |
| save_data      | whether save data output or not              | true                           |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                  |
| figures_path   | figures saving path                          | ./figures                      |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/model_15001.ckpt |
| save_data_path | path to save data                            | ./data                         |
| log_path       | log saving path                              | ./logs                         |
| lr             | learning rate                                | 2e-4                           |
| epochs         | number of epochs                             | 15001                          |
| num_grid       | number of grid intervals                     | 300                            |
| sol_epochs     | epochs to catch snapshot                     | 2000 8000 15000                |
| download_data  | necessary dataset and/or checkpoints         | laaf                           |
| force_download | whether download the dataset or not by force | false                          |
| amp_level      | MindSpore auto mixed precision level         | O3                             |
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
  step: 0, loss: 1.0621891, interval: 15.865238428115845s, total: 15.865238428115845s
  step: 10, loss: 0.8333796, interval: 0.27651143074035645s, total: 16.1417498588562s
  step: 20, loss: 0.6490651, interval: 0.24263739585876465s, total: 16.384387254714966s
  step: 30, loss: 0.49252713, interval: 0.24282169342041016s, total: 16.627208948135376s
  step: 40, loss: 0.37449843, interval: 0.24222493171691895s, total: 16.869433879852295s
  step: 50, loss: 0.317139, interval: 0.24213695526123047s, total: 17.111570835113525s
  step: 60, loss: 0.31154847, interval: 0.24191784858703613s, total: 17.35348868370056s
  step: 70, loss: 0.3132628, interval: 0.24203085899353027s, total: 17.595519542694092s
  step: 80, loss: 0.31056988, interval: 0.24201083183288574s, total: 17.837530374526978s
  step: 90, loss: 0.3099324, interval: 0.2420203685760498s, total: 18.079550743103027s
  step: 100, loss: 0.30981177, interval: 0.24202728271484375s, total: 18.32157802581787s
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
