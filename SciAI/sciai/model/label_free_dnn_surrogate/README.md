ENGLISH | [简体中文](README_CN.md)

# Contents

- [LabelFree DNN Surrogate Description](#labelfree-dnn-surrogate-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [LabelFree DNN Surrogate Description](#contents)

Numerical simulations for fluid dynamics problems are computationally expensive.
Developing a cost-effective surrogate model is significant.
A physics-constrained deep learning approach is proposed to model fluid flows without relying on simulation data.
The approach incorporates governing equations into the loss function and performs well in numerical experiments.

> [paper](https://www.sciencedirect.com/science/article/pii/S004578251930622X): Luning Sun, Han Gao, Shaowu Pan,
> Jian-Xun Wang. Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data.
> Computer Methods in Applied Mechanics and Engineering, Volume 361, 2020, 112732, ISSN 0045-7825.

Parametric Pipe Flow
<p >
    <img align = 'center' height="200" src="figures/pipe_uProfiles_nuIdx_.png?raw=true">
</p>

|                                      Small Aneurysm                                       |                                     Middle Aneurysm                                      |                                    Large Aneurysm                                     |
|:-----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|
| <img height="200" src="figures/486scale-0.0032346553257729654uContour_test.png?raw=true"> | <img height="200" src="figures/151scale-0.011815133162025654uContour_test.png?raw=true"> | <img height="200" src="figures/1scale-0.02267951024095881uContour_test.png?raw=true"> |

## [Dataset](#contents)

The dataset is generated randomly during runtime.
The size of dataset is controlled by parameter `batch_size` in `config.yaml`, and by default is 50.

For validation process, the folder `./data/` provides CFD and NN results under different case counts.
The dataset for validation and pretrained checkpoint files will be downloaded automatically at the first launch.

If you need to download the validation dataset or checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/label_free_dnn_surrogate/).

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
    --layers 3 20 20 20 1 \
    --save_ckpt true \
    --save_fig  true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path \
        ./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_u.ckpt \
        ./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_v.ckpt \
        ./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_P.ckpt \
    --load_data_path ./data \
    --save_data_path ./data \
    --figures_path ./figures \
    --log_path ./logs \
    --lr 1e-3 \
    --epochs_train 500 \
    --epochs_val 400 \
    --batch_size 50 \
    --print_interval 100 \
    --nu 1e-3 \
    --download_data label_free_dnn_surrogate \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── label_free_dnn_surrogate
│   ├── checkpoints         # checkpoints files
│   ├── data                # data files
│   ├── figures             # plot figures
│   ├── logs                # log files
│   ├── src                 # source codes
│   │   ├── network.py      # network architecture
│   │   ├── plot.py         # plotting results
│   │   └── process.py      # data process
│   ├── config.yaml         # hyper-parameters configuration
│   ├── README.md           # English model descriptions
│   ├── README_CN.md        # Chinese model description
│   ├── train.py            # python training script
│   └── eval.py             # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                  | default value                                                                                                                                                                           |
|----------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| layers         | neural network layer definition              | 3 20 20 20 1                                                                                                                                                                            |
| save_ckpt      | whether save checkpoint or not               | true                                                                                                                                                                                    |
| save_fig       | whether save and plot figures or not         | true                                                                                                                                                                                    |
| load_ckpt      | whether load checkpoint or not               | false                                                                                                                                                                                   |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                                                                                                                                                                           |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_u.ckpt <br/>./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_v.ckpt <br/>./checkpoints/geo_para_axisy_sigma0.1_epoch500hard_P.ckpt |
| load_data_path | path to load original data                   | ./data                                                                                                                                                                                  |
| save_data_path | path to save data and to load data for plots | ./data                                                                                                                                                                                  |
| figures_path   | figures saving path                          | ./figures                                                                                                                                                                               |
| log_path       | log saving path                              | ./logs                                                                                                                                                                                  |
| lr             | learning rate                                | 1e-3                                                                                                                                                                                    |
| epochs_train   | number of epochs for training                | 500                                                                                                                                                                                     |
| epochs_val     | number of epochs for validation and plotting | 400                                                                                                                                                                                     |
| batch_size     | size of training dataset                     | 50                                                                                                                                                                                      |
| print_interval | time and loss print interval                 | 100                                                                                                                                                                                     |
| nu             | nu parameter for the loss function           | 1e-3                                                                                                                                                                                    |
| download_data  | necessary dataset and/or checkpoints         | label_free_dnn_surrogate                                                                                                                                                                |
| force_download | whether download the dataset or not by force | false                                                                                                                                                                                   |
| amp_level      | MindSpore auto mixed precision level         | O0                                                                                                                                                                                      |
| device_id      | device id to set                             | None                                                                                                                                                                                    |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                                                                                                                                                                       |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  # grep "loss:" log
  epoch:0, step: 0/2000, loss: 0.010065492, interval: 64.38495588302612s, total: 64.38495588302612s
  epoch:0, step: 100/2000, loss: 0.009565717, interval: 16.189687490463257s, total: 80.57464337348938s
  epoch:0, step: 200/2000, loss: 0.009905259, interval: 16.150275468826294s, total: 96.72491884231567s
  epoch:0, step: 300/2000, loss: 0.009798448, interval: 16.015777587890625s, total: 112.7406964302063s
  epoch:0, step: 400/2000, loss: 0.010146898, interval: 15.762168169021606s, total: 128.5028645992279s
  epoch:0, step: 500/2000, loss: 0.009967192, interval: 15.626747369766235s, total: 144.12961196899414s
  epoch:0, step: 600/2000, loss: 0.010065671, interval: 15.571009874343872s, total: 159.700621843338s
  epoch:0, step: 700/2000, loss: 0.010013511, interval: 16.07706379890442s, total: 175.77768564224243s
  epoch:0, step: 800/2000, loss: 0.0097869225, interval: 15.972872018814087s, total: 191.75055766105652s
  epoch:0, step: 900/2000, loss: 0.009805476, interval: 16.053072452545166s, total: 207.80363011360168s
  epoch:0, step: 1000/2000, loss: 0.009835069, interval: 16.04354953765869s, total: 223.84717965126038s
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
