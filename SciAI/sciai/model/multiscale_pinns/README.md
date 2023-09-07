ENGLISH | [简体中文](README_CN.md)

# Contents

- [Multiscale PINNs Description](#multiscale-pinns-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Multiscale PINNs Description](#contents)

In this work, researchers investigate the limitations of Physics-informed neural networks (PINNs) in approximating
functions with high-frequency or multiscale features. They propose novel architectures that employ spatio-temporal and
multiscale random Fourier features to lead to robust and accurate PINN models. Numerical examples are presented for
several challenging cases where conventional PINN models fail, including wave propagation and reaction-diffusion
dynamics.

> [paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782521002759): Wang S, Wang H, Perdikaris P. On the
> eigenvector bias of Fourier feature networks: From regression to solving multiscale PDEs with physics-informed neural
> networks[J]. Computer Methods in Applied Mechanics and Engineering, 2021, 384: 113938.

## [Dataset](#contents)

The dataset is generated during runtime.
The size of dataset is controlled by parameter `nnum` in `config.yaml`, and by default is 1000.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/multiscale_pinns/).

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
    --layers 2 100 100 100 1  \
    --save_ckpt true  \
    --save_fig true  \
    --load_ckpt false  \
    --save_ckpt_path ./checkpoints \
    --figures_path ./figures \
    --load_ckpt_path ./checkpoints/model_10000.ckpt \
    --log_path ./logs \
    --lr 1e-3  \
    --epochs 40000  \
    --batch_size 128  \
    --net_type net_st_ff \
    --print_interval 100 \
    --nnum 1000 \
    --download_data multiscale_pinns \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── multiscale_pinns
│   ├── checkpoints                                 # checkpoints files
│   ├── data                                        # data files
│   ├── figures                                     # plot figures
│   ├── logs                                        # log files
│   ├── src                                         # source codes
│   │   ├── network.py                              # network architecture
│   │   ├── plot.py                                 # plotting results
│   │   └── process.py                              # data process
│   ├── config.yaml                                 # hyper-parameters configuration
│   ├── README.md                                   # English model descriptions
│   ├── README_CN.md                                # Chinese model description
│   ├── train.py                                    # python training script
│   └── eval.py                                     # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                    | default value                  |
|----------------|------------------------------------------------|--------------------------------|
| layers         | neural network widths                          | 2 100 100 100 1                |
| save_ckpt      | whether save checkpoint or not                 | true                           |
| save_fig       | whether save and plot figures or not           | true                           |
| load_ckpt      | whether load checkpoint or not                 | false                          |
| save_ckpt_path | checkpoint saving path                         | ./checkpoints                  |
| figures_path   | figures saving path                            | ./figures                      |
| load_ckpt_path | checkpoint loading path                        | ./checkpoints/model_10000.ckpt |
| log_path       | log saving path                                | ./logs                         |
| lr             | learning rate                                  | 1e-3                           |
| epochs         | number of epochs                               | 40000                          |
| batch_size     | batch size                                     | 128                            |
| net_type       | network type, can be net_nn, net_ff, net_st_ff | net_st_ff                      |
| print_interval | time and loss print interval                   | 100                            |
| nnum           | sample space number of points                  | 1000                           |
| download_data  | necessary dataset and/or checkpoints           | multiscale_pinns               |
| force_download | whether download the dataset or not by force   | false                          |
| amp_level      | MindSpore auto mixed precision level           | O3                             |
| device_id      | device id to set                               | None                           |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)    | 0                              |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  step: 0, total loss: 0.50939494, bcs_loss: 0.003596314, ics_loss: 0.49960062, res_loss: 0.0061979764, interval: 1.1743102073669434s, total: 1.1743102073669434s
  step: 100, total loss: 0.46047255, bcs_loss: 5.7089237e-05, ics_loss: 0.4603598, res_loss: 5.562951e-05, interval: 1.724724292755127s, total: 2.8990345001220703s
  step: 200, total loss: 0.55632657, bcs_loss: 6.789516e-05, ics_loss: 0.55621916, res_loss: 3.951962e-05, interval: 1.4499413967132568s, total: 4.348975896835327s
  step: 300, total loss: 0.51157826, bcs_loss: 2.2257213e-05, ics_loss: 0.5115249, res_loss: 3.1086667e-05, interval: 1.463547706604004s, total: 5.812523603439331s
  step: 400, total loss: 0.50273365, bcs_loss: 0.00047580304, ics_loss: 0.50186944, res_loss: 0.00038838215, interval: 1.4236555099487305s, total: 7.2361791133880615s
  step: 500, total loss: 0.5403254, bcs_loss: 6.049385e-05, ics_loss: 0.5401609, res_loss: 0.0001040334, interval: 1.5674817562103271s, total: 8.803660869598389s
  step: 600, total loss: 0.43764904, bcs_loss: 0.00024841435, ics_loss: 0.43689, res_loss: 0.0005106426, interval: 1.2663447856903076s, total: 10.070005655288696s
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