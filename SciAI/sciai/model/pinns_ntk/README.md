ENGLISH | [简体中文](README_CN.md)

# Contents

- [PINNs Neural Tangent Kernel Description](#pinns-neural-tangent-kernel-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [PINNs Neural Tangent Kernel Description](#contents)

Neural Tangent Kernel (NTK) is a kernel that captures the behavior of fully-connected neural networks in the infinite
width limit during training via gradient descent.
The NTK of PINNs, under appropriate conditions, converges to a deterministic kernel that stays constant during training
in the infinite-width limit.
This allows us to analyze the training dynamics of PINNs through the lens of their limiting NTK and find a remarkable
discrepancy in the convergence rate of the different loss components contributing to the total training error.

> [paper](https://www.sciencedirect.com/science/article/pii/S002199912100663X): Sifan Wang, Xinling Yu, Paris
> Perdikaris, When and why PINNs fail to train: A neural tangent kernel perspective,
> Journal of Computational Physics, Volume 449, 2022, 110768, ISSN 0021-9991.

Example details: PINNs Neural Tangent Kernel code for 1D Poisson distribution.

## [Dataset](#contents)

The dataset is generated during runtime.
The size of dataset is controlled by parameter `num` in `config.yaml`, and by default is 100.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pinns_ntk/).

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
    --layers 1 512 1 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final_float32.ckpt \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --num 100 \
    --lr 1e-4 \
    --epochs 40001 \
    --download_data pinns_ntk \
    --force_download false \
    --amp_level O2 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── pinns_ntk
│   ├── checkpoints                # checkpoints files
│   ├── data                       # data files
│   ├── figures                    # plot figures
│   ├── logs                       # log files
│   ├── src                        # source codes
│   │   ├── network.py             # network architecture
│   │   ├── plot.py                # plotting results
│   │   └── process.py             # data process
│   ├── config.yaml                # hyper-parameters configuration
│   ├── README.md                  # English model descriptions
│   ├── README_CN.md               # Chinese model description
│   ├── train.py                   # python training script
│   └── eval.py                    # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                  | default value                          |
|----------------|----------------------------------------------|----------------------------------------|
| layers         | neural network layers                        | 1 512 1                                |
| save_ckpt      | whether save checkpoint or not               | true                                   |
| save_fig       | whether save and plot figures or not         | true                                   |
| load_ckpt      | whether load checkpoint or not               | false                                  |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                          |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/model_final_float32.ckpt |
| figures_path   | figures saving path                          | ./figures                              |
| log_path       | log saving path                              | ./logs                                 |
| print_interval | time and loss print interval                 | 10                                     |
| num            | dataset sampling number                      | 100                                    |
| lr             | learning rate                                | 1e-4                                   |
| epochs         | number of epochs                             | 40001                                  |
| download_data  | necessary dataset and/or checkpoints         | pinns_ntk                              |
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
  step: 0, total loss: 12353.368, loss_bcs: 9.613263, loss_res: 12343.755, interval: 0.41385459899902344s, total: 0.41385459899902344s, checkpoint saved at: ./checkpoints/model_iter_0_2023-04-23-07-42-46.ckpt
  Compute NTK...
  Weigts stored...
  step: 10, total loss: 11993.846, loss_bcs: 8.270422, loss_res: 11985.224, interval: 1.02224523987624589s, total: 1.43609983887526933s
  step: 20, total loss: 11339.517, loss_bcs: 6.435134, loss_res: 11333.08, interval: 0.024523986245987602s, total: 1.460623825121256932s
  step: 30, total loss: 11287.906, loss_bcs: 6.306574, loss_res: 11281.6, interval: 0.0191900713459287945s, total: 1.4798138964671857265s
  step: 40, total loss: 6723.454, loss_bcs: 2.0566676, loss_res: 6721.3975, interval: 0.01975234587509485s, total: 1.4995662423422805765s
  step: 50, total loss: 8277.567, loss_bcs: 0.0453923, loss_res: 8277.522, interval: 0.01824523876245972s, total: 1.5178114811047402965s
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
  The result pictures are saved in `figures_path`, by default is [`./figures`](./figures).