ENGLISH | [简体中文](README_CN.md)

# Contents

- [MgNet Description](#mgnet-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [MgNet Description](#contents)

MgNet is a unified model that simultaneously recovers some convolutional neural networks (CNN) for image classification
and multigrid (MG) methods for solving discretized partial differential equations (PDEs). Here is a diagram of its
architecture.

 <img src="./figures/MgNet.png" width = "300" align=center />

> [paper](https://link.springer.com/article/10.1007/s11425-019-9547-2): He J, Xu J. MgNet: A unified framework of
> multigrid and convolutional neural network[J]. Science china mathematics, 2019, 62: 1331-1354.

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

Dataset used: [cifar10]/[cifar100]/[mnist]

- The dataset should be downloaded and unzipped in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── cifar-10-batches-bin
│   ├── cifar-100-binary
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
```

If you need to download the dataset or checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/mgnet/).

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
    --dataset cifar100 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_300.ckpt \
    --load_data_path ./data/cifar-100-binary \
    --log_path ./logs \
    --print_interval 10 \
    --ckpt_interval 2000 \
    --num_ite 2 2 2 2 \
    --num_channel_u 256 \
    --num_channel_f 256 \
    --wise_b true \
    --batch_size 128 \
    --epochs 300 \
    --lr 1e-1 \
    --download_data mgnet \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── mgnet
│   ├── checkpoints                                 # checkpoints files
│   ├── data                                        # data files
│   │   ├── cifar-10-batches-bin                    # cifar-10 dataset directory
│   │   ├── cifar-100-binary                        # cifar-100 dataset directory
│   │   ├── t10k-images.idx3-ubyte                  # mnist test data images
│   │   ├── t10k-labels.idx1-ubyte                  # mnist test data labels
│   │   ├── train-images.idx3-ubyte                 # mnist training data images
│   │   └──  train-labels.idx1-ubyte                # mnist training data labels
│   ├── figures                                     # figures directory
│   ├── logs                                        # log files
│   ├── src                                         # source codes
│   │   ├── network.py                              # network architecture
│   │   └── process.py                              # data process
│   ├── config.yaml                                 # hyper-parameters configuration
│   ├── README.md                                   # English model descriptions
│   ├── README_CN.md                                # Chinese model description
│   ├── requirements.txt                            # library requirements for this model
│   ├── train.py                                    # python training script
│   └── eval.py                                     # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                                          | default value                |
|----------------|----------------------------------------------------------------------|------------------------------|
| dataset        | dataset name to load, can be cifar10, cifar100, or mnist             | cifar100                     |
| save_ckpt      | whether save checkpoint or not                                       | true                         |
| load_ckpt      | whether load checkpoint or not                                       | false                        |
| save_ckpt_path | checkpoint saving path                                               | ./checkpoints                |
| load_ckpt_path | checkpoint loading path                                              | ./checkpoints/model_300.ckpt |
| load_data_path | path to load data                                                    | ./data/cifar-100-binary      |
| log_path       | log saving path                                                      | ./logs                       |
| print_interval | time and loss print interval                                         | 10                           |
| ckpt_interval  | checkpoint save interval                                             | 2000                         |
| num_ite        | the number of ite: in four level(layer), use with 2 2 2 2 or 3 4 5 6 | 2 2 2 2                      |
| num_channel_u  | number of channels of u                                              | 256                          |
| num_channel_f  | number of channels of f                                              | 256                          |
| wise_b         | different B in different grid                                        | true                         |
| lr             | learning rate                                                        | 1e-1                         |
| epochs         | number of epochs                                                     | 300                          |
| batch_size     | batch size                                                           | 128                          |
| download_data  | necessary dataset and/or checkpoints                                 | mgnet                        |
| force_download | whether download the dataset or not by force                         | false                        |
| amp_level      | MindSpore auto mixed precision level                                 | O0                           |
| device_id      | device id to set                                                     | None                         |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)                          | 0                            |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  total 10661220 parameters
  start training ...
  epoch: 0/300, step: 10/390, loss:5.592, interval: 66.312 ms, total: 7.027 s
  epoch: 0/300, step: 20/390, loss:6.389, interval: 66.504 ms, total: 7.725 s
  epoch: 0/300, step: 30/390, loss:4.639, interval: 83.561 ms, total: 8.464 s
  epoch: 0/300, step: 40/390, loss:4.428, interval: 68.359 ms, total: 9.181 s
  epoch: 0/300, step: 50/390, loss:4.408, interval: 70.381 ms, total: 9.897 s
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint loading path `load_ckpt_path` specified
in `config.yaml` for evaluation. You can download the checkpoint files as described [here](#dataset).

- running on GPU/Ascend

```bash
python eval.py
```

You can view the process and results through the `log_path`, `./logs` by default.
The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.