# Contents

- [Hidden Fluid Mechanics Description](#hidden-fluid-mechanics-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Hidden Fluid Mechanics Description](#contents)

Hidden fluid mechanics(HFM) is a physics-informed deep-learning framework,
which is capable of extracting hidden quantities of fluid motion such as velocity and pressure fields
by encoding Navier-Stokes equations into neural networks. The detailed studies are demonstrated in the following papers.

> [paper](https://www.science.org/doi/abs/10.1126/science.aaw4741):
> Raissi M, Yazdani A, Karniadakis G E. Hidden fluid mechanics:
> Learning velocity and pressure fields from flow visualizations[J]. Science, 2020, 367(6481): 1026-1030.

This model reporduces the training process of neural network
and prediction of velocity field and pressure field, given time and position.
The scenario used is a 2-D flow past a circular cylinder at Reynolds number `Re=100` and Peclet number `Pe=100`.

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

Dataset used: [Cylinder2D_flower]

- Dataset size
    - time: (201, 1)
    - position x: (1500, 201)
    - position y: (1500, 201)
- Data format: `.mat` files
    - Note: Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   └── Cylinder2D_flower.mat
```

If you need to download the dataset or checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/hfm/).

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
    --layers 3 200 200 200 200 200 200 200 200 200 200 4 \
    --save_ckpt true \
    --load_ckpt false \
    --save_result true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_iter_300000_float32.ckpt \
    --load_data_path ./data \
    --save_data_path ./data \
    --log_path ./logs \
    --print_interval 10 \
    --lr 1e-3 \
    --t 1500 \
    --n 1500 \
    --total_time 40 \
    --epochs 100001 \
    --batch_size 1000 \
    --download_data hfm \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── hfm
│   ├── checkpoints          # checkpoints files
│   ├── data                 # data folder
│   ├── figures              # plot figures
│   ├── logs                 # log files
│   ├── src                  # source codes
│   │   ├── network.py       # neural network definition
│   │   └── process.py       # data and network preparations
│   ├── case_studies_1.sh    # shell script for simultaneous multi-case training
│   ├── case_studies_2.sh    # shell script for simultaneous multi-case training
│   ├── config.yaml          # hyper-parameters configuration
│   ├── README.md            # English model descriptions
│   ├── README_CN.md         # Chinese model description
│   ├── train.py             # python training script
│   └── eval.py              # python evaluation script
```

### [Script Parameters](#contents)

All parameters for running `train.py` are listed below.

| parameter      | description                                            | default value                                |
|----------------|--------------------------------------------------------|----------------------------------------------|
| layers         | neural network layer definition                        | 3 200 200 200 200 200 200 200 200 200 200 4  |
| save_ckpt      | whether save the checkpoints during training or not    | true                                         |
| load_ckpt      | whether load the checkpoint when define the net or not | false                                        |
| save_result    | whether save the results of the training or not        | true                                         |
| save_ckpt_path | checkpoint saving path                                 | ./checkpoints                                |
| load_ckpt_path | checkpoint loading path                                | ./checkpoints/model_iter_300000_float32.ckpt |
| load_data_path | folder to load the training data                       | ./data                                       |
| save_data_path | folder to save the generated data                      | ./data                                       |
| log_path       | folder to store the logs                               | ./logs                                       |
| print_interval | time and loss print interval                           | 10                                           |
| lr             | learning rate                                          | 1e-3                                         |
| t              | size of time sampling                                  | 1500                                         |
| n              | size of position sampling                              | 1500                                         |
| total_time     | maximum training time, unit: hour                      | 40                                           |
| epochs         | maximum training iterations                            | 100001                                       |
| batch_size     | batch size for each epoch                              | 1000                                         |
| download_data  | necessary dataset and/or checkpoints                   | hfm                                          |
| force_download | whether download the dataset or not by force           | false                                        |
| amp_level      | MindSpore auto mixed precision level                   | O0                                           |
| device_id      | device id to set                                       | None                                         |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)            | 0                                            |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  step: 0, loss: 0.8028861, interval: 210.2149157524109s, total: 210.2149157524109s
  step: 10, loss: 0.66187924, interval: 13.888249158859253s, total: 224.10316491127014s
  step: 20, loss: 0.45909613, interval: 13.550164461135864s, total: 237.653329372406s
  step: 30, loss: 0.21840161, interval: 13.551252603530884s, total: 251.2045819759369s
  step: 40, loss: 0.043125667, interval: 13.55091643333435s, total: 264.75549840927124s
  step: 50, loss: 0.04197544, interval: 13.552476167678833s, total: 278.3079745769501s
  step: 60, loss: 0.017915843, interval: 13.578445672988892s, total: 291.88642024993896s
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint loading path `load_ckpt_path` specified
in `config.yaml` for evaluation, you can download the checkpoint files according to the command
in [Dataset Section](#dataset).

- running on GPU/Ascend

  ```bash
  python eval.py
  ```

  It will output the errors for (c ,u, v, p) as follows:

  ```bash
  # grep "Error" log
  ...
  Error c: 2.638576e-02, Error u: 4.955575e-02, Error v: 3.927004e-02, Error p: 1.061887e-01
  Error c: 2.622087e-02, Error u: 4.833636e-02, Error v: 3.940436e-02, Error p: 1.045989e-01
  Error c: 2.596794e-02, Error u: 4.727550e-02, Error v: 3.953079e-02, Error p: 1.030758e-01
  Error c: 2.543095e-02, Error u: 4.638828e-02, Error v: 3.969464e-02, Error p: 1.016249e-01
  Error c: 2.459827e-02, Error u: 4.566803e-02, Error v: 3.989391e-02, Error p: 1.002600e-01
  Error c: 2.360513e-02, Error u: 4.509190e-02, Error v: 4.006676e-02, Error p: 9.900088e-02
  Error c: 2.243761e-02, Error u: 4.463641e-02, Error v: 4.011014e-02, Error p: 9.788122e-02
  ...
  ```

  You can view the process and results through the `log_path`, `./logs` by default.
