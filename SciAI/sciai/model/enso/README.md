ENGLISH | [简体中文](README_CN.md)

# Contents

- [ENSO Description](#enso-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [ENSO Description](#contents)

El Niño/Southern Oscillation (ENSO) phenomenon has great impacts on regional ecosystem, and therefore, an accurate
forecasting of ENSO brings great regional benefits. However, forecasting ENSO with more than one years' horizon remains
problematic. Recently, the convolutional neural network (CNN) has been proven to be effective tool in forecasting ENSO.

In this model, we implemented the training and evaluation process of a CNN, which is used to forecast ENSO,
with meteorological data.

> [paper](https://doi.org/10.1038/s41586-019-1559-7): Ham, Y.-G., J.-H. Kim, and J.-J. Luo, 2019:
> Deep learning for multi-year ENSO forecasts. Nature, 573, 568–572.

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

- Data format: `.npy` files
    - Note: Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── htmp_data
│   ├── train_data
│   │   ├── ACCESS-CM2
│   │   ├── CCSM4
│   │   ├── CESM1-CAM5
│   │   ├── ...
│   │   └── obs
│   └── var_data
```

If you need to download the dataset or checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/enso/).

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
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/exp2_aftertrain/enso_float16.ckpt \
    --save_data true\
    --load_data_path ./data \
    --save_data_path ./data \
    --save_figure true \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 10 \
    --lr 0.01 \
    --epochs 20 \
    --batch_size 400 \
    --skip_aftertrain false \
    --epochs_after 5 \
    --batch_size_after 30 \
    --lr_after 1e-6 \
    --download_data enso \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── enso
│   ├── checkpoints                       # checkpoints files
│   ├── data                              # data folder
│   │   ├── htmp_data                     # save folder for validation result
│   │   ├── var_data                      # validation data
│   │   └── train_data                    # training data
│   ├── figures                           # plot figures
│   ├── logs                              # log files
│   ├── src                               # source codes
│   │   ├── network.py                    # neural network
│   │   ├── plot.py                       # plot functions
│   │   └── process.py                    # data preparation
│   ├── config.yaml                       # hyper-parameters configuration
│   ├── README.md                         # English model descriptions
│   ├── README_CN.md                      # Chinese model description
│   ├── train.py                          # python training script
│   └── eval.py                           # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter        | description                                  | default value                                   |
|------------------|----------------------------------------------|-------------------------------------------------|
| save_ckpt        | whether save checkpoint or not               | true                                            |
| load_ckpt        | whether load checkpoint or not               | false                                           |
| save_ckpt_path   | checkpoint saving path                       | ./checkpoints                                   |
| load_ckpt_path   | checkpoint loading path                      | ./checkpoints/exp2_aftertrain/enso_float16.ckpt |
| save_data        | whether save data output or not              | true                                            |
| load_data_path   | path to load data                            | ./data                                          |
| save_data_path   | path to save data                            | ./data                                          |
| save_figure      | whether save and plot figures or not         | true                                            |
| figures_path     | figures saving path                          | ./figures                                       |
| log_path         | log saving path                              | ./logs                                          |
| print_interval   | interval for time and loss printing          | 10                                              |
| lr               | learning rate                                | 0.01                                            |
| epochs           | number of epochs                             | 20                                              |
| batch_size       | size of data batch                           | 400                                             |
| skip_aftertrain  | whether skip the after train process         | false                                           |
| epochs_after     | number of epochs in after train              | 5                                               |
| batch_size_after | size of data batch for after train           | 30                                              |
| lr_after         | learning rate for after train                | 1e-6                                            |
| download_data    | necessary dataset and/or checkpoints         | enso                                            |
| force_download   | whether download the dataset or not by force | false                                           |
| amp_level        | MindSpore auto mixed precision level         | O3                                              |
| device_id        | device id to set                             | None                                            |
| mode             | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                               |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # python train.py
  ...
  epoch: 1 step: 1, loss is 0.9130635857582092
  epoch: 1 step: 2, loss is 1.0354164838790894
  epoch: 1 step: 3, loss is 0.8914494514465332
  epoch: 1 step: 4, loss is 0.9377754330635071
  epoch: 1 step: 5, loss is 1.0472232103347778
  epoch: 1 step: 6, loss is 1.0421113967895508
  epoch: 1 step: 7, loss is 1.100639820098877
  epoch: 1 step: 8, loss is 0.9849204421043396
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_ckpt_path`, `./checkpoint` directory by default.

### [Evaluation Process](#contents)

Before running the command below, please check the checkpoint loading path `load_ckpt_path` specified
in `config.yaml` for evaluation.

```bash
python eval.py
```

You can view the process and results through the `log_path`, `./logs` by default.
The result pictures are saved in `figures_path`, [`./figures`](./figures) by default.
