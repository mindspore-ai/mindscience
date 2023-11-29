ENGLISH | [简体中文](README_CN.md)

# Contents

- [Conservative DeepLSTM Description](#conservative-DeepLSTM-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Conservative DeepLSTM Description](#contents)

<br />

![Scheme](/figures/scheme.png)

 <br />

> [paper](https://www.sciencedirect.com/science/article/abs/pii/S0045794919302263):Ruiyang Zhang, Zhao Chen, Su Chen, Jingwei Zheng, Oral Büyüköztürk, Hao Sun,
> Deep long short-term memory networks for nonlinear structural seismic response prediction, Computers & Structures,
> Volume 220, 2019, Pages 55-68, ISSN 0045-7949.

## [Dataset](#contents)

The training dataset can be downloaded from:
[link](https://www.dropbox.com/sh/xyh9595l79fbaer/AABnAqV_WdhVHgPAav73KX8oa?dl=0).

Dataset used: [burgers shock]

- Dataset size
    - data_BoucWen.mat(44MB)
    - data_MRFDBF.mat(564MB)
    - data_SanBernardino.mat(2.82MB)
- Data format: `.mat` files
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── data_BoucWen.mat
│   ├── data_MRFDBF.mat
│   └── data_SanBernardino.mat
```

If you need to download the dataset files manually,
please visit [this link](https://www.dropbox.com/sh/xyh9595l79fbaer/AABnAqV_WdhVHgPAav73KX8oa?dl=0).

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
    --dataset data_BoucWen.mat \
    --model lstm-s
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── deep_lstm
│   ├── data                # data files
│   ├── figures             # plot figures
│   ├── src                 # source codes
│   │   ├── network.py      # network architecture
│   │   └── utils.py        # data process
│   ├── README.md           # English model descriptions
│   ├── README_CN.md        # Chinese model description
│   ├── train.py            # python training script
│   └── eval.py             # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter | description            | default value    |
|-----------|------------------------|------------------|
| dataset   | dataset file name      | data_BoucWen.mat |
| model     | neural network version | lstm-s           |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  Train total epoch:  50000
  ---------------train start---------------
  step: 0 epoch: 0 batch: 0 loss: 0.08907777
  step time is 4.832167148590088
  step: 1 epoch: 0 batch: 1 loss: 0.09467506
  step time is 0.03872847557067871
  step: 2 epoch: 0 batch: 2 loss: 0.09016898
  step time is 0.04083538055419922
  step: 3 epoch: 0 batch: 3 loss: 0.08822981
  step time is 1.1140174865722656
  train_mse: 0.090537906  test_mse: 0.081995904
  step: 4 epoch: 1 batch: 0 loss: 0.0851737
  step time is 0.03679323196411133
  step: 5 epoch: 1 batch: 1 loss: 0.09027029
  step time is 0.02897500991821289
  ...
  ```

- After training, you can still review the training process through the log file saved in `log_path`, `./logs` directory
  by default.

- The model checkpoint will be saved in `save_dir`, `./results` directory by default.

### [Evaluation Process](#contents)

- running on GPU/Ascend

  ```bash
  python eval.py
  ```

  You can view the process and results through the `save_dir`by default.
  The result pictures are saved in `save_dir`, [`./results`](./results) by default.