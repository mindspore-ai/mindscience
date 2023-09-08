ENGLISH | [简体中文](README_CN.md)

# Contents

- [Helmholtz PINNs Description](#helmholtz-pinns-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Helmholtz PINNs Description](#contents)

The following paper applies physics-informed neural networks (PINNs) to solve the Helmholtz equation
for isotropic and anisotropic media. The network uses sine as activation function, as it is effective in
solving time- and freqeuncy-domain wave equations.

This repository implements the neural networks with fixed-sine activation function to solve
the helmholtz equation with numerical data collected from isotropic Marmoussi model.

> [paper](https://academic.oup.com/gji/article-abstract/228/3/1750/6409132):
> Song C, Alkhalifah T, Waheed U B.
> A versatile framework to solve the Helmholtz equation using physics-informed neural networks[J].
> Geophysical Journal International, 2022, 228(3): 1750-1762.

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

The dataset is collected with isotropic Marmousi model, which is identical to the original paper.
The single source is placed on the surface at location 4.625km. The sampling interval is 25m
in both vertical and horizontal direction.

- Data format: `.mat` files
    - Note: Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   └── Marmousi_3Hz_singlesource_ps.mat
```

If you need to download the dataset and checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pinn_helmholtz/).

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
    --layers 2 40 40 40 40 40 40 40 40 2 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_adam_100000_float16.ckpt \
    --load_data_path ./data \
    --save_fig true \
    --figures_path ./figures \
    --save_results true \
    --results_path ./data/results \
    --print_interval 20 \
    --log_path ./logs \
    --lr 0.001 \
    --epochs 100000 \
    --num_batch 1 \
    --lbfgs false \
    --epochs_lbfgs 50000 \
    --download_data pinn_helmholtz \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── pinn_helmholtz
│   ├── checkpoints                                 # checkpoints files
│   ├── data                                        # data files
│   │   ├── results                                 # results of training
│   │   └── Marmousi_3Hz_singlesource_ps.mat        # Marmousi dataset with 3Hz single source
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

| parameter      | description                                  | default value                                |
|----------------|----------------------------------------------|----------------------------------------------|
| layers         | neural network widths                        | 2 40 40 40 40 40 40 40 40 2                  |
| save_ckpt      | whether save checkpoint or not               | true                                         |
| load_ckpt      | whether load checkpoint or not               | false                                        |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                                |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/model_adam_100000_float32.ckpt |
| load_data_path | path to load data                            | ./data                                       |
| save_fig       | whether save and plot figures or not         | true                                         |
| figures_path   | figures saving path                          | ./figures                                    |
| save_results   | whether save the prediction and loss         | true                                         |
| results_path   | path to prediction and loss                  | ./data/results                               |
| print_interval | time and loss print interval                 | 20                                           |
| log_path       | log saving path                              | ./logs                                       |
| lr             | learning rate                                | 1e-3                                         |
| epochs         | number of epochs                             | 100000                                       |
| num_batch      | number of batches                            | 1                                            |
| lbfgs          | whether using lbfgs or not                   | false                                        |
| epochs_lbfgs   | number of epochs for lbfgs                   | 50000                                        |
| download_data  | necessary dataset and/or checkpoints         | pinn_helmholtz                               |
| force_download | whether download the dataset or not by force | false                                        |
| amp_level      | MindSpore auto mixed precision level         | O0                                           |
| device_id      | device id to set                             | None                                         |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                            |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  ...
  step: 40, loss: 2.0659282, interval: 16.741257190704346s, total: 63.9092378616333s
  step: 60, loss: 1.27979, interval: 16.732836723327637s, total: 80.64207458496094s
  step: 80, loss: 1.0679382, interval: 16.754377841949463s, total: 97.3964524269104s
  step: 100, loss: 0.96829647, interval: 16.702229022979736s, total: 114.09868144989014s
  step: 120, loss: 0.9059235, interval: 16.710976123809814s, total: 130.80965757369995s
  step: 140, loss: 0.86077166, interval: 16.749966621398926s, total: 147.55962419509888s
  step: 160, loss: 0.825172, interval: 16.73036813735962s, total: 164.2899923324585s
  step: 180, loss: 0.7951189, interval: 16.77035140991211s, total: 181.0603437423706s
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