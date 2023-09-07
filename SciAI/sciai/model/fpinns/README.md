ENGLISH | [简体中文](README_CN.md)

# Contents

- [Fractional PINNs Description](#fractional-pinns-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Fractional PINNs Description](#contents)

To solve space-time fractional advection-diffusion equations (fractional ADEs), "Fractional PINNs" construct the
residual
in the loss function using both automatic differentiation for the integer-order operators and numerical discretization
for the fractional operators. This approach bypasses the difficulties stemming from the fact that automatic
differentiation is not applicable to fractional operators because the standard chain rule in integer calculus is not
valid in fractional calculus, and obtain accurate results given proper initializations even in the presence of
significant noise.

> [paper](https://arxiv.org/abs/1811.08967): Pang G, Lu L, Karniadakis G E. fPINNs: Fractional physics-informed neural
> networks[J]. SIAM Journal on Scientific Computing, 2019, 41(4): A2603-A2626.

Example details: Fractional PINNs code for 1-D Diffusion model.

## [Dataset](#contents)

The dataset used for training is randomly generated during training in each problem case.
The size of dataset depends on the number of samples. which are controlled by `num_domain`, `num_boundary`
and `num_initial` in `config.yaml`.

The pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/fpinns/).

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official, you can start training
and evaluation as follows:

- running on Ascend or on GPU

Default:

```bash
python train.py
```

Full command:

```bash
python train.py \
    --problem fractional_diffusion_1d \
    --layers 2 20 20 20 20 1 \
    --x_range 0 1 \
    --t_range 0 1 \
    --num_domain 400 \
    --num_boundary 0 \
    --num_initial 0 \
    --num_test 400 \
    --lr 1e-3 \
    --save_ckpt true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints/fractional_diffusion_1d \
    --load_ckpt_path ./checkpoints/fractional_diffusion_1d/model_iter_10000_float32.ckpt \
    --figures_path ./figures/fractional_diffusion_1d \
    --log_path ./logs \
    --print_interval 100 \
    --epochs 10001 \
    --download_data fpinns \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── fpinns
│   ├── checkpoints          # checkpoints files
│   ├── data                 # data files
│   ├── figures              # plot figures
│   ├── logs                 # log files
│   ├── src                  # source codes
│   │   ├── dataset.py       # generate random points
│   │   ├── net.py           # architecture of the net
│   │   ├── problem.py       # abstract problem base class
│   │   └── process.py       # process definition for two cases
│   ├── config.yaml          # hyper-parameters configuration
│   ├── README.md            # English model descriptions
│   ├── README_CN.md         # Chinese model description
│   ├── train.py             # python training script
│   └── eval.py              # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in `train.py` are as follows:

| parameter      | description                                  | default value                                                       |
|----------------|----------------------------------------------|---------------------------------------------------------------------|
| problem        | problem cases                                | fractional_diffusion_1d                                             |
| layers         | layer structure                              | 2 20 20 20 20 1                                                     |
| x_range        | the range of space                           | 0 1                                                                 |
| t_range        | the range of time                            | 0 1                                                                 |
| num_domain     | number of domains                            | 400                                                                 |
| num_boundary   | number of boundary datasets                  | 0                                                                   |
| num_initial    | number of initial datasets                   | 0                                                                   |
| num_test       | number of tests                              | 400                                                                 |
| lr             | learning rate                                | 1e-3                                                                |
| save_ckpt      | whether save checkpoint or not               | true                                                                |
| save_fig       | whether save and plot figures or not         | true                                                                |
| load_ckpt      | whether load checkpoint or not               | false                                                               |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints/fractional_diffusion_1d                               |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/fractional_diffusion_1d/model_iter_10000_float32.ckpt |
| figures_path   | figures saving path                          | ./figures/fractional_diffusion_1d                                   |
| log_path       | log saving path                              | ./logs                                                              |
| print_interval | time and loss print interval                 | 100                                                                 |
| epochs         | number of epochs                             | 10001                                                               |
| download_data  | necessary dataset and/or checkpoints         | fpinns                                                              |
| force_download | whether download the dataset or not by force | false                                                               |
| amp_level      | MindSpore auto mixed precision level         | O0                                                                  |
| device_id      | device id to set                             | None                                                                |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                                                   |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  step: 0, loss: 0.1720775, interval: 0.515510082244873s, total: 0.515510082244873s
  step: 100, loss: 0.004217985, interval: 0.15668702125549316s, total: 0.6721971035003662s
  step: 200, loss: 0.0026953542, interval: 0.14049434661865234s, total: 0.8126914501190186s
  step: 300, loss: 0.002297479, interval: 0.13532018661499023s, total: 0.9480116367340088s
  step: 400, loss: 0.0018170077, interval: 0.13717007637023926s, total: 1.085181713104248s
  step: 500, loss: 0.0009912008, interval: 0.1338338851928711s, total: 1.2190155982971191s
  step: 600, loss: 0.00050001504, interval: 0.14569568634033203s, total: 1.3647112846374512s
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