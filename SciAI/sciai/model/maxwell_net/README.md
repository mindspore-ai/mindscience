ENGLISH | [简体中文](README_CN.md)

# Contents

- [Maxwell Net Description](#maxwell-net-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Maxwell Net Description](#contents)

This work solves Maxwell's equations using physics-driven loss.
In other words, we are using the residual of Maxwell's equations as a loss function to train MaxwellNet,
therefore, it does not require ground truth solutions to train it. Furthermore, we utilized MaxwellNet in a
novel inverse design scheme, and we encourage you to refer to the main article for details.
<br />

![Scheme](/figures/scheme.png)

 <br />

> [paper](https://arxiv.org/abs/2107.06164):Lim J, Psaltis D. MaxwellNet: Physics-driven deep neural network training
> based on Maxwell’s equations[J]. Apl Photonics, 2022, 7(1).

## [Dataset](#contents)

The training dataset and pretrained checkpoint files will be downloaded automatically at the first launch.

Dataset Use:

- Dataset size
    - scat_pot: (1, 1, 160, 192)
    - ri: (1,)
- Data format: `.npz` files
    - Note: Data will be processed in `process.py`
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   ├── spheric_te
│   │   ├── sample.npz
│   │   └── train.npz
│   ├── spheric_tm
│   │   ├── sample.npz
│   │   └── train.npz
```

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

> Note: `tm` case does not support Pynative mode

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
    --problem te \
    --in_channels 1 \
    --out_channels 2 \
    --depth 6 \
    --filter 16 \
    --norm weight \
    --up_mode upconv \
    --wavelength 1 \
    --dpl 20 \
    --nx 160 \
    --nz 192 \
    --pml_thickness 30 \
    --symmetry_x true \
    --high_order 4 \
    --lr 0.0005 \
    --lr_decay 0.5 \
    --lr_decay_step 50000 \
    --epochs 250001 \
    --ckpt_interval 50000 \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/te_latest.ckpt \
    --load_data_path ./data/spheric_te \
    --save_fig true \
    --figures_path ./figures \
    --log_path ./logs \
    --download_data maxwell_net \
    --force_download false \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── maxwell_net
│   ├── checkpoints         # checkpoints files
│   ├── data                # data files
│   │   ├── spheric_te      # spheric te case data
│   │   └── spheric_tm      # spheric tm case data
│   ├── figures             # plot figures
│   ├── logs                # log files
│   ├── src                 # source codes
│   │   ├── network.py      # network architecture
│   │   ├── plot.py         # plotting results
│   │   └── process.py      # data process
│   ├── config.yaml         # hyper-parameters configuration
│   ├── README.md           # English model descriptions
│   ├── README_CN.md        # Chinese model description
│   ├── train.py            # training python script
│   └── eval.py             # evaluation python script
```

### [Script Parameters](#contents)

There are two problem cases. In `config.yaml` or command parameter, the case can be chosen by the parameter `problem`.

| parameter | description                                                                       | default value |
|-----------|-----------------------------------------------------------------------------------|---------------|
| problem   | problem case to be solved, `te`(transverse electric) or `tm`(transverse magnetic) | te            |

For each problem case, the parameters are as follows.

| parameter      | description                                                                                                                                                                         | default value                |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| in_channels    | UNet input channel                                                                                                                                                                  | 1                            |
| out_channels   | UNet output channel                                                                                                                                                                 | 2                            |
| depth          | UNet downsampling or upsampling depth                                                                                                                                               | 6                            |
| filter         | channel numbers in the first layer of UNet                                                                                                                                          | 16                           |
| norm           | type of normalization in UNet. 'weight' for weight normalization, 'batch' for batch normalization, 'no' for no normalization                                                        | weight                       |
| up_mode        | upsample mode of UNet. 'upcov' for transpose convolution, 'upsample' for upsampling                                                                                                 | upconv                       |
| wavelength     | wavelength                                                                                                                                                                          | 1                            |
| dpl            | one pixel size is 'wavelength / dpl'                                                                                                                                                | 20                           |
| nx             | pixel number along the x-axis, equivalent to the pixel number along the x-axis of scattering sample                                                                                 | 160                          |
| nz             | pixel number along the z-axis (light propagation direction), equivalent to the pixel number along the z-axis of scattering sample                                                   | 192                          |
| pml_thickness  | perfectly-matched-layer (PML) thickness in pixel number. 'pml_thickness * wavelength / dpl' is the actual thickness of PML layer in micrometers                                     | 30                           |
| symmetry_x     | whether input scattering sample is symmetric along the x-axis. If symmetry_x True, Nx=100, Nz=200 and symmetric along the x-axis, it suffices only half(Nx=50, Nz=200) in train.npz | true                         |
| high_order     | 2 or 4. It decides which order (2nd or 4th order) to calculate the gradient. 4 is more accurate than 2                                                                              | 4                            |
| lr             | learning rate                                                                                                                                                                       | 0.0005                       |
| lr_decay       | learning rate decay rate                                                                                                                                                            | 0.5                          |
| lr_decay_step  | learning rate decay step                                                                                                                                                            | 50000                        |
| epochs         | number of epochs for training                                                                                                                                                       | 250001                       |
| print_interval | time and loss print interval                                                                                                                                                        | 100                          |
| ckpt_interval  | checkpoint saving interval                                                                                                                                                          | 50000                        |
| save_ckpt      | whether save checkpoint or not                                                                                                                                                      | true                         |
| load_ckpt      | whether load checkpoint or not                                                                                                                                                      | false                        |
| save_ckpt_path | checkpoint saving path                                                                                                                                                              | ./checkpoints                |
| load_ckpt_path | checkpoint loading path                                                                                                                                                             | ./checkpoints/te_latest.ckpt |
| load_data_path | path to load original data                                                                                                                                                          | ./data/spheric_te            |
| save_fig       | whether save and plot figures or not                                                                                                                                                | true                         |
| figures_path   | figures saving path                                                                                                                                                                 | ./figures                    |
| log_path       | log saving path                                                                                                                                                                     | ./logs                       |
| download_data  | necessary dataset and/or checkpoints                                                                                                                                                | maxwell_net                  |
| force_download | whether download the dataset or not by force                                                                                                                                        | false                        |
| amp_level      | MindSpore auto mixed precision level                                                                                                                                                | O2                           |
| device_id      | device id to set                                                                                                                                                                    | None                         |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)                                                                                                                                         | 0                            |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  # grep "loss:" log
  step: 0, loss: 446.1874, interval: 89.43078088760376s, total: 89.43078088760376s, checkpoint saved at: ./checkpoints/model_iter_0_2000-12-31-23-59-59te.ckpt
  'latest' checkpoint saved at 0 epoch.
  step: 10, loss: 149.06134, interval: 1.5497097969055176s, total: 90.98049068450928s
  step: 20, loss: 83.69271, interval: 1.2006518840789795s, total: 92.18114256858826s
  step: 30, loss: 43.22249, interval: 1.1962628364562988s, total: 93.37740540504456s
  step: 40, loss: 33.38814, interval: 1.1976008415222168s, total: 94.57500624656677s
  step: 50, loss: 26.913471, interval: 1.1968715190887451s, total: 95.77187776565552s
  step: 60, loss: 20.579971, interval: 1.1951792240142822s, total: 96.9670569896698s
  step: 70, loss: 17.35663, interval: 1.197744369506836s, total: 98.16480135917664s
  step: 80, loss: 15.115046, interval: 1.2009501457214355s, total: 99.36575150489807s
  step: 90, loss: 12.830681, interval: 1.206284999847412s, total: 100.57203650474548s
  step: 100, loss: 11.197462, interval: 1.2086222171783447s, total: 101.78065872192383s
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
