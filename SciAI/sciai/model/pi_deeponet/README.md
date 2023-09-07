ENGLISH | [简体中文](README_CN.md)

# Contents

- [Physics-informed DeepONets Description](#physics-informed-deeponets-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [Physics-informed DeepONets Description](#contents)

This project learns infinite-dimensional operators that map random initial conditions to associated PDE solutions
within a short time interval. Global long-time predictions across a range of initial conditions can be obtained by
iteratively evaluating the trained model using each prediction as the initial condition for the next evaluation step.
This introduces a new approach to temporal domain decomposition that is shown to be effective in performing accurate
long-time simulations for a wide range of parametric PDEs systems, from wave propagation, to reaction-diffusion dynamics
and stiff chemical kinetics, all at a fraction of the computational cost needed by classical numerical solvers.

> [paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999122009184): Wang S, Perdikaris P. Long-time
> integration of parametric evolution equations with physics-informed deeponets[J]. Journal of Computational Physics,
> 2023, 475: 111855.

## [Dataset](#contents)

The dataset is generated during runtime.
The size of dataset is controlled by parameter `batch_size` in `config.yaml`, and by default is 10000.

The dataset for validation and pretrained checkpoint files will be downloaded automatically at the first launch.
If you need to download the validation dataset and checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/pi_deeponet/).

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
    --branch_layers 100 100 100 100 100 100 \
    --trunk_layers 2 100 100 100 100 100 \
    --save_ckpt true \
    --save_data true \
    --save_fig true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/dr_float32_final.ckpt \
    --save_data_path ./data \
    --figures_path ./figures \
    --log_path ./logs \
    --print_interval 100 \
    --lr 8e-4 \
    --epochs 200001 \
    --n_train 10000 \
    --batch_size 10000 \
    --download_data pi_deeponet \
    --force_download false \
    --amp_level O3 \
    --device_id 0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
├── pi_deeponet
│   ├── checkpoints                  # checkpoints files
│   ├── data                         # data files
│   ├── figures                      # plot figures
│   ├── logs                         # log files
│   ├── src                          # source codes
│   │   ├── network.py               # network architecture
│   │   ├── plot.py                  # plotting results
│   │   └── process.py               # data process
│   ├── config.yaml                  # hyper-parameters configuration
│   ├── README.md                    # English model descriptions
│   ├── README_CN.md                 # Chinese model description
│   ├── train.py                     # python training script
│   └── eval.py                      # python evaluation script
```

### [Script Parameters](#contents)

Important parameters in train.py are as follows:

| parameter      | description                                  | default value                       |
|----------------|----------------------------------------------|-------------------------------------|
| branch_layers  | branch neural network layer widths           | 100 100 100 100 100 100             |
| trunk_layers   | trunck neural network layer widths           | 2 100 100 100 100 100               |
| save_ckpt      | whether save checkpoint or not               | true                                |
| save_data      | whether save data or not                     | true                                |
| save_fig       | whether save and plot figures or not         | true                                |
| load_ckpt      | whether load checkpoint or not               | false                               |
| save_ckpt_path | checkpoint saving path                       | ./checkpoints                       |
| load_ckpt_path | checkpoint loading path                      | ./checkpoints/dr_float32_final.ckpt |
| save_data_path | path to save data                            | ./data                              |
| figures_path   | figures saving path                          | ./figures                           |
| log_path       | log saving path                              | ./logs                              |
| print_interval | time and loss print interval                 | 100                                 |
| lr             | learning rate                                | 8e-4                                |
| epochs         | number of epochs                             | 200001                              |
| n_train        | times of generating training data            | 10000                               |
| batch_size     | batch size                                   | 10000                               |
| download_data  | necessary dataset and/or checkpoints         | pi_deeponet                         |
| force_download | whether download the dataset or not by force | false                               |
| amp_level      | MindSpore auto mixed precision level         | O3                                  |
| device_id      | device id to set                             | None                                |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)  | 0                                   |

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

  The loss values during training will be printed in the console, which can also be inspected after training in log
  file.

  ```bash
  # grep "loss:" log
  step: 0, total loss: 0.0971143, ic_loss: 0.0706561, bc_loss: 0.0047654584, res_loss: 0.021692745, interval: 9.959319114685059s, total: 9.959319114685059s
  step: 100, total loss: 0.0612279, ic_loss: 0.056949332, bc_loss: 0.0030672695, res_loss: 0.0012112984, interval: 9.085834741592407s, total: 19.045153856277466s
  step: 200, total loss: 0.059222076, ic_loss: 0.05515627, bc_loss: 0.0030886163, res_loss: 0.0009771917, interval: 9.108893632888794s, total: 28.15404748916626s
  step: 300, total loss: 0.05733742, ic_loss: 0.052310925, bc_loss: 0.003073115, res_loss: 0.0019533802, interval: 9.576531648635864s, total: 37.730579137802124s
  step: 400, total loss: 0.055052415, ic_loss: 0.049479727, bc_loss: 0.0032956824, res_loss: 0.0022770043, interval: 10.003910541534424s, total: 47.73448967933655s
  step: 500, total loss: 0.051897146, ic_loss: 0.047461353, bc_loss: 0.0025362624, res_loss: 0.0018995304, interval: 9.252656698226929s, total: 56.98714637756348s
  step: 600, total loss: 0.047137313, ic_loss: 0.04395392, bc_loss: 0.0014622104, res_loss: 0.0017211806, interval: 9.413921594619751s, total: 66.40106797218323s
  step: 700, total loss: 0.050823156, ic_loss: 0.044430587, bc_loss: 0.0040090764, res_loss: 0.002383494, interval: 9.160758018493652s, total: 75.56182599067688s
  step: 800, total loss: 0.029433459, ic_loss: 0.026467426, bc_loss: 0.00096403103, res_loss: 0.0020020034, interval: 8.86798882484436s, total: 84.42981481552124s
  step: 900, total loss: 0.0065431646, ic_loss: 0.0051204017, bc_loss: 0.0005367383, res_loss: 0.00088602427, interval: 9.333975076675415s, total: 93.76378989219666s
  step: 1000, total loss: 0.004916694, ic_loss: 0.0040391637, bc_loss: 0.00033330295, res_loss: 0.00054422737, interval: 9.447664737701416s, total: 103.21145462989807s
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
  The result pictures are saved in `figures_path`, [`./figures`](./figures) by default. will be saved in `save_ckpt_path`
  , `./checkpoint` directory by default.