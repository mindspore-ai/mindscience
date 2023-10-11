ENGLISH | [简体中文](README_CN.md)

# Contents

- [GOMO Description](#GOMO Description)
- [Dataset](#Dataset)
- [Environment Requirements](#Environment-Requirements)
- [Quick Start](#Quick-Start)
- [Script Description](#Script-Description)
    - [Script and Sample Code](#Script-and-Sample-Code)
    - [Training Process](#Training-Process)
- [Model Description](#Model-Description)
    - [Evaluation Performance](#Evaluation-Performance)

## [GOMO Description](#contents)

Generalized Operator Modelling of the Ocean (GOMO) is a three-dimensional ocean model based on [OpenArray v1.0](https://gmd.copernicus.org/articles/12/4729/2019/gmd-12-4729-2019-discussion.html) which is a simple operator library for the decoupling of ocean modelling and parallel computing (Xiaomeng Huang et al, 2019). GOMO is a numerical solution model using finite differential algorithm to solve PDE equations. With MindSpore and GPU/Ascend, we can achieve great performance improvements in solving those PDE equations compared with CPU.

## [Dataset](#contents)

Dataset used: Seamount

- Dataset size: 65x49x21
- Data format：`.nc` file
- The dataset is in the `./data` directory, the directory structure is as follows:

```text
├── data
│   └── seamount65_49_21.nc
```

If you need to download the dataset or checkpoint files manually,
please visit [this link](https://download.mindspore.cn/mindscience/SciAI/sciai/model/ocean_model/).

## [Environment Requirements](#contents)

- Hardware: GPU
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website and the required [dataset](#dataset) above, you can start training
as follows:

- running on Ascend or on GPU

Default:

```bash
python train.py
```

Full command:

```bash
python train.py \
    --load_data_path ./data \
    --output_path ./data/outputs \
    --save_ckpt true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final.ckpt \
    --force_download false \
    --download_data ocean_model \
    --im 65 \
    --jm 49 \
    --kb 21 \
    --stencil_width 1 \
    --epochs 10 \
    --amp_level O0 \
    --mode 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
└── ocean_model
│   ├── checkpoints                  # checkpoints files
│   ├── data
│   │   └── seamount65_49_21.nc      # dataset file
│   ├── logs                         # log files
│   ├──src
│   │   ├── GOMO.py                  # GOMO model
│   │   ├── Grid.py                  # grid initial
│   │   ├── stencil.py               # averaging and differential stencil operator
│   │   ├── oa_operator.py           # averaging and differential kernel operator
│   │   ├── read_var.py              # read variables from nc file
│   │   └── utils.py                 # model setup
│   ├── config.yaml                  # hyper-parameters configuration
│   ├── README.md                    # English model descriptions
│   ├── README_CN.md                 # Chinese model description
│   ├── train.py                     # python training script
│   └── eval.py                      # python evaluation script
```

### [Training Process](#contents)

- running on GPU/Ascend

  ```bash
  python train.py
  ```

## Model Description

Model parameters are defined as follows:

| parameter      | description                                            | default value                  |
|----------------|--------------------------------------------------------|--------------------------------|
| load_data_path | path to load data                                      | ./data                         |
| output_path    | path to save result                                    | ./data/outputs                 |
| save_ckpt      | whether save checkpoint or not                         | true                           |
| save_ckpt_path | checkpoint saving path                                 | ./checkpoints                  |
| load_ckpt_path | checkpoint loading path                                | ./checkpoints/model_final.ckpt |
| force_download | whether to enforce dataset download                    | false                          |
| download_data  | the model of which dataset/checkpoint need to download | ocean_model                    |
| amp_level      | MindSpore auto mixed precision level                   | O0                             |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)            | 0                              |
| device_id      | device id to set                                       | None                           |
| im             | GOMO parameter im                                      | 65                             |
| jm             | GOMO parameter jm                                      | 49                             |
| kb             | GOMO parameter kb                                      | 21                             |
| stencil_width  | stencil width as in stencil computation                | 1                              |
| epochs         | number of epochs                                       | 10                             |

### [Evaluation Performance](#contents)

| parameter           | value                           |
|---------------------|---------------------------------|
| Resource            | GPU(Tesla V100 SXM2)，Memory 16G |
| Dataset             | Seamount                        |
| Training Parameters | step=10, im=65, km=49, kb=21    |
| Outputs             | numpy file                      |
| Speed               | 17 ms/step                      |
| Total time          | 3 mins                          |
