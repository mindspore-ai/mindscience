ENGLISH | [简体中文](README_CN.md)

# Contents

- [PFNN Description](#DeepBSDE-description)
- [model architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)

## [PFNN Description](#contents)

PFNN (Penalty-free neural network) is a neural network-based differential equation solving method, which is suitable for solving second-order differential equations on complex regions. This method overcomes the shortcomings of existing similar methods in dealing with problem smoothness constraints and boundary constraints, and has higher accuracy, efficiency and stability.

[paper](https://www.sciencedirect.com/science/article/pii/S0021999120308597)：H. Sheng, C. Yang, PFNN: A penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries, Journal of Computational Physics 428 (2021) 110085.

## [model architecture](#Contents)

PFNN uses neural networks to approximate the solution of differential equations. Unlike most neural network methods that only use a single network to construct solution spaces, PFNN uses two networks to approximate true solutions on essential boundaries and other parts of the region, respectively. To eliminate the influence between the two networks, a length factor function constructed by the spline function is introduced to separate the two networks. In order to further reduce the smoothness requirements of the solution, PFNN uses Ritz variational principle to transform the problem into a weak form and eliminate the higher-order differential operator in the loss function, thereby reducing the difficulty of minimizing the loss function and improving the accuracy of the method.

## [Dataset](#contents)

PFNN generates training and test sets based on equation information and calculation area information.

- Training set: divided into an inner set and a boundary set, which are sampled inside the calculation area and on the boundary, respectively.
    - Inner Set: Inside the calculation area, 3600 points are sampled and the values of the right term of the control equation are calculated at these points as labels.
    - Boundary Set: 60 and 180 points were sampled on the Dirichlet boundary and the Neumann boundary, respectively, and the values of the right end terms of the boundary equation term at these points were calculated as labels.
- Test Set: 10201 points are sampled over the entire calculation area and the true solution values at these points are calculated as labels.

    Note: This dataset is used in the anisotropic diffusion equation scenario. The data will be processed in pfnn/Data/Data.py

## [Environment Requirements](#contents)

- Hardware(Ascend/GPU/CPU)
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https:#www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https:#www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https:#www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend or GPU

Default:

```bash
python train.py
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

File structures are as follows:

```text
├── data
│   ├── gendata.py             # generate data according to equations
│   └── dataset.py             # generate dataset
├── src
│   ├── callback.py            # callback module
│   ├── process.py             # training preparation
│   └── pfnnmodel.py           # network architecture
├── README_CN.md               # English model descriptions
├── README.md                  # Chinese model description
├── config.yaml                # hyper-parameters configuration
├── train.py                   # python training script
└── eval.py                    # python evaluation script
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in `config.yaml`

| parameter      | description                                        | default value                                                                      |
|----------------|----------------------------------------------------|------------------------------------------------------------------------------------|
| problem_type   | problem type                                       | 1                                                                                  |
| bound          | boundary                                           | [-1.0, 1.0, -1.0, 1.0]                                                             |
| inset_nx       | InnerSet num                                       | [60, 60]                                                                           |
| bdset_nx       | BoundarySet num                                    | [60, 60]                                                                           |
| teset_nx       | TestSet num                                        | [101, 101]                                                                         |
| g_epochs       | g_net epochs                                       | 6000                                                                               |
| f_epochs       | f_net epochs                                       | 6000                                                                               |
| g_lr           | g_net learning rate                                | 0.01                                                                               |
| f_lr           | f_net learning rate                                | 0.01                                                                               |
| tests_num      | testing num                                        | 5                                                                                  |
| log_path       | log saving path                                    | ./logs                                                                             |
| load_ckpt_path | checkpoint path                                    | [./checkpoints/optimal_state_g_pfnn.ckpt, ./checkpoints/optimal_state_f_pfnn.ckpt] |
| force_download | whether to enforce dataset and checkpoint download | false                                                                              |
| amp_level      | MindSpore auto mixed precision level               | O0                                                                                 |
| mode           | MindSpore Graph mode(0) or Pynative mode(1)        | 0                                                                                  |

### [Training Process](#contents)

  ```bash
  python train.py --problem_type [PROBLEM] --g_epochs [G_EPOCHS] --f_epochs [F_EPOCHS] --g_lr [G_LR] --f_lr [F_LR]
  ```

### [Evaluation Process](#contents)

  Before running the command below, please check `load_ckpt_path` used for evaluation in `config.yaml`. An example would be `./checkpoints/deepbsde_HJBLQ_end.ckpt`

  ```bash
  python eval.py
  ```
