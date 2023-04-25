
# Table of Contents

<!-- TOC -->

- [Table of Contents](#table-of-contents)
    - [Maxwell's Equations in Frequency Domain](#maxwells-equations-in-frequency-domain)
    - [Solving Maxwell's Equations in Frequency Domain with AI](#solving-maxwells-equations-in-frequency-domain-with-ai)
    - [Datasets](#datasets)
    - [Environmental Requirements](#environmental-requirements)
    - [Script Description](#script-description)
        - [Script and Code Sample](#script-and-code-sample)
        - [Script Parameters](#script-parameters)
    - [Model Training](#model-training)
    - [Random Seed Setting](#random-seed-setting)
    - [MindScience Home Page](#mindscience-home-page)

<!-- /TOC -->

## Maxwell's Equations in Frequency Domain

The Maxwell's equations in frequency domain is an elliptic partial differential equation describing electromagnetic waves. Its basic form is as follows:

$$(\nabla^2 + k^2)u=0$$

where $k=\omega c$ indicates the wavenumber, $\omega$ indicates the frequency and $c$ indicates the light speed.

## Solving Maxwell's Equations in Frequency Domain with AI

The overall network architecture for AI to solve the Maxwell's equations in frequency domain is as follows:

![network_architecture](./docs/pinns_for_frequency_domain_maxwell.png)

Taking the two-dimensional Maxwell's equations in frequency domain as an example, the network input is $\Omega=(x, y)\in [0,1]^2$, and the network output is the PDE solution $u(x, y)$. The training loss function of the network can be constructed based on the network output and the automatic differentiation ability of the MindSpore framework. The loss function consists of PDE and BC parts:
$$L_{pde}= \dfrac{1}{N_1}\sum_{i=1}^{N_1} ||(\nabla^2 + k^2)u(x_i, y_i)||^2$$
$$L_{bc} = \dfrac{1}{N_2}\sum_{i=1}^{N_2} ||u(x_i, y_i)||^2$$
In order to ensure the uniqueness of the solution of the above equations, the boundary condition  is $u_{|\partial \Omega}=\sin(kx)$. You can customize the constant wavenumber $k$. In this case, the value is $k=2$.

## Datasets

AI uses self-supervised training to solve the Maxwell's equations in frequency domain. Data sets are generated in real time during the running. The training and inference data generation modes are as follows:

- Training data: In each iteration, 128 sample points are selected from a uniform grid of 101 x 101 within the feasible domain to calculate a PDE part $L_{pde}$ of the loss function. The BC part $L_{bc}$ of the loss function is calculated by randomly generating 128 sample points on the boundary.
- Evaluation data: Generate 101 x 101 uniform grid points within the feasible domain, and the corresponding label is the analytical solution of the equation $u=\sin(kx)$.
    - Note: Data is processed in src/dataset.py.

## Environmental Requirements

- Hardware (Ascend)
    - Prepare the Ascend AI Processor to set up the hardware environment.
- Framework
    - [MindSpore Elec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- For more information, see the following resources:
    - [MindSpore Elec Tutorial](https://www.mindspore.cn/mindelec/docs/en/master/intro_and_install.html)
    - [MindSpore Elec Python API](https://www.mindspore.cn/mindelec/docs/en/master/mindelec.architecture.html)

## Script Description

### Script and Code Sample

```path
.
└─FrequencyDomainMaxwell
  ├─README.md
  ├─docs                              # schematic diagram of README
  ├─src
    ├──callback.py                    # callback function
    ├──config.py                      # parameter configuration
    ├──dataset.py                     # dataset
    ├──model.py                       # network structure
  ├──solve.py                         # train and test
```

### Script Parameters

You can set training parameters and sampling parameters in `src/config.py`.

```python
Helmholtz2D_config = ed({
    "name": "Helmholtz2D",                  # PDE name
    "columns_list": ["input", "label"],     # Evaluation Dataset Name
    "epochs": 10,                           # training epochs
    "batch_size": 128,                      # batch size
    "lr": 0.001,                            # learning rate
    "coord_min": [0.0, 0.0],                # lower bound of the domain
    "coord_max": [1.0, 1.0],                # upper bound of the domain
    "axis_size": 101,                       # grid resolution
    "wave_number": 2                        # wavenumber
})

rectangle_sampling_config = ed({
    'domain' : ed({                         # Defining the domain sampling
        'random_sampling' : False,          # random sampling or not
        'size' : [100, 100],                # Grid resolution without random sampling
    }),
    'BC' : ed({                             # Defining the boundary sampling
        'random_sampling' : True,           # random sampling or not
        'size' : 128,                       # batch size
        'with_normal' : False,              # return normal direction or not
    })
})
```

## Model Training

You can use the solve.py script to train and solve the Maxwell's equations in frequency domain. During the training, the model parameters are automatically saved as a checkpoint file.

```shell
python solve.py
```

The loss values are displayed in real time during training:

```log
epoch: 1 step: 79, loss is 630.0
epoch time: 26461.205 ms, per step time: 334.952 ms
epoch: 2 step: 79, loss is 196.4
epoch time: 278.594 ms, per step time: 3.527 ms
epoch: 3 step: 79, loss is 191.4
================================Start Evaluation================================
Total prediction time: 10.388108491897583 s
l2_error:  0.1875916075312643
=================================End Evaluation=================================
epoch time: 10678.531 ms, per step time: 135.171 ms
epoch: 4 step: 79, loss is 3.998
epoch time: 277.924 ms, per step time: 3.518 ms
epoch: 5 step: 79, loss is 3.082
epoch time: 274.681 ms, per step time: 3.477 ms
epoch: 6 step: 79, loss is 2.469
================================Start Evaluation================================
Total prediction time: 0.009278535842895508 s
l2_error:  0.019952444820775538
=================================End Evaluation=================================
epoch time: 292.866 ms, per step time: 3.707 ms
epoch: 7 step: 79, loss is 1.934
epoch time: 275.578 ms, per step time: 3.488 ms
epoch: 8 step: 79, loss is 2.162
epoch time: 274.334 ms, per step time: 3.473 ms
epoch: 9 step: 79, loss is 1.744
================================Start Evaluation================================
Total prediction time: 0.0029311180114746094 s
l2_error:  0.017332553759497542
=================================End Evaluation=================================
epoch time: 277.262 ms, per step time: 3.510 ms
epoch: 10 step: 79, loss is 1.502
epoch time: 272.946 ms, per step time: 3.455 ms
l2 error: 0.0173325538
per step time: 3.4550081325
```

## Random Seed Setting

The seed in the create_dataset function is set in dataset.py, and the random seed in train.py is also used.

## MindScience Home Page

Visit the official website [home page](https://gitee.com/mindspore/mindscience).
