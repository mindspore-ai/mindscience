# Prediction of spatiotemporal field of pulsation velocity in cylindrical wake by Cascade Net

## Background 1

In the process of turbulent spatiotemporal evolution, the pulsating velocity field includes a series of important fluid physical processes, such as separation, transition, and energy transfer. At high Reynolds numbers, the pulsating velocity field exhibits significant nonlinear characteristics. There are vortex structures in turbulent wake that range from maximum to minimum scales, and these fluid motion patterns constitute complex flow field structural characteristics. The process of energy transfer from large-scale structures to small-scale structures in these flow field structures is called the energy cascade physics principle. Inspired by this principle, the small-scale prediction problem can be transformed into a step-by-step prediction problem from large-scale to small-scale.

## Model framework

The model framework is as shown in the following figure:

![Cascade-Net](images/Cascade-Net.png)

Where, the generator is a U-Net structure with spatial and channel attention gates, and its framework is shown in the following figure:

![The U-Net structure of the generator with spatial and channel attention gates](images/The_U-Net_structure_of_the_generator_with_spatial_and_channel_attention_gates.png)

And for the spatial attention gate *S* and channel attention gate *C*, the integration diagrams are shown as follows:

![Spatial attention gate S](images/Spatial_attention_gate_S.png)

![Channel attention gate C](images/Channel_attention_gate_C.png)

## QuickStart

Dataset download link: [Cascade_Net/dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/research/Cascade_Net/), Save the dataset to `./dataset`.

The case provides two training methods

- Run Option 1: Call `train.py` from command line

  ```python
  # Call `train.py` from command line
  python train.py --device_target GPU --device_id 0 --config_file_path ./config/Cascade-Net.yaml

  ```

  `--config_path` indicates the path of the parameter file. Default path "./config/config.yaml".

  In the "./config/config.yaml" parameter file:

  'lambda_GP' represents the gradient penalty coefficient, with a default value of 10;

  'critic_model_lr' represents the learning rate of discriminator, with a default value of 0.00025;

  'gan_model_lr' represents the learning rate of generator, with a default value of 0.004；

- Run Option 2: Run Jupyter Notebook

  You can use [Chinese](./Cascade-Net_CN.ipynb) or [English](./Cascade-Net.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Performance

|        Parameters         |           GPU           |        NPU         |
|:-------------------------:|:-----------------------:|:------------------:|
|         hardware          | NVIDIA 3090(memory 24G) | Ascend(memory 32G) |
|     MindSpore version     |         2.2.14          |       2.2.14       |
|         data_size         |          10792          |       10792        |
|        batch_size         |           128           |        128         |
|          epochs           |           300           |        300         |
|         optimizer         |         RMSProp         |      RMSProp       |
|  scale I train loss(MAE)  |         3.4e-2          |       4.6e-2       |
|  scale I test loss(MAE)   |         5.4e-2          |       5.2e-2       |
| scale II train loss(MAE)  |         3.2e-2          |       3.6e-2       |
|  scale II test loss(MAE)  |         5.2e-2          |       5.0e-2       |
| scale III train loss(MAE) |         3.3e-2          |       3.7e-2       |
| scale III test loss(MAE)  |         5.4e-2          |       5.2e-2       |
| scale IV train loss(MAE)  |         3.6e-2          |       4.1e-2       |
|  scale IV test loss(MAE)  |         5.5e-2          |       5.4e-2       |
|       speed(s/step)       |          2.60           |        3.94        |

## Contributor

gitee id: [GQEm](https://gitee.com/guoqicheng2024)
email: qicheng@isrc.iscas.ac.cn

## Reference

Mi J, Jin X, Li H. Cascade-Net for predicting cylinder wake at Reynolds numbers ranging from subcritical to supercritical regime[J]. Physics of Fluids, 2023, 35: 075132.  https://doi.org/10.1063/5.0155649
