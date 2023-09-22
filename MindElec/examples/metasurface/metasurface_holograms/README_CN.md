# 目录

- [目录](#目录)
- [超表面全息设计](#超表面全息设计)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [模型训练](#模型训练)
- [MindScience主页](#mindscience主页)

# 超表面全息设计

“全息显示”是一种利用干涉原理记录并再现物体真实三维图像的3D成像技术，但是传统的光场调控器件存在视场角狭窄、信息容量小等问题。“超表面（metasurface）”作为一种新型平面光学器件，有望利用其像素尺寸小和光场调控能力强的特点在全息技术领域实现新的突破。研究发现，当物质结构尺度小于光波长时，会出现与宏观条件下完全不同的光学调制作用，使用亚波长结构，可以对光进行相位、振幅或偏振等多个维度的调制。本案例基于东南大学的一项研究，使用物理辅助对抗生成网络（Physics-assisted GAN）的AI方法针对超表面全息成像设计进行非监督学习，避免了数据集的制作过程，并且和传统的GS算法相比在指标和视觉感受上效果更优。

具体做法如下：首先构建一个深度神经网络，此神经网络的输入为目标像，输出为超材料编码；其次使用格林函数推导出从超材料编码到目标像的前向传播过程；最后将神经网络和这一前向传播过程对接，形成一个输入和输出都是目标图像的自编码器结构，自编码器结构如图1所示。将输入图像与输出图像间的均方误差作为目标函数训练该自编码器，使得目标函数尽可能的小，这样训练完成后其中神经网络的输出就是所需的超材料编码。实际训练时发现，由于以成像作为目标，均方误差并不能完全反应所成像在人类视觉感官上的相似度，因此在误差函数中还引入了GAN的判别误差，使得自编码器同时是一个GAN的生成器，所成全息像的视觉语义特征更加明晰。

<div  align="center">
<img src="./docs/autoencoder.jpg" width = "800" alt="auto encoder structural view" align=center />
</div>

参考论文：

Liu, Che, Wen Ming Yu, Qian Ma, Lianlin Li and Tie jun Cui. “Intelligent coding metasurface holograms by physics-assisted unsupervised generative adversarial network.” Photonics Research (2021): n. pag.

# 数据集

训练时的目标图像使用的是：MNIST手写数字数据集

# 环境要求

- 硬件（Ascend/GPU）
- 准备Ascend/GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)　　
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```path
└─metasurface_holograms
  ├──README.md                        # README示意图
  ├──docs
  ├──config.py                        # argparse
  ├──dataset.py                       # 数据集
  ├──main.py                          # main模块
  ├──mse.py                           # MSE函数
  ├──resnet34.py                      # ResNet网络
  ├──train_test.py
  ├──utils.py                         # 功能函数模块
  ├──wgan_gradient_penalty.py         #
```

## 脚本参数

可以使用python的命令行参数--option=value设置相关选项

```python
"model": ['GAN', 'DCGAN', 'WGAN-CP', 'cWGAN-GP'],                      # 模型名称, 默认值: 'cWGAN-GP'
"device": 0,                                                           # 设备编号
"is_train": False,                                                     # 是否训练
"is_finetune": True,                                                   # 是否微调
"is_evaluate": False,                                                  # 是否使用来自test_dataroot的图片做测试
"is_test_single_image": True,                                          # 是否使用来自single_image_path的单图片做测试
"dataroot": './MNIST/binary_images',                                   # dataroot路径
"train_dataroot": '../MNIST/binary_images/trainimages',                # train_dataroot路径
"test_dataroot": '../MNIST/binary_images/testimages',                  # test_dataroot路径
"valid_dataroot": '../MNIST/binary_images/validimages',                # valid_dataroot路径
"single_image_path": './MNIST/binary_images/testimages/9/38.png',      # single_image_path路径
"dataset": 'mnist',                                                    # 数据集名称
"download": False,                                                     # 是否download
"epochs": 200,                                                         # epoch数
"Gen_learning_rate": 3e-5,                                             # 生成器学习率
"Dis_learning_rate": 3e-5,                                             # 判别器学习率
"save_per_times": 500,                                                 # 每更新多少次存一次模型
"batch_size": 64,
"Dis_per_train": 1,                                                    # 每多少个iter训练一次判别器
"Gen_per_train": 3,                                                    # 每多少个iter训练一次生存器

```

## 模型训练

您可以通过 wgan_gradient_penalty.py 脚本训练模型：

```shell
python wgan_gradient_penalty.py --epochs 1000 --is_evaluate True
```

# MindScience主页

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。

