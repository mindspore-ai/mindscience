# MindSpore Flow Release Notes

[View English](./RELEASE.md)

MindSpore Flow是基于昇思MindSpore开发的流体仿真领域套件，支持航空航天、船舶制造以及能源电力等行业领域的AI流场仿真、AI气动外形设计、AI流动控制，旨在于为广大的工业界科研工程人员、高校老师及学生提供高效易用的AI计算流体仿真软件。

## MindSpore Flow 0.3.0 Release Notes

### 主要特性和增强

#### 数据驱动

* [STABLE] [Burgers_SNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/sno1d)/[Navier_Stokes_SNO2D](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/sno2d)/[Navier_Stokes_SNO3D](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/sno3d): 新增数据驱动下的谱神经算子求解一维Burgers方程、二维/三维纳维·斯托克斯方程案例。
* [STABLE] [API-SNO1D/2D/3D](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/neural_operators/sno.py): 新增谱神经算子（包括SNO和U-SNO）API，利用多项式变换将计算转换到频谱空间的类FNO架构，其优势在于有效减小由混淆误差引起的系统偏差。
* [STABLE] [API-Attention](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/attention.py): 重构transformer类网络常用的Attention、MultiHeadAttention、AttentionBlock、ViT网络接口。
* [STABLE] [API-Diffusion](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/diffusion.py): 新增扩散模型全套训练、推理接口，支持DDPM、DDIM业界两种主流扩散方式，通过简洁易用的DiffusionScheduler、DiffusionTrainer、DiffusionPipeline、DiffusionTransformer接口即可完成扩散模型训练推理全流程。
* [STABLE] [API-Core_Refactor](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/mindflow/core): 将原mindflow.common，mindflow.loss，mindflow.operators优化重构为mindflow.core。
* [RESEARCH] [CascadeNet](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cascade_net): 新增CascadeNet案例，以表面压强、雷诺数和少量尾迹速度测点作为输入，通过具有尺度传递拓扑结构的生成对抗网络，预测圆柱尾迹脉动速度时空场。
* [RESEARCH] [MultiScaleGNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/multiscale_gnn): 新增多级图神经网络案例，求解大规模压力泊松方程，可以支撑使用投影法（或分步法）求解不可压缩纳维 · 斯托克斯方程。
* [RESEARCH] [TurbineUQ](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/turbine_uq): 新增涡轮级流场预测与不确定性优化设计案例，通过蒙特卡洛结合深度学习的方法来完成不确定性的量化评估。

#### 数据-机理融合驱动

* [STABLE] [PhyMPGN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/phympgn): 新增基于图神经网络的物理方程求解模型 PhyMPGN 针对圆柱绕流问题的应用案例。PhyMPGN 能够在非结构网格下实现 Burgers、FitzHugh-Nagumo、Gray-Scott 等方程的求解，相关[论文](http://arxiv.org/abs/2410.01337)被接收为 ICLR 2025 Spotlight。
* [RESEARCH] [Heat_Conduction](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/heat_conduction): 新增数据与物理混合驱动下的稳态热传导物理场预测案例。
* [RESEARCH] [SuperPosition](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/superposition): 新增叠加原理的算子神经网络SDNO，用于航空发动机内流叶栅复杂流态下温度场的预测。

#### 物理驱动

* [RESEARCH] [NSFNets](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/nsf_nets): 新增纳维 · 斯托克斯流动网络（NSFNets），该网络对应[论文](https://www.sciencedirect.com/science/article/pii/S0021999120307257)为该领域高被引论文，用以求解不适定问题（例如部分边界条件缺失）或反演问题。

#### 求解器

* [STABLE] [CBS求解器](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/acoustic): 新增声波方程求解器 CBS 在复杂参数场下求解二维声波方程的应用案例。CBS 求解器在频域求解声波方程，在空间各方向均具有谱精度，精度高于有限差分法。相关论文参考 [Osnabrugge et al. 2016](https://linkinghub.elsevier.com/retrieve/pii/S0021999116302595)

#### 优化器

* [STABLE] [API-AdaHessian二阶优化器](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/core/optimizers.py): 新增AdaHessian二阶优化器，基于Hessian矩阵对角元提供的二阶信息进行优化计算，实测相同step数下相比Adam实现20%以上的loss下降。

#### 基础模型

* [RESEARCH] [PDEformer](https://github.com/functoreality/pdeformer-2): PDEformer支持[一维](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/pdeformer1d)/[二维](https://github.com/functoreality/pdeformer-2)含时偏微分方程通用求解，典型场景下Zero-Shot求解精度超过专有模型。

### 贡献者

感谢以下开发者做出的贡献：

hsliu_ustc, gracezou, mengqinghe0909, Yi_zhang95, b_rookie, WhFanatic, xingzhongfan, juliagurieva, GQEm, chenruilin2024, ZYF00000, chenchao2024, wangqineng2024, BingyangWu-pkusms21, Bochengz, functoreality, huangxiang360729, ChenLeheng, juste_une_photo.

欢迎以任何形式对项目提供贡献！

## MindSpore Flow 0.2.0 Release Notes

### 主要特性和增强

#### 数据驱动

* [STABLE] [Airfoil2D_Unsteady](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_unsteady): 支持数据驱动(FNO2D和Unet2D两种backbone)下跨声速翼型复杂流场的多时间步预测。
* [STABLE] [API-FNO1D/2D/3D](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/neural_operators/fno.py): 重构FNO1D、FNO2D、FNO3D API，提升接口的通用性，支持"channels_last"和"channels_first"两种输入数据格式，支持mlp层和FNOBlock层分别设置激活函数，支持SpectralConvDft和FNO skip分别设置计算精度，支持设置projection和lifting中间层参数，支持选择残差增强和嵌入位置信息。
* [STABLE] [API-UNet2D](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/unet2d.py): 重构UNet2D API，新增base_channels作为基准通道数，以控制上/下采样的通道数增/减，支持"NCHW"和"NHWC"两种输入数据格式。

#### 数据-机理融合驱动

* [STABLE] [API-Percnn](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/mindflow/cell/neural_operators/percnn.py): 新增percnn API，通过递归卷积神经网络，在粗网格上学习物理场时空演化规律，默认支持两个物理分量的输入可自定义调节conv layer数量及kernel size，实现在不同物理现象上的应用。
* [STABLE] [PeRCNN-gsrd3d](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn/gsrd_3d): 新增PeRCNN求解三维GS反应扩散方程的案例。

#### 物理驱动

* [STABLE] [Boltzmann](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann): 支持D1V3的BGK以及二次碰撞项的玻尔兹曼方程求解，相关论文发表在《[SIAM Journal on Scientific Computing](https://www.siam.org/publications/journals/siam-journal-on-scientific-computing-sisc)》。
* [STABLE] [Periodic Hill](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/periodic_hill): 支持PINNs方法求解周期山流通问题。
* [STABLE] [Possion](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/continuous): 添加PINNs求解poisson方程时对periodic以及robin边界条件的支持。
* [RESEARCH] [Cma_Es_Mgda](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cma_es_mgda): 支持CMA-ES和多目标梯度优化算法（mgda）结合求解PINNs问题。
* [RESEARCH] [Moe_Pinns](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/moe_pinns): 支持多专家模型求解PINNs问题。
* [RESEARCH] [Allen-Cahn](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/allen_cahn): 反应扩散的Allen-Cahn和NS特解的Kovasznay流是常见的物理过程，通过PINNs方式以无监督方式完成对特定I/BC的求解。

### 贡献者

感谢以下开发者做出的贡献：

hsliu_ustc, Yi_zhang95, zwdeng, liulei277, chengzrz, mengqinghe0909, xingzhongfan, jiangchenglin3, positive-one, yezhenghao2023, lunhao2023, lin109, xiaoruoye, b_rookie, Marc-Antoine-6258, yf-Li21, lixin07, ddd000g, huxin2023, leiyixiang1, dyonghan, huangxiang360729, liangjiaming2023, yanglin2023

欢迎以任何形式对项目提供贡献！

## MindSpore Flow 0.1.0 Release Notes

### 主要特性和增强

#### 数据驱动

* [STABLE] [CAE-LSTM](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm): 支持数据驱动的卷积自编码器–长短时记忆神经网络求解非定常可压缩流动。
* [STABLE] [Move Boundary Hdnn](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/move_boundary_hdnn): 支持数据驱动的HDNN网络求解动边界的非定常流场。

#### 数据-机理融合驱动

* [STABLE] [PeRCNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn): 支持物理编码递归卷积神经网络(Physics-encoded Recurrent Convolutional Neural Network，PeRCNN)。

#### 物理驱动

* [STABLE] [Boltzmann](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann): 支持PINNs方法求解玻尔兹曼方程。
* [STABLE] [Poisson with Point Source](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source): 支持PINNs方法求解带点源的泊松方程。

## MindSpore Flow 0.1.0.rc1 Release Notes

### 主要特性和增强

#### 数据驱动

* [STABLE] [KNO](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/kno2d): 支持KNO神经算子，提升NS方程仿真精度
* [STABLE] [东方.御风](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady): 东方御风大模型，支持翼型端到端快速仿真。

## MindSpore Flow 0.1.0-alpha Release Notes

### 主要特性和增强

#### 数据驱动

* [STABLE] 提供了多种神经网络，包括全连接网络、残差网络、傅里叶神经算子、Vision Transformer，支持多种数据格式的读取和数据集的合并，MindFlow提供了模型训练和推理的高阶API，支持多种学习率和损失函数的使用。

#### 数据-物理融合驱动

* [STABLE] [PDE-Net](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/pde_net): 提供数据-物理融合驱动的深度学习方法PDE-Net，用于流场的时序预测和偏微分方程的回归。

#### 物理驱动

* [STABLE] 支持物理信息神经网络求解偏微分方程，使用sympy定义微分方程及基本方程的求解，支持计算神经网络输出对输入的一阶和二阶导数矩阵，提供了基本几何形状、时域的定义及其操作，便于几何区域内和边界上的采样。

#### 可微分CFD求解器

* [STABLE] 我们推出了端到端可微分的可压缩CFD求解器MindFlow-CFD，支持WENO5重构、Rusanov通量以及龙格-库塔积分，支持对称、周期性、固壁及诺依曼边界条件。

### 贡献者

感谢以下开发者做出的贡献：

hsliu_ustc, Yi_zhang95, zwdeng, liulei277, chengzrz, liangjiaming2023, yanglin2023

欢迎以任何形式对项目提供贡献！
