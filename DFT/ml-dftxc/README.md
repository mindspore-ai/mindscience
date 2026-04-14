# 模型名称

> ML-DFTXC potential model

## 介绍

> ML-DFTXC potential model基于三维卷积神经网络，通过映射准局部电子密度到局部交换-相关（XC）势来确定DFT的精确XC势。

## 数据集

> 从 http://aisccc.cn/database/data-details?id=171&type=resource 下载 dataset.zip 到并解压，将`trainset_np.pkl`和`testset_np.pkl`放在`data`目录下。

## 环境要求

> 1. 安装`mindspore`
> 2. 安装`numpy`，`tqdm`，`scipy`，`pyscf`，`ConfigParser/configparser`

## 快速入门

> 1. 将数据集下载到当前目录
> 2. 训练命令： `python train.py config/train.cfg`
> 3. 推理命令： `python xcnn.py config/test.cfg`

### 代码目录结构

```txt
EEDM
    │  README.md        README文件
    │  train.py         训练启动脚本
    │  xcnn.py          调用模型预测XC势并进行DFT计算
    │  
    └─data   数据
            *.npy       训练数据
            *.str       推理数据
    │  
    └─src
            __init__.py
            dataset.py  数据集构建
            model.py    模型结构
            loss.py     损失函数
            Config.py   默认配置
            utils.py    辅助函数
            xcnn        DFT计算相关代码
    └─configs
            train.cfg   训练配置文件
            test.cfg    推理配置文件
```

## 训练推理过程

### 训练

```bash
python train.py config/train.cfg
```

### 推理

在 `test.cfg` 中修改模型权重路径 `ModelPath`

```bash
python xcnn.py config/test.cfg
```

### 训练推理过程日志

```log
# 推理
================
SCF calculation
================
...
Evaluate xc potential on grids. Block size: 200 Total: 19008 Number of blocks: 95 Residual: 8
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19008/19008 [00:29<00:00, 650.42it/s]
Construct vxc based on ZF constrain.
Iter   2	max abs diff in dm: 2.78852994e-07	 sum abs diff in dm: 7.75270336e-06	# elec: 2.00000000e+00
SCF converged.
converged SCF energy = -1.17926185615524
converged SCF energy = -1.13263362433774
<class 'pyscf.cc.ccsd.CCSD'> does not have attributes  converged
E(CCSD) = -1.172592143483996  E_corr = -0.03995851914626056

---- Force ----
Evaluate xc potential on grids. Block size: 200 Total: 19008 Number of blocks: 95 Residual: 8
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19008/19008 [00:29<00:00, 652.39it/s]
Construct vxc based on ZF constrain.
Iter  99	max abs diff in dm: 1.35168857e-08	 sum abs diff in dm: 8.75681552e-07	# elec: 2.00000000e+00
```

```log
# 训练
main: CNN_GGA_1_zsym
main: Max iteration: 1000
main: Learning rate: 5.000000e-03
main: Loss function
main: MSELoss_zsym<>
main: Optimiser
main: SGD<>
main: Model saved to ./model_chk/H2_0.9_0_CNN_GGA_1
main: Train start
-
train: Epoch     0: average loss on train: 8.98117044e+00
train: elapse time: 18.893718
validate: Epoch     0: average loss on validate: 5.34661189e+00
validate: elapse time: 1.918868
.
.
.
train: Epoch   999: average loss on train: 3.26439670e-02
train: elapse time: 4.594539
validate: Epoch   999: average loss on validate: 2.57725163e-02
validate: elapse time: 0.116159
main: Model saved.
========Task Finish========
```