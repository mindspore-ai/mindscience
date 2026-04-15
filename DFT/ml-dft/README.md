
# 模型名称

> ML-DFT

## 介绍

> ML-DFT是三个深度学习模型的组合，功能：根据输入的POSCAR格式结构信息，在DFT水平上预测分子和聚合物电子结构的各种特性：电子密度、态密度和总势能（包括力和应力张量）。

## 环境要求

> 1. 安装`mindspore（2.3.1）`
> 2. 安装依赖包：`pip install -r requirement.txt`

## 快速入门

> 1. 安装依赖包：`pip install -r requirement.txt`
> 2. 修改配置文件`config/config.yaml`
> 3. 运行命令： `python ML_DFT.py`
> 4. 评估结果放在`config.yaml`中指定的`output_path`路径的文件中

### 代码目录结构

```txt
ML-DFT
    │  README.md    README文件
    │  ML_DFT.py    训练推理脚本
    │  requirement.txt    环境依赖
    │  
    └─ config.yaml  配置文件
            T1_config.yaml   推理案例1配置文件，推理电荷密度
            T2_config.yaml   推理案例2配置文件，推理除电荷密度外性质
            T3_config.yaml   推理案例3配置文件，重训练能量模型
            T4_config.yaml   推理案例4配置文件，重训练态密度模型
    └─ dataset      数据集
            chains
            crystals
            molecules
    └─ src
            dataset.py       用于读取数据并进行预处理
            CHG.py           电子密度模型实现及预测处理
            DOS.py           态密度模型结构和重训练
            Energy.py        能量模型结构和重训练
            utils.py         处理数据和生成分子指纹
    └─ checkpoints  模型检查点和预处理类检查点
            weight_*.ckpt    三个子模型的检查点
            Scale_*.joblib   MaxAbsScaler预处理类检查点
```

## 训练过程

### 推理评估

预训练模型保存在`checkpoints`文件夹中。

更改config文件中的`test_chg`, `test_e`, `test_dos`, `comp_chg`, `write_chg`字段来更改推理参数，前三个字段**决定了预测的性质**，后两个**决定是否比较电荷密度并写入文件**。`infer_config.yaml`示例见`T1_config.yaml`和`T2_config.yaml`

```bash
python ML_DFT.py --config infer_config.yaml
```

推理得到的各个性质将保存在output_path指定的.txt, .dat文件中。

得到的评估结果日志：

```log
Atomic charges for the C atoms (same order as in POSCAR): [4.0054417, 4.008255, 3.9998262, 4.0072227, 4.007003, 4.0009627]
Atomic charges for the H atoms (same order as in POSCAR): [0.98231983, 0.98133796, 0.98198545, 0.9825573, 0.983065, 0.98278385]
Writing atomic charges to text files...
(1, 6, 340) (1, 6, 208) (1, 1, 340) (1, 1, 340)
Total potential energy: -75.71218872070312  eV
Atomic forces (eV/A): [[ 0.55497956  0.46530035  0.28090364]
 [-1.3031019   0.64198923 -0.4945764 ]
 [ 2.0284073  -0.48662314  1.0479561 ]
 [-1.8955303   1.9894108  -0.6157311 ]
 [-3.105498   -0.74499136 -1.5150017 ]
 [ 1.2782366  -0.09168521  0.53638756]
 [-0.26253316 -0.5608424  -0.28614005]
 [ 0.76136017 -0.34346697  0.28699273]
 [-0.23988284  0.15507291 -0.11566731]
 [-0.19850907  0.23939659 -0.06908016]
 [ 1.3046951  -1.0190146   0.17075655]
 [ 0.1287648  -0.22165236 -0.0345892 ]]
The stress tensor components are (kB): Sxx: 4.447639  Syy: -4.3828983  Szz: 1.8630034  Sxy: -0.30226827  Syz: -0.2496823  Sxz: 2.207878
Valence band maximum: -6.237622261047363 +- 0.015520952  eV
Conduction band minimum: -1.335547685623169 +- 0.01623134  eV
Bandgap: 4.902074575424194 +- 0.024345977  eV
Writing DOS curve to text file...
Made the dos plot ..
Total pred_charge: 29.922751461867122
Predicted charge error (%): 1.2707557798268063
```

### 训练

更改config文件，设置训练参数: (`train_config.yaml`示例见`T3_config.yaml`和`T4_config.yaml`)
> 1. 更改config文件中的`train_e`, `train_dos`字段，分别表示重训练能量模型和态密度模型
> 2. 设置epoch和batch size
> 3. 训练后工作目录下得到模型检查点`newEmodel.ckpt`或`newDOSmodel.ckpt`

```bash
pip install -r requirement.txt
python ML_DFT.py --config train_config.yaml
```
