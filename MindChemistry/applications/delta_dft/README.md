# 单分子DFT计算组件

## 1. 介绍

[Mihail Bogojeski et,al (2020)](https://www.nature.com/articles/s41467-020-19093-1)
提出了一种基于机器学习的单分子密度泛函近似耦合簇能量的方法。
本实验基于该论文方法进行改进，对分子能量进行归一化，将势能投影到对数-极坐标系做傅里叶变换，
以消除旋转对势能场的影响，结合神经网络以计算密度泛函结果与耦合的簇能量结果的差值后进行对齐。

## 2. 运行方法

执行train.py完成单分子在[MD17数据集](http://quantum-machine.org/datasets/) 上的求解与评估。

### 2.1 环境要求

- python>=3.7
- numpy>=1.19
- scikit-learn>=0.23
- scipy>=1.5
- mindspore>=2.2.12
- pickle

### 2.2 config配置选项

在配置文件config.yml中完成：

- 设置charge，各分子对应charge参数，即分子式 (C-6为碳原子，O-8为氧原子，H-1为氢原子)

```python
charges = [1, 1, 8]  # H2O 水
charges = [6, 6, 6, 6, 6, 6, 8, 1, 8, 1, 1, 1, 1, 1]  # C6(OH)2H4 间笨二酚
charges = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1]  # C6H6 苯
charges = [6, 6, 8, 1, 1, 1, 1, 1, 1]  # C2(OH)H5 乙醇
charges = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 8, 1]  # C6H5(OH) 苯酚
```

- 设置*data_dir, train_dir, test_dir*等数据文件夹存放地址
- 设置材料归一化的均值和方差，经实验分子密度采用0,1的均值方差效果更好

```python
# 乙醇
mean_std_density: [0.0, 1.0]
mean_std_energy: [-70290.9146, 1.0859]
# 苯环
mean_std_density: [0.0, 1.0]
mean_std_energy: [-108401.3426, 0.8652]
# 苯酚
mean_std_density: [0.0, 1.0]
mean_std_energy: [-142429.3332, 1.1881]
# 间笨二酚
mean_std_density: [0.0, 1.0]
mean_std_energy: [-176457.5149, 1.3288]
```

- 设置评估指标类型，以下为可选类型

```python
metric = ['L-1', 1]  # 默认为L-1距离
metric = ['L-inf', np.inf]  # 此处为'L-inf'，代表不同阶norm计算得到的距离，L2即为欧氏距离，cosine是余弦距离
metric = ['L-1', 1]
metric = ['L-2', 2]
metric = ['cosine', '']
```

- 其他需要设置的参数（解释在注释中）：

```python
n_trainings = [50, 100, 200, 400, 600, 800, 1000]  # 训练样本数，列表形式
gaussian_width = 0.36  # 高斯核宽度，可以遍历搜索，这里最优值为： 0.60-水 0.36-乙醇 0.42-苯 0.42-间二苯酚 0.42-苯酚
spacing = 0.19  # 人工电势网格宽度，可以遍历搜索，这里最优值为： 0.33-水 0.19-乙醇 0.20-苯 0.20-间二苯酚 0.20-苯酚
energy_types = ['cc', 'diff']  # 使用何种能量预测电子密度，此处要对比'cc'和'diff'两种
```

- 以乙醇为例设置config参数：

```yaml
ensem_params:
  charges: [ 6, 6, 8, 1, 1, 1, 1, 1, 1 ]
  data_dir: '/home/xxx/Dataset/'
  workspace_dir: '/home/xxx/workspace/ethanol/'
  train_dir: 'ethanol_train/'
  test_dir: 'ethanol_test/'
  n_trainings: [ 1000 ]
  n_test: 500
  gaussian_width: 0.36
  spacing: 0.19
  energy_types: [ 'diff' ]
  metric: [ 'L-1', 1 ]
  verbose: 0
  density_kernel_type: 'rbf'
  energy_kernel_type: 'rbf'
  invariance: False

train_params:
  opt_type: 'adam'

norm_params:
  mean_std_density: [ 0.0, 1.0 ]
  mean_std_energy: [ -70290.9146, 1.0859 ]
```

## 3. 脚本说明

```txt
└─dft
    │  README.md    README文件
    │  config.yaml  配置文件
    │  train.py 训练启动脚本
    └─src
            dataset.py    数据处理模块
            model.py      网络模块
            module.py     势能网络模块处理
            utils.py      工具处理模块
            trainer.py    阶段训练脚本
```

## 4. config及parser参数说明

- config初始化参数：
    - charges 各分子对应charge参数，即分子式
    - data_dir 数据目录
    - workspace_dir 文件缓存目录
    - train_dir 训练集目录
    - test_dir 测试集目录
    - n_trainings 训练集样本数
    - n_test 测试样本数
    - gaussian_width 高斯核宽度
    - spacing 人工电势网格宽度
    - energy_types 能量预测电子密度类型
    - metric 评估指标类型
    - verbose 输出信息，verbose等于0时不输出，大于等于1时输出具体信息
    - density_kernel_type 密度核类型，'rbf'为RBF核,'lap'为Laplacian核,'mat'为Matérn核
    - energy_kernel_type 能力核类型
    - opt_type 模型优化器，默认使用adamw优化器
    - mean_std_density 密度的均值和方差
    - mean_std_energy 能量的均值和方差
    - invariance 是否开启傅里叶变换

- parser参数：
    - -max_iter 模型训练步数，类型为int，默认30
    - -mode mindspore的GRAPH_MODE模式和PYNATIVE_MODE模式，默认为ms.GRAPH_MODE
    - -device_target 待运行的目标设备，支持 ‘Ascend’、 ‘GPU’
    - -device_id 目标设备的ID
    - -config_path 配置文件的路径，默认在文件目录下

## 5. 训练结果

- 乙醇1000样本的训练结果：
    - mae：0.9084
    - coefficients errors：182623.3535
- 苯酚1000样本的训练结果：
    - mae：0.9004
    - coefficients errors：172782.0914
- 苯环1000样本的训练结果：
    - mae：0.6860
    - coefficients errors：104033.5208