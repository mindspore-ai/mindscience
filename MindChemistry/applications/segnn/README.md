
# 模型名称

> Steerable-E3-GNN（SEGNN）

## 介绍

> SEGNN是基于E(3)等变图神经网络构建的SOTA模型，它拓展了等变图神经网络，使得图神经网络中节点和边的属性可以包含协变量，并且对旋转、反射、平移以及置换具有鲁棒性。最终，作者在计算物理和化学领域多个任务中验证了SEGNN模型的有效性。

[Geometric and Physical Quantities improve E(3) Equivariant Message Passing](https://arxiv.org/pdf/2110.02905v2.pdf)

```txt
@article{brandstetter2021geometric,
      title={Geometric and Physical Quantities improve E(3) Equivariant Message Passing},
      author={Johannes Brandstetter and Rob Hesselink and Elise van der Pol and Erik Bekkers and Max Welling},
      year={2021},
      eprint={2110.02905},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 数据集

> QM9数据集下载地址：[数据文件1](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip)；[数据文件2](https://ndownloader.figshare.com/files/3195404)

## 环境要求

> 1. MindSpore：2.2.12
> 2. MindChemistry：暂不支持安装方式；可将`mindscience/MindChemistry`目录下的`mindchemistry`文件夹拷贝至工程目录，以便程序调用相关模块
> 3. 其它依赖

```txt
pip install -r requirements.txt
```

## 快速入门

> 脚本启动

```txt
# 脚本启动样式
python train.py [MODE] [DEVICE_TARGET] [DEVICE_ID] [CONFIG_FILE] [DTYPE]

# 参数说明
MODE：计算图模式，GRAPH--静态图，PYNATIVE--动态图
DEVICE_TARGET：运行的目标设备类型，Ascend
DEVICE_ID：目标设备的ID
CONFIG_FILE：配置文件
DTYPE：模型前向计算的数据类型

# 脚本启动的参考指令
python train.py
--mode GRAPH
--device_target Ascend
--device_id 0
--config_file_path ./qm9.yaml
--dtype float32
```

> qm9.yaml配置文件关键参数说明

```txt
run_mode: 程序运行模式 【train，infer】
profiling: 是否进行模型的profiling分析 【False，True】
profiling_step: 进行profiling分析的step
log_file: 训练的日志文件

model:
  ckpt_file: 训练保存或推理加载的模型权重文件
  num_layers: segnn模型中SEGNNLayer层数
  ncon_dtype: Ascend设备上进行ncon计算的数据类型 【float16，float32】

optimizer:
  num_epoch: 模型训练的epoch数

data:
  dataset_dir: 数据文件的目录
  pre_process_file: 预处理的数据文件
  batch_size: 训练的batch size
```

## 代码目录结构

```txt
└─segnn
    │  README.md            README文件
    │  requirements.txt     依赖文件
    │  qm9.yaml             配置文件
    │  train.py             训练启动脚本
    │  
    └─src
            balanced_irreps.py      Irreps生成
            dataset.py              数据集处理
            inspector.py            函数参数提取模块
            instance_norm.py        特征归一化模块
            o3_building_blocks.py   张量积计算及后处理模块
            segnn.py                SEGNN模型
            trainer.py              训练脚本
```

## 训练与推理

### 训练

step1. 修改qm9.yaml配置文件，将run_mode改为train

step2. 执行脚本，启动训练

```txt
python train.py
```

训练过程日志

```log
Loading data...
train_set's mean and mad:  75.26605 6.2882
Initializing model...
Determined irrep type: 36x0e+36x1o+36x2e
Initializing train...
epoch:   0, step:   0, loss: 0.95143652, train MAE: 5.9828  , time: 206.24
epoch:   0, step: 100, loss: 0.58375533, train MAE: 3.6708  , time: 49.01
epoch:   0, step: 200, loss: 0.45873420, train MAE: 2.8846  , time: 48.29
epoch:   0, step: 300, loss: 0.43487877, train MAE: 2.7346  , time: 48.11
epoch:   0, step: 400, loss: 0.39558753, train MAE: 2.4875  , time: 48.97
epoch:   0, step: 500, loss: 0.39054746, train MAE: 2.4558  , time: 49.82
epoch:   0, step: 600, loss: 0.35352476, train MAE: 2.2230  , time: 48.81
epoch:   0, step: 700, loss: 0.36744346, train MAE: 2.3106  , time: 48.94
epoch:   0, train loss: 0.42016299  , time used: 588.49
eval MAE:2.2511  , time used: 204.01
```

### 推理

step1. 修改qm9.yaml配置文件，将run_mode改为infer

step2. 将配置文件中model配置项的ckpt_file子项更改为要加载的权重文件

step3. 执行脚本，启动推理

```txt
python train.py
```

推理过程日志

```log
Loading data...
train_set's mean and mad:  75.26605 6.2882
Initializing model...
Determined irrep type: 36x0e+36x1o+36x2e
test MAE: 0.1367  , time used: 190.31
```