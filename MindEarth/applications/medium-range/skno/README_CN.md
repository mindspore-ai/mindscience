## 一、案例特性概述

### 1.1 需求来源及价值概述

SKNO案例由本团队开发完成，融合了KNO模型和SHT算子。SKNO基于16年同化再分析数据集ERA5开发，能够对6小时时间分辨率和1.4度空间分辨率的全球气象进行预测，预测结果包括温度、湿度、风度等指标。

在本案例中，我们同时集成了SKNO算法模型和基于MindSpore开发的SHT算子。用户既可以直接调用SKNO算法进行使用，也可以单独调用SHT算子进行开发。

球面卷积的公式：
$$\mathcal{F}[k\times u](l,m)=2\pi \sqrt{\frac{4\pi }{2l+1} }\mathcal{F}[u]\cdot \mathcal{F}[k](l,0)$$

利用可学习参数$\widetilde{k}_{\theta}(l)$替代固定算子$\mathcal{F}[k](l,0)$，则可以转换成：
$$\mathcal{F}[ \mathcal{K}_{\theta}[u] ](l,m)=\widetilde{k}_{\theta}(l)\cdot \mathcal{F}[u](l,m)$$

表 1-1：涉及的特性列表

| *ISSUE编号* | *ISSUE标题*                                    | *特性等级*         | *支持后端*       | *支持模式*              | 支持平台          | *MindSpore支持版本* |  *规划版本* |
| ------------- | ------------------------------------------------ | -------------------- | ------------------ | ------------------------- | ----------------- | ------------ |  ------------ |
| *mindearth-2023Q4-2*    | *application-mindearth-skno*    | *STABLE* | *Ascend 910A*            | *Graph* | LINUX | *1.10.1*      | *0.2*      |

### 1.2 场景分析

_总体目标：作为MindEarth套件的一部分，提供SKNO模型的训练和推理的场景供客户使用。_

- SKNO可以作为中长期气象预测模型。
- SKNO中的SHT算子可以单独作为模型的组成部分，进行二次开发。
- 提供多卡训练的接口。

表 1-2：约束说明

| *支持后端*       | *支持模式*              | 支持平台          |
| ------------------ | ------------------------- | ----------------- |
| *ASCEND 910A* | *Graph* | LINUX |

## 二、详细设计

### 2.1 总体方案描述

_SKNO包括模型的训练、推理、代码测试、算子测试等代码。_

图 2-1 SKNO整体框架

![image desc](images/SKNO.PNG)

#### 2.1.1 模型架构

- SKNO模型主要由三部分组成，分别是Encoder、SKNOBlock、Decoder。在模型中，输入的数据先通过Encoder进行编码，编码后的特征利用SKNOBlock进行特征增强和学习，增强后的特征利用Decoder进行解码还原。
- Encoder：编码端主要包含两部分，一个是用于划分Patch的Patch_embedd模块，另一个是可选择的MLP。在Patch_embedd中，输入的特征利用卷积步长很大的卷积核进行Patch划分。之后，对划分Patch后的特征进行位置编码。最后可以选择是否利用MLP对特征进行进一步的融合。
- SKNO2d：在该模块中，我们利用SHT算子对气象数据进行球谐波变换，将气象数据进行球面分解。之后，对分解后的数据进行增强和融合。再利用iSHT算子对增强后的数据进行球谐波反变换。还原到变换前的格式。
- Decoder：在解码端，我们利用MLP对输入的数据进行增强和融合，再利用维度变换将特征的维度还原到输入数据的形式。
- 在SKNO模型中，我们利用reconstruction loss约束模型对于输入信息的重构能力。同时，利用重构模块，降低模型的计算量。

#### 2.1.2 模型训练

- 模型采用单步预测进行训练。即，每次训练时，往后推理一步，利用后一天的预测结果，对当天输入的数据进行监督训练。
- 训练的过程利用Lploss进行监督，该损失函数计算预测结果与真实结果的差值。该损失函数利用归一化的差值作为模型的损失函数值。

$$Lploss = \frac{ \sqrt[p]{(\sum (abs(Y_{pred}-Y_{label})))^p} }{ \sqrt[p]{(\sum(Y_{label}))^p} }, (p=2)$$

- 模型的训练过程利用AdamW进行约束。

### 2.2 目录结构

```plaintext
applications
├── medium-range                                # 中期模型
│   └── SKNO                                    # SKNO
│       ├── README.md                           # 模型英文说明文档
│       ├── README_CN.md                        # 模型中文说明文档
│       ├── skno.ipynb                          # 模型说明文档
│       ├── requirements.txt                    # 依赖说明文件
│       ├── scripts                             # 脚本文件
│       │   ├── run_distributed_train.sh        # 分布式训练脚本
│       │   ├── run_eval.sh                     # 验证脚本
│       │   └── run_standalone_train.sh         # 单机训练脚本
│       ├── src                                 # 模型定义源码目录
│       │   ├── solver.py                       # 模型结构定义
│       │   ├── callback.py                     # 回调函数定义
│       │   ├── dataset.py                      # 功能函数定义
│       │   ├── skno_block.py                   # 模型模块定义
│       │   └── skno.py                         # 模型结构化定义
│       ├── configs                             # 案例配置目录
│       │   └── skno.yaml                       # 案例配置yaml文件
│       └── main.py                             # 主文件
```

### 2.3 配置文件结构

_skno.yaml_

```plaintext
model:
  name: SKNO
  backbone: "SKNO"
  encoder_depth: 16
  encoder_network: False
  encoder_embed_dim: 768
  num_blocks: 16
  mlp_ratio: 4
  dropout_rate: 1.
data:
  name: "era5"
  root_dir: './dataset'
  feature_dims: 68
  pressure_level_num: 13
  patch: True
  patch_size: 4
  batch_size: 1
  h_size: 128
  w_size: 256
  t_in: 1
  t_out_train: 20
  t_out_valid: 20
  t_out_test: 20
  valid_interval: 36
  test_interval: 36
  train_interval: 6
  pred_lead_time: 6
  data_frequency: 1
  train_period: [start_year, end_year]
  valid_period: [start_year, end_year]
  test_period: [start_year, end_year]
  recon: True
  ori_shape: False
optimizer:
  name: "adam"
  finetune_lr: 0.0000003
  warmup_epochs: 1
  weight_decay: 0.1
  gamma: 0.5
summary:
  summary_dir: "./summary"
  save_checkpoint_steps: 5
  keep_checkpoint_max: 20
train:
  name: "oop"
```

### 2.4 验收规格

|        Parameter         |        NPU              |
|:----------------------:|:--------------------------:|
|     Hardware         |     Ascend memory 32G      |
|     MindSpore version   |        2.2.0             |
|     dataset      |      [ERA5_1_4_16yr](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)             |
|     parameters      |          91214592         |
|     training config    |        batch_size=1, steps_per_epoch=24834, epochs=200              |
|     test config      |        batch_size=1,steps=39              |
|     optimizer      |        AdamW              |
|        training loss(MSE)      |        0.0857             |
|        Z500(6h,72h,120h)(RMSE)      |        28, 164, 349             |
|        T2M(6h,72h,120h)(RMSE)      |        0.86, 1.36, 1.78             |
|        T850(6h,72h,120h)(RMSE)      |        0.61, 1.35, 2.01             |
|        U10(6h,72h,120h)(RMSE)      |        0.66, 1.87, 2.87             |  
|        speed(ms/step)          |     692ms       |
|        total training time （h/m/s）         |     430332s       |

#### tiny数据

|        Parameter         |        NPU             |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend memory 32G      |      NVIDIA V100 memory 32G       |
|     MindSpore version   |        2.2.0             |      2.2.0       |
|     dataset      |      [ERA5_1_4_tiny400](https://download.mindspore.cn/mindscience/mindearth/dataset/WeatherBench_1.4_69/)             |     [ERA5_1_4_tiny400](https://download.mindspore.cn/mindscience/mindearth/dataset/WeatherBench_1.4_69/)      |
|     parameters      |       91214592         |     91214592    |
|     training config    |        batch_size=1, steps_per_epoch=408, epochs=200             |     batch_size=1, steps_per_epoch=408, epochs=200       |
|     optimizer      |        Adamw              |    Adamw     |
|        training loss(Lp)      |        0.136             |    0.093   |
|        Z500(6h,72h,120h)(RMSE)      |       150,539,772    |       160,605,819      |
|        T2M(6h,72h,120h)(RMSE)      |      1.84,3.19,3.60       |       1.86,3.86.4.63      |
|        T850(6h,72h,120h)(RMSE)      |     1.33,3.02,3.57           |       1.30,3.31,4.29      |
|        U10(6h,72h,120h)(RMSE)      |       1.26,3.46,4.35         |       1.42,3.82,4.71      |
|        speed(ms/step)          |       640       |      340     |

## 三、快速开始

在[SKNO/dataset](https://download.mindspore.cn/mindscience/mindearth/dataset/WeatherBench_1.4_69/)下载数据并保存在`./dataset`。

### 运行方式一: 在命令行调用`main.py`脚本

```shell
python -u ./main.py \
  --config_file_path .configs/skno.yaml \
  --device_target Ascend \
  --device_id 0
```

其中，
`--config_file_path` 配置文件的路径，默认值".configs/skno.yaml"。

`--device_target` 表示设备类型，默认Ascend。

`--device_id` 表示运行设备的编号，默认值0。

### 运行方式二: 运行Jupyter Notebook

使用[中文](https://gitee.com/mindspore/mindscience/blob/master/MindEarth/applications/medium-range/skno/SKNO.ipynb) Jupyter Notebook可以逐行运行训练和推理代码

## 四、参考文献

Bonev, Boris, et al. "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere." arXiv preprint arXiv:2306.03838 (2023).

## Contributor

gitee id: chenhao
email: 2306963526@qq.com
