
## 特性概述

### 需求来源及价值
- 介绍当前特性功能的需求来源和价值。

### 背景信息
- 介绍特性功能的背景信息。

### 软硬件版本

| 后端类型| 硬件具体类别 |
| --- | --- |
| 后端类型| Atlas 800T A2|
| CANN版本|8.1.RC1.beta1|
| Mindspore版本| 2.6.0 |
| Mindscience版本| 0.1.0 |
| 模式 | 静态图&动态图模式 |

### 场景分析
实现了XXX场景下XXX功能，支持XXX任务。

### 特性影响分析
在原有xxx接口上增加xxx功能，增强套件基础能力完备性和易用性
- 不影响原有xxx功能。
- 支持xxx场景。

## Design（设计方案）

### 详细设计
详细介绍案例/API接口的设计方案。

### 目录结构（案例必选）

```shell
.
├──images
│  ├──background.png
│  ├──result1.png
│  ├──result2.png
│  └──result3.png
├──src
│  ├──__init__.py
│  ├──dataset.py
│  ├──model.py
│  └──utils.py
├──configs
│  ├──fno1d.yaml
├──README.md
├──README_CN.md
├──problem.ipynb
├──problem_CN.ipynb
├──problem_cfg.yaml
├──eval.py
└──train.py
```
### 可靠性/可用性

#### 异常情况：
- 当输入不是`Tensor`类型时，报错。
- 当维度`in_dim`不可以被`num_heads`整除时，报错。

### 对外接口（可选）

#### 接口说明

|序号|基本项|内容|
|----|----|----|
| 1 |接口定义| Transformer |
| 2 |接口描述| Transformer网络接口|
| 3 |输入输出参数| in_dim, hidden_dim, num_heads |
| 4 |属性| query, key, value, proj |
| 5 |方法| construct |
| 6 |返回值| Tensor |


## 测试

### 结果指标（案例必选）
介绍案例训练/推理的关键性能/精度指标。
| 参数 |   指标    |
|:----------------------:|:--------------------------:|
| 硬件资源             | Atlas 800T A2        |
| MindSpore版本        | >=2.5.0            |
| 数据集               | ...      |
| 参数量               | 6e7            |
| 训练参数             | batch_size=32, steps_per_epoch=70, epochs=1000 |
| 测试参数             | batch_size=32        |
| 优化器               | Adam                 |
| 训练损失(MSE)        | 0.07                |
| 验证损失(RMSE)       | 0.0002                |
| 速度(ms/step)        | 150                   |

### 测试用例设计（API接口必选）

介绍相关API接口的配套测试用例。

#### UT用例
|用例编号|用例类型|用例名称|测试对象|测试功能|测试条件|预期结果|
|----|----|----|----|----|----|----|
|1|UT|test_forward|网络前向功能|网络|按照`config.yaml`参数初始化网络|前向结果的shape和dtype符合预期与预期结果相同|


#### ST用例
|用例编号|用例类型|用例名称|测试对象|测试功能|测试条件|预期结果|
|----|----|----|----|----|----|----|
|1|ST|test_train|网络端到端训练|网络|按照`config.yaml`参数初始化网络和数据集|训练100轮，loss下降到0.01|
