# MindFlow代码合入CheckList

本文档介绍如何向MindFlow合入代码，包括合入前需要准备的文件、数据，合入步骤以及需要注意的事项，帮助贡献者更高效地进行代码合入。

如果缺少调试代码的硬件环境，可以参考[启智社区云脑使用指南](https://download-mindspore.osinfra.cn/mindscience/mindflow/tutorials/%E5%90%AF%E6%99%BA%E6%8C%87%E5%8D%97.pdf), [NPU使用录屏](https://download-mindspore.osinfra.cn/mindscience/mindflow/tutorials/npu%E4%BD%BF%E7%94%A8.MP4), [GPU使用录屏](https://download-mindspore.osinfra.cn/mindscience/mindflow/tutorials/gpu%E4%BD%BF%E7%94%A8.MP4)。

## API代码

API代码主要指合入`MindFlow/mindflow`目录的代码，主要为案例提供高效、易用的公共API接口，因此API代码编写时需要注意以下几点：

1、考虑在多个案例上的可扩展性，避免'Hard Code'，在维度、深度等参量上预留足够的入参，以供用户根据实际情况选择，注意非法入参的检查；

2、入参命名上，MindFlow追求尽量统一，因此新的API合入时，需要与原有API的入参尽量对齐，新的入参命名可与Commiter联系；

3、API的存放位置需根据MindFlow的套件架构决定，注意更新`__init__.py`文件和`cmake/package.cmake`文件；

4、API文档包含两部分，一个是代码注释部分，一个是`mindscience/docs/api_python/mindflow`和`mindscience/docs/api_python_en/mindflow`中的中英文文档；

5、API相关测试用例来进行维护，保证其随时可用，测试用例提交在`mindscience/tests`中，可根据具体用例修改，但运行时间不宜过长，结果检查合理；

## 案例目录格式

案例代码主要指合入`MindFlow/applications`目录的代码，需要根据研究范式，归入`physics_driven`、`data_driven`、`data_mechanism_fusion`、`cfd`几个目录中。

【必须】Jupyter Notebook中英文：为用户提供逐行的代码实现方式，详细讲解案例的实现方式和运行结果。

【必须】`images`：包含了README、notebook等文件里的所有图片。

【必须】`src`：为了保证训练代码的整洁性，可以抽取的函数和类可以统一放在src目录中，`__init__.py`一般为必须，`dataset.py`中包含数据集相关函数和类，`model.py`中包含模型相关函数和类，`utils.py`中包含工具函数和类，外部文件的调用统一从src导入。

【必须】参数文件：案例中具体参数的配置，一般采用yaml文件，为了方便查看，按照优化器、模型等进行分类。

【必须】训练脚本：案例的训练和验证脚本，在训练时除特殊情况，必须有测试集进行验证；训练脚本中的代码应该尽量简洁，复杂的调用封装到后端函数里。

* 注意：类和函数中需要避免'Hard Code'，变量名需要有实际含义；尽量避免使用'Magic Number'，必要的需要在注释里说明；超过50行以上的代码可以考虑抽取出函数调用，减少重复代码；函数的功能尽可能单一，遵从'高内聚，低耦合'原则。

### 单个案例目录格式

单一的案例代码如[`PINNs求解Burgers`](./applications/physics_driven/burgers)为例，代码目录分成以下结构：

```shell
.
├──images
│  ├──background.png
│  └──result.png
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
├──burgers_cfg.yaml
├──eval.py
└──train.py
```

### 多个案例目录格式

有时，有多个案例会使用相同的模型和方法，使用不同的数据集，为了避免代码和文档的重复，`src`目录下统一存放所有案例公共的代码和每个案例自定义的代码，`images`目录统一存放图片文件，`README.md`文件在总体上介绍模型方法和所有的案例，`problem.ipynb`文件介绍具体的案例代码，所有案例具有相同的入口，在命令行里通过指定参数来确定运行的具体案例，文件格式如下：

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

外层训练/测试文件调用的方式如下：

```python
...
parser = argparse.ArgumentParser(description="Cae-Lstm")
parser.add_argument("--case", type=str, default='riemann', choices=['riemann', 'kh', 'sod'],
                        help="Which case to run")
...
args = parser.parse_args()
...
model = Model()
if args.case == 'riemann':
    dataset = create_riemann_dataset()
elif args.case == 'kh':
    dataset = create_kh_dataset()
else:
    dataset = create_sod_dataset()
model.train(dataset)
...
```

## 训练文件格式

训练文件train.py为模型训练的入口，格式如下：

```python
import os
import time
import argparse
import numpy as np

from mindspore import context, nn, Tensor, set_seed, ops, data_sink, jit, save_checkpoint
from mindspore import dtype as mstype

from mindflow import FNO1D, load_yaml_config, get_warmup_cosine_annealing_lr
from mindflow.pde import FlowWithLoss

from src import create_training_dataset, visual, calculate_l2_error
# 相关依赖导入，按照python官方库、第三方库、mindflow、src的顺序导入，导入mindflow时，精确到二级目录

set_seed(123456)
np.random.seed(123456)
# 设置随机数

def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Problem description')
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=3, help="ID of the target device")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    input_args = parser.parse_args()
    return input_args


def train(input_args):
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    # 读取训练配置
    config = load_yaml_config(input_args.config_file_path)

    # 创建训练集和测试集
    train_dataset, test_dataset = create_training_dataset(data_params, shuffle=True)
    # 初始化模型
    model = Model(config)

    problem = FlowWithLoss(model)
    # 前向函数
    def forward_fn(data, label):
        ...

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
    # 训练的前向和反向过程
    @jit
    def train_step(data, label):
        ...
    # 数据下沉
    sink_process = data_sink(train_step, train_dataset, 1)

    # 训练流程
    for epoch in range(1, config["epochs"] + 1):
        model.set_train()
        train()
        # 训练和验证函数，采用MindSpore函数式编程范式编写，注意打印内容尽量统一
        print(f"epoch: {epoch} train loss: {step_train_loss} epoch time:  {time.time() - time_beg:.2f}s")
        # 验证
        if epoch % config['eval_interval'] == 0:
            model.set_train(False)
            print("================================Start Evaluation================================")
            eval()
            print(f"epoch: {epoch} eval loss: {step_train_loss} epoch time:  {time.time() - time_beg:.2f}s")
            print("=================================End Evaluation=================================")
        if epoch % config['save_ckpt_interval'] == 0:
            save_checkpoint(model, 'my_model.ckpt')


if __name__ == '__main__':
    print(f"pid: {os.getpid()}")
    print(datetime.datetime.now())
    # 读取脚本入参
    args = parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    # context设置，由于Ascend和GPU使用的差异，需要使用use_ascend变量进行判断
    start_time = time.time()
    # 调用训练函数
    train(args)
    print("End-to-End total time: {}s".format(time.time() - start_time))
```

## 配置文件格式

参数按照模型、数据、优化器等类别分类，放在"./configs"目录下，配置中的路径参数都是根目录的相对路径。参数命名规范统一格式，格式如下：

```yaml
model:
  in_channels: 3
  out_channels: 3
  height: 192
  width: 384
  encoder_depth: 6
  decoder_depth: 6
  decoder_num_heads: 16

data:
  train_dataset_path: "./dataset/test.npy"
  test_dataset_path: "./dataset/train.npy"
  grid_path: "./dataset/grid.npy"
  batch_size: 32

optimizer:
  epochs: 1000
  lr: 0.0005
  wave_level: 1
```

## README文件格式

其中，总目录中的README对整体背景、技术路线、结果进行讲解，在每个案例中，可以分别在案例的角度描述，注意整体和局部的详略关系，避免重复描述和重复代码。

【必须】README.md和README_CN.md，中英文README文件，一般包含以下部分：

```md
# 标题

## 概述

简单介绍一下案例的背景、方法、数据集、效果等。

## 快速开始

为用户提供快速运行脚本的方法，一般提供脚本调用和Jupyter Notebook两种方式。其中，脚本调用需要展示启动命令的入参含义

### 训练方式一：在命令行中调用`train.py`脚本

python train.py --config_file_path ./configs/burgers.yaml --mode GRAPH --device_target Ascend --device_id 0

其中，
`--config_file_path`表示参数文件的路径，默认值'./burgers_cfg.yaml'；

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

### 训练方式二：运行Jupyter Notebook

您可以使用中英文版本的Jupyter Notebook（附链接）逐行运行训练和验证代码。

## 结果展示

用1-2张图的方式展示模型推理的效果，最好为gif。

## 性能

如果案例涉及到GPU和Ascend双后端，则需要用表格的形式展示训练的主要性能指标进行对比。

|        参数         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     硬件资源         |     Ascend, 显存32G      |      NVIDIA V100, 显存32G       |
|     MindSpore版本   |        >=2.0.0             |      >=2.0.0       |
|     数据集         |      [Burgers数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)             |      [Burgers数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)       |
|      参数量       |       6e4       |         6e4         |
|      训练参数     |    batch_size=8192, steps_per_epoch=1, epochs=15000 | batch_size=8192, steps_per_epoch=1, epochs=15000 |
|     测试参数      |  batch_size=8192, steps=4   | batch_size=8192, steps=4 |
|     优化器         |        Adam     |        Adam         |
|     训练损失(MSE)    |      0.001        |     0.0001       |
|        验证损失(RMSE)     |        0.010       |       0.008       |
|     训练速度(ms/step)   |     10       |    130  |

## 贡献者

gitee id: [id](开发者gitee个人空间的链接)

email: myemail@163.com

```

## Jupyter Notebook文件格式

Jupyter Notebook文件格式可参考[2D_steady_CN.ipynb](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/data_driven/airfoil/2D_steady/2D_steady_CN.ipynb)。

将主要代码模块从训练脚本中抽出，有序分块放入Jupyter Notebook文件。Jupyter Notebook一般包含`概述`、`问题背景`、`技术路径`、`依赖导入`、`数据集制作`、`模型搭建`、`模型训练`、`结果展示`等部分。在每个部分，应当对代码重要内容进行说明，保证按照说明执行代码块能正常运行。

## PR创建和合入

请认真阅读[MindScience贡献指南](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)，提高代码合入效率。在确认代码完成度接近完成后，联系代码仓管理员进行代码review，根据review意见修改，代码仓门禁通过后，需要2位审核人通过审核，完成合入。