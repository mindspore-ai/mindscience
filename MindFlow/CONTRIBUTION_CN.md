# MindFlow代码合入CheckList

本文档介绍如何向MindFlow合入代码，包括合入前需要准备的文件、数据，合入步骤以及需要注意的事项，帮助贡献者更高效地进行代码合入。

## API代码

API代码主要指合入`MindFlow/mindflow`目录的代码，主要为案例提供高效、易用的公共API接口，因此API代码编写时需要注意以下几点：

1、考虑在多个案例上的可扩展性，避免'Hard Code'，在维度、深度等参量上预留足够的入参，以供用户根据实际情况选择，注意非法入参的检查；

2、入参命名上，MindFlow追求尽量统一，因此新的API合入时，需要与原有API的入参尽量对齐，新的入参命名可与Commiter联系；

3、API的存放位置需根据MindFlow的套件架构决定，注意更新`__init__.py`文件和`cmake/package.cmake`文件；

3、API文档包含两部分，一个是代码注释部分，一个是`mindscience/docs/api_python/mindflow`和`mindscience/docs/api_python_en/mindflow`中的中英文文档；

4、API相关测试用例来进行维护，保证其随时可用，测试用例提交在`mindscience/tests`中，可根据具体用例修改，但运行时间不宜过长，结果检查合理；

## 案例目录格式

案例代码主要指合入`MindFlow/applications`目录的代码，需要根据研究范式，归入`physics_driven`、`data_driven`、`data_mechanism_fusion`、`cfd`几个目录中。

【必须】Jupyter Notebook中英文：为用户提供逐行的代码实现方式，详细讲解案例的实现方式和运行结果。

【必须】参数文件：案例中具体参数的值放在这里，为了方便查看，按照优化器、模型等进行分类。

【必须】训练脚本：案例的训练和验证脚本。

### 单个案例目录格式

单一的案例代码如[`PINNs求解Burgers`](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/burgers)为例，代码目录分成以下结构：

```shell
.
├──images
│  └──result.jpg
├──src
│  ├──__init__.py
│  ├──dataset.py
│  ├──model.py
│  └──utils.py
├──README.md
├──README_CN.md
├──burgers1D.ipynb
├──burgers1D_CN.ipynb
├──burgers_cfg.yaml
└──train.py
```

### 多个案例目录格式

有时，有多个案例会采用相似的方法和数据集，为了避免代码和文档的重复，此时可以把这几个案例放在一起，提供统一的文档和`src`目录，如：

```shell
.
├──images
│  └──result.jpg
├──src
│  ├──__init__.py
│  ├──dataset.py
│  ├──model.py
│  └──utils.py
├──README.md
├──README_CN.md
├──case1
│  ├──src
│  │  ├──__init__.py
│  │  ├──dataset_1.py
│  │  └──model_1.py
│  ├──case1.ipynb
│  ├──case1_CN.ipynb
│  ├──case1_cfg.yaml
│  ├──README.md
│  ├──README_CN.md
│  └──train.py
├──case2
│  ├──src
│  │  ├──__init__.py
│  │  ├──dataset_2.py
│  │  └──model_2.py
│  ├──case2.ipynb
│  ├──case2_CN.ipynb
│  ├──case2_cfg.yaml
│  ├──README.md
│  ├──README_CN.md
│  └──train.py
├──case3
│  ├──src
│  │  ├──__init__.py
│  │  ├──dataset_3.py
│  │  └──model_3.py
│  ├──case3.ipynb
│  ├──case3_CN.ipynb
│  ├──case3_cfg.yaml
│  ├──README.md
│  ├──README_CN.md
│  └──train.py
```

## 训练文件格式

训练文件train.py为模型训练的入口，格式如下：

```python
import argparse
...
from src import create_training_dataset, create_test_dataset, visual, calculate_l2_error, Burgers1D
# 相关依赖导入，按照python官方库、第三方库、mindflow、src的顺序导入，导入mindflow时，精确到二级目录

set_seed(123456)
np.random.seed(123456)
# 设置随机数

def train():
    # 训练和验证函数，采用mindspore2.0函数式编程范式编写，注意打印内容尽量统一，即："epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) * 1000 :.3f}ms"

if __name__ == '__main__':
    # 调用训练函数

    parser = argparse.ArgumentParser(description="burgers train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./burgers_cfg.yaml")
    args = parser.parse_args()
    # 脚本入参

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    # context设置，由于Ascend和GPU使用的差异，需要使用use_ascend变量进行判断
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print("pid:", os.getpid())
    start_time = time.time()
    train()
    print("End-to-End total time: {} s".format(time.time() - start_time))
```

## README文件格式

其中，总目录中的README对整体背景、技术路线、结果进行讲解，在每个案例中，可以分别在案例的角度描述，注意整体和局部的详略关系，避免重复描述和重复代码。

【必须】`images`：包含了README文件里的所有图片。

【必须】`src`：为了保证训练代码的整洁性，可以单独抽取的函数和类可以统一放在src目录中，`__init__.py`一般为必须，`dataset.py`中包含数据集相关函数和类，`model.py`中包含模型相关函数和类，`utils.py`中包含工具函数和类。

* 注意：类和函数中需要避免'Hard Code'，变量名需要有实际含义，必要的'Magic Number'需要在注释里说明。

【必须】README.md和README_CN.md，中英文README文件，一般包含以下部分：

```md
# 标题

## 概述

简单介绍一下案例的背景、方法、数据集、效果等。

## 快速开始

为用户提供快速运行脚本的方法，一般提供脚本调用和Jupyter Notebook两种方式。其中，脚本调用需要展示启动命令的入参含义

### 训练方式一：在命令行中调用`train.py`脚本

python train.py --config_file_path ./burgers_cfg.yaml --mode GRAPH --device_target Ascend --device_id 0

其中，
`--config_file_path`表示参数文件的路径，默认值'./burgers_cfg.yaml'；

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式，默认值'GRAPH'；

`--device_target`表示使用的计算平台类型，可以选择'Ascend'或'GPU'，默认值'Ascend'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认值0；

### 训练方式二：运行Jupyter Notebook

您可以使用中英文版本的Jupyter Notebook（附链接）逐行运行训练和验证代码。

## 结果展示

用1-2张图的方式展示模型推理的效果，最好为gif。

## Contributor

代码贡献者的gitee id:
代码贡献者的email:

```

## PR创建和合入

请参考[MindScience贡献指南](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)