# 目录

- [目录](#目录)
- [点云数据生成](#点云数据生成)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [python导出json文件和stp文件](#python导出json文件和stp文件)
    - [生成点云数据](#生成点云数据)
- [MindScience主页](#mindscience主页)

# 点云数据生成

为了将电磁仿真的模型转换成神经网络可以识别的模式，我们提供将模型转换成点云数据的点云生成工具。该工具的使用分为导出几何/材料信息和点云生成两步：

## 导出几何/材料信息

首先需要从CST软件中导出模型的几何和材料信息。我们提供将cst格式文件转换为python可读取的stp文件的两种自动化执行脚本：

- 基于CST的VBA接口自动调用CST导出json文件和stp文件

  打开CST软件的VBA Macros Editor， 导入`export_stp.bas`文件，将json文件和stp文件路径改成你想要存放的位置，然后点击`Run`即可导出json文件和stp文件。其中，stp文件是模型各个子部件的几何结构；json文件中包含了模型的端口位置以及stp文件对应的材料信息。
- 使用python调用CST导出json文件和stp文件

  直接运行`export_stp.py`文件即可。需要注意，目前只有2019或更新版本的CST才支持python调用。

## 生成点云数据

MindSpore Elec框架提供将stp文件高效转化为点云张量数据的接口`PointCloud`，`generate_cloud_point.py`文件提供该接口调用脚本。

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore Elec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- 如需查看详情，请参见如下资源：
    - [MindSpore Elec教程](https://www.mindspore.cn/mindelec/docs/zh-CN/master/intro_and_install.html)
    - [MindSpore Elec Python API](https://www.mindspore.cn/mindelec/docs/zh-CN/master/mindelec.architecture.html)

# 脚本说明

## 脚本及样例代码

```path
.
└─generete_pointcloud
  ├─README.md
  ├─export_stp.py                             # python导出json文件和stp文件
  ├─export_stp.bas                            # VAB导出json文件和stp文件
  ├─generate_cloud_point.py                   # 生成点云数据
```

## python导出json文件和stp文件

```shell  
python export_stp.py --cst_path CST_PATH
                     --stp_path STP_PATH
                     --json_path JSON_PATH
```

其中，`cst_path`用来指定需要导出stp的cst文件的路径，`stp_path`和`json_path`分别用来指定导出的stp和json文件存放的路径。

## 生成点云数据

```shell  
python generate_cloud_point.py --stp_path STP_PATH
                               --json_path JSON_PATH
                               --material_dir MATERIAL_DIR
                               --sample_nums (500, 2000, 80)
                               --bbox_args (-40., -80., -5., 40., 80., 5.)
```

其中，`stp_path`和`json_path`分别用来指定用来生成点云的stp和json文件的路径；`material_dir`用来指定stp对应的材料信息的路径，材料信息可以直接在CST软件中导出；`sample_nums`用来指定x，y，z三个维度分别生成多少个点云数据；`bbox_args`用来指定生成点云数据的区域，即(x_min, y_min, z_min, x_max, y_max, z_max)。

# MindScience主页

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
