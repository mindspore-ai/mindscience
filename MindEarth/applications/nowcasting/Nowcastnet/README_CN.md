# NowcastNet: 融入物理机制的生成式短临降水预报模型

## 概述

NowcastNet是由清华大学龙明盛老师团队开发的一个基于雷达数据的短临降水预报模型。 它提供了0-3h的短临降水预报结果，空间分辨率为1km左右。
该模型主要分为evolution和generation两大模块，其中evolution模块融入了物理机制，给出一个粗糙的预测结果。接着，generation模块在此
基础上生成精细化的结果，从而得到最终的降水预报。模型框架图入下图所示(图片来源于论文 [Skilful nowcasting of extreme precipitation with NowcastNet](https://www.nature.com/articles/s41586-023-06184-4))

![nowcastnet](images/nowcastnet.png)

本教程展示了如何通过MindEarth训练和快速推理模型。更多信息参见[文章](https://www.nature.com/articles/s41586-023-06184-4)
本教程中使用原作者开放的[USA-MRMS](https://cloud.tsinghua.edu.cn/d/b9fb38e5ee7a4dabb2a6/)数据集进行训练和推理。由于NowcastNet分为
evolution和generation两个模块，并且这两个模块是独立训练和推理的，因此我们在配置文件`./configs/Nowcastnet.yaml`中设置了`module_name`参数
来进行控制，该参数默认只能设置为`evolution`或者`generation`。

## 快速开始

在[NowcastNet/dataset](https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/nowcastnet/)下载数据并保存，然后在`./configs/Nowcastnet.yaml`中修改`root_dir`路径。

### 运行方式： 在命令行调用`.sh`脚本

若运行evolution模块则在`./configs/Nowcastnet.yaml`中设置`module_name: evolution`。若运行generation模块
则设置`module_name: generation`，后续操作相同。

### 单卡训练

在`./configs/Nowcastnet.yaml`中设置`distribute: False`。

```shell
cd scripts
bash run_standalone_train.sh $device_id
```

### 多卡训练

在`./configs/Nowcastnet.yaml`中设置`distribute: True`。

```shell
cd scripts
bash run_distributed_train.sh $path/to/rank_table.json $device_num $device_start_id
```

### 推理

在`./configs/Nowcastnet.yaml`中设置`load_ckpt: True`。若是evolution模块推理则设置`evolution_ckpt_path`参数，generation模块则设置`generate_ckpt_path`参数。

```shell
cd scripts
bash run_eval.sh $device_id
```

### 结果展示：

#### Evolution模块可视化

下图展示了使用约5w条样本训练后进行推理绘制的结果。

![evo](./images/evo_results.png)

#### Generation模块可视化

下图展示了使用generation模块训练后的1h的预报结果。

![gen](./images/gen_results_1h.png)

## 贡献者

gitee id: Zhou Chuansai

email: chuansaizhou@163.com