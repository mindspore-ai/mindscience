[ENGLISH](README.md) | 简体中文

# 目录

- [GOMO 描述](#Deep-Hpms-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [训练过程](#训练过程)
- [模型说明](#模型说明)
    - [评估性能](#评估性能)

## [GOMO 描述](#目录)

海洋广义算子建模（GOMO）是基于[OpenArray v1.0](https://gmd.copernicus.org/articles/12/4729/2019/gmd-12-4729-2019-discussion.html)的三维海洋模型，OpenArray v1.0是用于海洋建模和并行计算解耦的简单算子库（Xiaomeng Huang et al，2019）。GOMO是一种使用有限微分算法求解偏微分方程的数值解模型，使用MindSpore和GPU/Ascend求解这些PDE方程能得到较大的性能提升。

## [数据集](#目录)

数据集: Seamount

- 数据size: 65x49x21
- 数据格式: `.nc`文件
- 数据集位于`./data`目录下，目录结构如下:

```text
├── data
│   └── seamount65_49_21.nc
```

您如果需要手动下载数据集或checkpoints文件，
请访问[此链接](https://download.mindspore.cn/mindscience/SciAI/sciai/model/ocean_model/)。

## [环境要求](#目录)

- 硬件（Ascend/GPU）
    - 使用 Ascend 或 GPU 处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 欲了解更多信息，请查看以下资源:
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#目录)

通过官网安装好MindSpore和上面需要的[数据集](#数据集)后，就可以开始训练如下:

- 在 Ascend 或 GPU 上运行

默认:

```bash
python train.py
```

完整命令:

```bash
python train.py \
    --load_data_path ./data \
    --output_path ./data/outputs \
    --save_ckpt true \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/model_final.ckpt \
    --force_download false \
    --download_data ocean_model \
    --im 65 \
    --jm 49 \
    --kb 21 \
    --stencil_width 1 \
    --epochs 10 \
    --amp_level O0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── deep_hpms
│   ├── checkpoints                      # checkpoint文件
│   ├── data
│   │   └── seamount65_49_21.nc          # 数据集文件
│   ├── logs                             # 日志文件目录
│   ├── src                              # 源代码目录
│   │   ├── GOMO.py                      # GOMO模型
│   │   ├── Grid.py                      # 网格定义
│   │   ├── stencil.py                   # stencil算子
│   │   ├── oa_operator.py               # stencil kernel算子
│   │   ├── read_var.py                  # 从数据集中读取变量
│   │   └── utils.py                     # 模型setup
│   ├── config.yaml                      # 超参数配置
│   ├── README.md                        # 英文模型说明
│   ├── README_CN.md                     # 中文模型说明
│   ├── train.py                         # python训练脚本
│   └── eval.py                          # python评估脚本
```

### [训练过程](#目录)

- 在 Ascend 或 GPU 上运行

  ```bash
  python train.py
  ```

## [模型说明](#目录)

可设置的模型参数如下:

| 参数             | 说明                            | 默认值                            |
|----------------|-------------------------------|--------------------------------|
| load_data_path | 加载数据的路径                       | ./data                         |
| output_path    | 保存结果的路径                       | ./data/outputs                 |
| save_ckpt      | 是否保存checkpoint                | true                           |
| save_ckpt_path | checkpoint保存路径                | ./checkpoints                  |
| load_ckpt_path | checkpoint加载路径                | ./checkpoints/model_final.ckpt |
| force_download | 是否强制下载数据                      | false                          |
| download_data  | 需下载数据集/checkpoints的模型         | ocean_model                    |
| amp_level      | MindSpore自动混合精度等级             | O0                             |
| mode           | MindSpore图模式(0)或Pynative模式(1) | 0                              |
| device_id      | 设置用来训练推理的卡id                  | None                           |
| im             | GOMO变量 im                     | 65                             |
| jm             | GOMO变量 jm                     | 49                             |
| kb             | GOMO变量 kb                     | 21                             |
| stencil_width  | stencil宽度                     | 1                              |
| epochs         | 时期（迭代次数）                      | 10                             |

### [评估性能](#目录)

| 参数                  | 值                               |
|---------------------|---------------------------------|
| Resource            | GPU(Tesla V100 SXM2)，Memory 16G |
| Dataset             | Seamount                        |
| Training Parameters | step=10, im=65, km=49, kb=21    |
| Outputs             | numpy file                      |
| Speed               | 17 ms/step                      |
| Total time          | 3 mins                          |
