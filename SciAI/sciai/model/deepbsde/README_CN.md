[ENGLISH](README.md) | 简体中文

# 目录

- [DeepBSDE 描述](#DeepBSDE-描述)
- [HJB方程](#HJB方程)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)
- [随机过程描述](#随机过程描述)

## [DeepBSDE 描述](#目录)

DeepBSDE网络开发了一种深度学习策略用以求解一大类的高维非线性抛物面偏微分方程。

[论文](https:#www.pnas.org/content/115/34/8505): Han J , Arnulf J , Weinan E . Solving high-dimensional partial differential equations using deep learning[J]. Proceedings of the National Academy of Sciences, 2018:201718942-.

## [HJB方程](#目录)

哈密顿-雅各比-贝尔曼（HJB）方程是确定性离散动态规划算法在时间上的连续化模拟，现已成为经济学、行为科学、计算机科学甚至生物学等许多领域的基石，其中智能决策是主要问题。

## [环境要求](#目录)

- 硬件（GPU）
    - 使用 GPU 处理器准备硬件环境。
- 框架
    - [Mindspore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [Mindspore教程](#https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [Mindspore Python API](#https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#目录)

通过官网安装好MindSpore后，就可以开始训练和验证如下:

- 在 GPU 上运行

默认:

```bash
python train.py
```

完整命令如下:

```bash
python train.py \
    --save_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/deepbsde_HJBLQ_end.ckpt \
    --log_path ./logs \
    --print_interval 100 \
    --total_time 1.0 \
    --dim 100 \
    --num_time_interval 20 \
    --y_init_range 0 1 \
    --num_hiddens 110 110 \
    --lr_values 0.01 0.01 \
    --lr_boundaries 1000 \
    --num_iterations 1001 \
    --batch_size 64 \
    --valid_size 256 \
    --sink_size 100 \
    --file_format MINDIR \
    --amp_level O0 \
    --device_id 0 \
    --mode 0
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
.
├── src
│     ├── config.py            # 配置解析脚本
│     ├── equation.py          # 方程定义和数据集helper文件
│     ├── eval_utils.py        # 评估回调和其他utils
│     └── net.py               # DeepBSDE 网络结构
├── config.yaml                # 超参数配置
├── export.py                  # 导出模型 API entry
├── README_CN.md               # 模型中文说明
├── README.md                  # 模型英文说明
└── train.py                   # python训练脚本
└── eval.py                    # python验证脚本
```

### [脚本参数](#目录)

训练和评估的超参数设置可以在 `config.yaml` 文件中进行配置

- HBJ参数配置

| 参数                | 描述                         | 默认值                                                 |
|-------------------|----------------------------|-----------------------------------------------------|
| eqn_name          | 方程名                        | HJBLQ                                               |
| save_ckpt         | 是否保存checkpoint             | true                                                |
| load_ckpt         | 是否加载checkpoint             | false                                               |
| save_ckpt_path    | checkpoint保存路径             | ./checkpoints                                       |
| load_ckpt_path    | checkpoint加载路径             | ./checkpoints/discriminator/deepbsde_HJBLQ_end.ckpt |
| log_path          | 日志保存路径                     | ./logs                                              |
| print_interval    | 损失与时间打印间隔                  | 100                                                 |
| total_time        | 方程函数的总时间                   | 1.0                                                 |
| dim               | 隐藏层的维度                     | 100                                                 |
| num_time_interval | 时间间隔的个数                    | 20                                                  |
| y_init_range      | y_init随机的初始化取值范围           | [0, 1]                                              |
| num_hiddens       | 一组隐藏层的过滤数                  | [110, 110]                                          |
| lr_values         | 分段常量学习率的lr值                | [0.01, 0.01]                                        |
| lr_boundaries     | 分段常量学习率的lr边界               | [1000]                                              |
| num_iterations    | 迭代次数                       | 2001                                                |
| batch_size        | 训练时的批尺寸                    | 64                                                  |
| valid_size        | 评估时的批尺寸                    | 256                                                 |
| sink_size         | 数据下沉尺寸                     | 100                                                 |
| file_format       | 导出的模型格式                    | MINDIR                                              |
| amp_level         | MindSpore自动混合精度等级          | O0                                                  |
| device_id         | 处理器卡号                      | None                                                |
| mode              | MindSpore静态图模式（0）或动态图模式（1） | 0                                                   |

### [训练流程](#目录)

   ```bash
   python train.py
   ```

  以上命令会将训练过程打印到控制台：

  ```console
  step: 0, loss: 1225.2937, interval: 8.1262, total: 8.1262
  eval loss: 4979.3413, Y0: 0.2015
  step: 100, loss: 320.9811, interval: 11.70984, total: 19.2246
  eval loss: 1399.8747, Y0: 1.1023
  step: 200, loss: 160.01154, interval: 6.7937, total: 26.0184
  eval loss: 807.4655, Y0: 1.4009
  ...
  ```

  训练完成后，模型checkpoint将保存在`save_ckpt_path`中, 默认为`./checkpoints`目录。

### [推理流程](#目录)

  在运行下面的命令之前，请检查`config.yaml`中的checkpoint文件加载路径`load_ckpt_path`, 比如`./checkpoints/deepbsde_HJBLQ_end.ckpt`

  ```bash
  python eval.py
  ```

  以上命令会将评估结果打印到控制台：

  ```console
  eval loss: 5.146923065185527, Y0: 4.59813117980957
  Total time running eval 5.8552136129312079 seconds
  ```

## [随机过程描述](#目录)

  equation.py 中使用了随机采样，可以将通过固定随机种子来固定随机性。