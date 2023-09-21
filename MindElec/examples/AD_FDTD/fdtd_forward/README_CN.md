# 目录

- [目录](#目录)
    - [麦克斯韦方程组](#麦克斯韦方程组)
    - [基于可微分FDTD的电磁正问题求解器](#基于可微分fdtd的电磁正问题求解器)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [模型训练](#模型训练)
    - [模型精度](#模型精度)
    - [MindScience主页](#mindscience主页)

## 麦克斯韦方程组

麦克斯韦方程组是一组描述电场、磁场与电荷密度、电流密度之间关系的偏微分方程，有激励源的控制方程具体描述如下：

$$
\begin{split}
\nabla\times E&=-\mu \dfrac{\partial H}{\partial t},\\
\nabla\times H&=\epsilon \dfrac{\partial E}{\partial t} + \sigma E + J,
\end{split}
$$

其中$\epsilon,\mu, \sigma$分别是介质的介电常数、磁导率和电导率。$J$是电磁仿真过程中的激励源，常用的加源方式包括点源，线源和面源。本案例为三维天线的S参数仿真问题，激励源为软电压源，可以看作线源，其数学表示为：

$$
J(x, y, z, t)=\delta(x - x_0, y - y_0, z - z_0)f(t)
$$

## 基于可微分FDTD的电磁正问题求解器

时域有限差分（FDTD）方法求解麦克斯韦方程组的过程等价于一个循环卷积网络（RCNN）。简单起见，以二维FDTD为例。当采用CFS-PML对无限大区域进行截断时，TM模式的二维FDTD的第$n$个时间步的更新过程如下：

![FDTD-RCNN-Update](docs/FDTD_RCNN_Update_TM_Mode.png)

利用MindSpore的可微分算子重写更新流程，便可得到端到端可微分FDTD。该求解器基于严格的数值格式，精度与传统FDTD求解器相当。

## 数据集

- 评估数据：从参考文献提取的Invert-F天线的S11参数和微带滤波器的S11与S21参数

## 环境要求

- 硬件（GPU/CPU）
    - 准备GPU/CPU处理器搭载硬件环境
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 脚本说明

### 脚本和样例代码

```path
└─fdtd_forward
  ├─README.md
  ├─docs                                      # README示意图
  ├─src
    ├──constants.py                           # 物理常数
    ├──antenna.py                             # 设置天线介质参数与激励
    ├──grid_helper.py                         # 在FDTD网格上建模的辅助类
    ├──lumped_element.py                      # 集总元件类
    ├──full3d.py                              # 三维可微分FDTD
    ├──cfs_pml.py                             # CFS-PML边界条件
    ├──waveforms.py                           # 时域波形
    ├──utils.py                               # 功能函数
    ├──polts.py                               # 绘图函数
    ├──solver.py                              # 基于ADFDTD的求解器
  ├─dataset                                   # 预先生成的数据集
  ├─solve_invert_f.py                         # 求解Invert-F天线S11参数的脚本
  ├─solve_microstrip_filter.py                # 求解Microstrip Filter S11和S21参数的脚本
```

## 脚本参数

在`solve_invert_f.py`（或`solve_microstrip_filter.py`）中可以配置求解参数。

```python
"device_target": "GPU",                 # 设备名称GPU
"nt": 3000,                             # 时间步个数
"max_call_depth": 1000,                 # 函数调用最大深度
"dataset_dir": "./dataset",             # 数据集路径
"result_dir": "./result",               # 反演结果保存路径
"cfl_number": 0.9,                      # CFL系数
"fmax": 10e9,                           # 最高频率
```

## 模型训练

您可以通过`solve_invert_f.py`和`solve_microstrip_filter.py`脚本分别定义并求解Invert-F天线的S11参数和微带滤波器的S11与S21参数：

```shell
python solve_invert_f.py --device_target 'GPU'
```

和

```shell
python solve_microstrip_filter.py --device_target 'GPU'
```

## 模型精度

求解结束后，脚本会自动绘制S参数，并与`dataset`中的参考结果进行比较。
Invert-F天线的计算结果与参考结果的对比如下图所示：

![invert_f_result](./docs/invert_f_s_parameters.png)

微带滤波器的S参数计算结果与参考结果的对比如下图所示：

![invert_f_result](./docs/microstrip_filter_s_parameters.png)

## MindScience主页

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
