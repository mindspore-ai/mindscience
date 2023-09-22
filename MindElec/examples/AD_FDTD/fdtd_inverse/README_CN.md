# 目录

- [目录](#目录)
    - [麦克斯韦方程组](#麦克斯韦方程组)
    - [基于可微分FDTD的电磁逆问题求解器](#基于可微分fdtd的电磁逆问题求解器)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
    - [模型训练](#模型训练)
        - [训练日志](#训练日志)
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

其中$\epsilon,\mu, \sigma$分别是介质的介电常数、磁导率和电导率。$J$是电磁仿真过程中的激励源，常用的加源方式包括点源，线源和面源。本案例为二维TM极化的介质反演问题，激励源为$z$方向无限延伸的线源（二维时可看作点源），其数学表示为：

$$
J(x, y, t)=\delta(x - x_0, y-y_0)f(t)
$$

## 基于可微分FDTD的电磁逆问题求解器

时域有限差分（FDTD）方法求解麦克斯韦方程组的过程等价于一个循环卷积网络（RCNN）。当采用CFS-PML对无限大区域进行截断时，TM模式的二维FDTD的第$n$个时间步的更新过程如下：

![FDTD-RCNN-Update](docs/FDTD_RCNN_Update_TM_Mode.png)

利用MindSpore的可微分算子重写更新流程，便可得到端到端可微分FDTD。将端到端可微分FDTD、可微分介质拓扑映射和可微分损失函数相结合，利用MindSpore框架的自动微分能力和各种基于梯度的优化器，便可求解如电磁逆散射问题等各种电磁逆问题。整体求解流程为：

![ADFDTD_Flow](docs/AD_FDTD_Flow.png)

案例为TM模式的二维电磁逆散射问题，问题的具体设置如下图所示：

![Inversion_Problem](docs/inversion_problem_setup.png)

反演目标为两个相对节点常数为$4$的介质圆柱，反演区域为$\Omega=\{(x,y)|0.3\leq x\leq 0.7, 0.3\leq y\leq 0.7\}$，反演区域外侧设置有$4$个激励源（红色三角）和$8$个观察点（绿色圆点）。待优化参数为$\rho(x,y)$，其中$(x,y)\in\Omega$。通过可微分映射得到反演区域内的相对介电常数$\varepsilon_r(x,y)=\varepsilon_{r,\min} + \text{elu}(\rho, \alpha=1e-2)$。网络的输入为激励源的时域波形$f(t)$，网络的输出为各观察点处的电场$E_z$，损失函数为均方误差函数。通过MIndSpore的自动微分功能得到损失函数相对$\rho$的梯度，之后便可用SGD、Adam等优化器对参数$\rho$进行优化。

## 数据集

- 训练数据：采用有限差分方法计算得到的4个点源激励在8个观察点上感应出的电场分量$E_z$时域波形
- 评估数据：待反演的介质分布

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
└─AD_FDTD
  ├─README.md
  ├─docs                                      # README示意图
  ├─src
    ├──constants.py                           # 物理常数
    ├──base_topology_designer.py              # 生成介质拓扑的基类
    ├──transverse_mangetic.py                 # TM模式的二维可微分FDTD
    ├──transverse_electric.py                 # TE模式的二维可微分FDTD
    ├──cfs_pml.py                             # CFS-PML边界条件
    ├──waveforms.py                           # 时域波形
    ├──utils.py                               # 功能函数
    ├──metric.py                              # 反演结果评价函数
    ├──solver.py                              # 基于ADFDTD的求解器
  ├─dataset                                   # 预先生成的数据集
  ├─solve.py                                  # 求解脚本
```

## 脚本参数

在`solve.py`中可以配置求解参数。

```python
"epoch": 100,                           # 训练轮数
"lr": 0.1,                              # 学习率
"device_target": "GPU",                 # 设备名称GPU
"nt": 350,                              # 时间步个数
"max_call_depth": 2000,                 # 函数调用最大深度
"dataset_dir": "./dataset",             # 数据集路径
"result_dir": "./result",               # 反演结果保存路径
```

## 模型训练

您可以通过`solve.py`脚本定义并求解电磁反演问题：

```shell
python solve.py --epoch 100
    --device_target 'GPU'
```

### 训练日志

脚本提供了边训练边评估的功能：

```log
Epoch: [  0 / 100], loss: 2.114364e-01
Epoch: [  1 / 100], loss: 1.751708e-01
Epoch: [  2 / 100], loss: 1.517662e-01
Epoch: [  3 / 100], loss: 1.315168e-01
Epoch: [  4 / 100], loss: 1.142960e-01
Epoch: [  5 / 100], loss: 9.941817e-02
...
Epoch: [ 95 / 100], loss: 1.188854e-04
Epoch: [ 96 / 100], loss: 1.157768e-04
Epoch: [ 97 / 100], loss: 1.127892e-04
Epoch: [ 98 / 100], loss: 1.099153e-04
Epoch: [ 99 / 100], loss: 1.071493e-04
```

## 模型精度

求解结束后，脚本会自动评估反演效果，输出反演结果的PSNR和SSIM：

```log
[epsr] PSNR: 27.835317 dB, SSIM: 0.963564
```

最终反演结果如下图所示：

![epsr_result](./docs/epsr_reconstructed.png)

## MindScience主页

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
