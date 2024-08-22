# MindSPONGE 案例解析

MindSONGE 是一款基于MindSpore开发的模块化、高通量、端到端可微的下一代智能分子模拟程序库。由深圳湾实验室、华为MindSpore开发团队、北京大学和昌平实验室共同开发。

## 软件架构

![MindSPONGE Architecture](https://gitee.com/helloyesterday/mindsponge/raw/develop/docs/mindsponge.png)

## 模拟流程

1. 通过template或者pdb、mol2格式的输入文件定义一个system，基本单元为Molecule()。

2. 通过ForceField()或者加载一个神经网络训练力场，对system进行建模，得到一个potential。在potential的基础上，可以增加bias_potential，应用于增强采样/软约束。

3. 可以使用UpdaterMD()定义一个动力学模拟过程，或者使用MindSpore内部的optimizer，如Adam()等等，来对system进行演化。

4. 将system,potential,optimizer包装成一个Sponge()实例，就可以开始运行模拟了。

5. 最后就是使用各种callback来定义输出格式，RunInfo()可以在屏幕输出每一步的结果，WriteH5MD可以将每一步的轨迹保存到一个h5md格式的文件中，并且支持VMD的可视化插件。

## 案例简介

**tutorial_c01.py**：定义2个SPCE模型的真空水分子，并做能量极小化。

**tutorial_c02.py**：定义64个TIP3P模型的真空水分子，和一个球形谐振子势，在该约束下进行NVT模拟。

**tutorial_c03.py**：定义125个TIP3P模型的周期性水盒子，并做能量极小化，以及NVT和NPT模拟。

**tutorial_c04.py**：定义1个氨基酸链和水分子的混合体系，做能量极小化，以及SETTLE约束下的NVT模拟。

**tutorial_p01.py**：读取1个pdb蛋白质分子，在真空条件下做能量极小化。

**tutorial_p02.py**：读取1个pdb蛋白质分子，构建周期性水盒子，做能量极小化。

**tutorial_p03.py**：读取p02保存的水盒子，进行蛋白模拟流程示例：NVT模拟—NPT模拟—成品模拟。

**tutorial_p04.py**：读取1个pdb蛋白质分子，在真空条件下做能量极小化，定义反应坐标CV，然后进行MetaD增强采样。

## 运行方法

`.ipynb`格式的文件可以在jupyter notebook中运行，`.py`格式的文件可以通过下面的命令运行：

```shell
# -e    运行平台：Ascend/GPU，默认：GPU
# -id   device id，默认：0
python tutroial_xxx.py -e Ascend -id 0
```

> 因为一些案例依赖上一步的结果，所以请按照编号的顺序执行案例
