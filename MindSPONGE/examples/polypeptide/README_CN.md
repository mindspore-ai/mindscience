# 丙氨酸三肽水溶液体系模拟

`Linux` `GPU` `模型开发` `高级`

<!-- TOC -->

- [丙氨酸三肽水溶液体系模拟](#丙氨酸三肽水溶液体系模拟)
    - [概述](#概述)
    - [整体执行](#整体执行)
    - [准备环节](#准备环节)
    - [模拟多肽水溶液体系示例](#模拟多肽水溶液体系示例)
        - [准备输入文件](#准备输入文件)
        - [加载数据](#加载数据)
        - [构建模拟流程](#构建模拟流程)
        - [运行脚本](#运行脚本)
        - [运行结果](#运行结果)
    - [性能描述](#性能描述)

    - [MindSpore.Numpy方式运行SPONGE](#以MindSpore.Numpy方式运行SPONGE)
    - [MindSPONGE-Numpy运行机制](#MindSPONGE-Numpy运行机制)
        - [CUDA核函数与MindSpore的映射及迁移](#CUDA核函数与MindSpore的映射及迁移)
        - [使用图算融合/算子自动生成进行加速](#使用图算融合/算子自动生成进行加速)
    - [性能描述](#性能描述)

<!-- /TOC -->
<a href="https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/examples/polypeptide/README_CN.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 概述

本篇教程将主要介绍如何在GPU上，使用MindSPONGE进行丙氨酸三肽水溶液体系模拟。

## 整体执行

1. 准备分子模拟输入文件，加载数据，确定计算的分子体系；
2. 定义 SPONGE 模块并初始化，确定计算流程；
3. 运行训练脚本，输出模拟的热力学信息文件，并查看结果；

## 准备环节

实践前，确保已经正确安装MindSpore。如果没有，可以通过[MindSpore安装页面](https://www.mindspore.cn/install)安装MindSpore。

## 模拟多肽水溶液体系示例

SPONGE具有高性能及易用的优势，本教程使用SPONGE模拟多肽水溶液体系。模拟体系为丙氨酸三肽水溶液体系。

### 准备输入文件

本教程模拟体系中需要加载三个输入文件，分别是：

- 属性文件（后缀为`.in`的文件），声明模拟的基本条件，对整个模拟过程进行参数控制。
- 拓扑文件（后缀为`.param7`的文件），拓扑文件描述的是体系内部分子的拓扑关系及各种参数。
- 坐标文件（后缀为`.rst7`的文件），坐标文件描述的是每个原子在体系中的初始时刻的坐标及速度。

拓扑文件和坐标文件可以通过建模过程由AmberTools中自带的tleap工具（下载地址<http://ambermd.org/GetAmber.php>， 遵守GPL协议）建模完成。建模过程如下：

- 打开tleap

    ```bash
    tleap
    ```

- 加载tleap自带的ff14SB力场

    ```bash
    > source leaprc.protein.ff14SB
    ```

- 搭建丙氨酸三肽模型

    ```bash
    > ala = sequence {ALA ALA ALA}
    ```

- 利用tleap加载其自带的tip3p力场

    ```bash
    > source leaprc.water.tip3p
    ```

- 利用tleap中的`slovatebox`溶解丙氨酸三肽链， 完成体系构建。`10.0`代表加入的水距离我们溶解的分子及体系边界至少在`10.0`埃以上

    ```bash
    > solvatebox ala TIP3PBOX 10.0
    ```

- 将建好的体系保存成`parm7`及`rst7`文件

    ```bash
    > saveamberparm ala WATER_ALA.parm7 WATER_ALA_350_cool_290.rst7
    ```

通过tleap构建了所需要的拓扑文件（`WATER_ALA.parm7`）和坐标文件（`WATER_ALA_350_cool_290.rst7`）后，需要通过属性文件声明模拟的基本条件，对整个模拟过程进行参数控制。以本教程中的属性文件`NVT_290_10ns.in`为例，其文件内容如下：

```text
NVT 290k
   mode = 1,                              # Simulation mode ; mode=1 for NVT ensemble
   dt= 0.001,                             # Time step in picoseconds (ps). The time length of each MD step
   step_limit = 1,                        # Total step limit, number of MD steps run
   thermostat=1,                          # Thermostat for temperature ; thermostat=0 for Langevin thermostat
   langevin_gamma=1.0,                    # Gamma_ln for Langevin thermostat represents coupling strength between thermostat and system
   target_temperature=290,               # Target temperature
   write_information_interval=1000,       # Output frequency
   amber_irest=0,                         # Input style ;  amber_irest=1 for using amber style input & rst7 file contains veclocity
   cut=10.0,                              # Nonbonded cutoff distance in Angstroms
```

- `mode`，分子动力学（MD）模式，`1`表示模拟采用`NVT`系综。
- `dt`，表示模拟步长。
- `step_limit`，表示模拟总步数。
- `thermostat`，表示控温方法，`1`表示采用的是`Liujian-Langevin`方法。
- `langevin_gamma`，表示控温器中的`Gamma_ln`参数。
- `target_temperature`，表示目标温度。
- `amber_irest`，表示输入方式，`0`表示使用amber方式输入，`rst7`文件中不包含`veclocity`属性。
- `cut`，表示非键相互作用的距离。

### 加载数据

完成输入文件的构建后，将文件存放在本地工作区的`data`路径下，其目录结构如下：

```text
└─data
    ├─polypeptide
    │      NVT_290_10ns.in                 # specific MD simulation setting
    │      WATER_ALA.parm7                 # topology file include atom & residue & bond & nonbond information
    │      WATER_ALA_350_cool_290.rst7     # restart file record atom coordinate & velocity and box information
```

从三个输入文件中，读取模拟体系需要的参数，用于MindSpore的计算。加载代码如下：

```python
import argparse
import time
from mindspore import context

parser = argparse.ArgumentParser(description='Sponge Controller')
parser.add_argument('--i', type=str, default=None, help='Input .in file')
parser.add_argument('--amber_parm', type=str, default=None, help='paramter file in AMBER type')
parser.add_argument('--c', type=str, default=None, help='initial coordinates file')
parser.add_argument('--r', type=str, default="restrt", help='')
parser.add_argument('--x', type=str, default="mdcrd", help='')
parser.add_argument('--o', type=str, default="mdout", help="")
parser.add_argument('--box', type=str, default="mdbox", help='')
parser.add_argument('--device_id', type=int, default=0, help='')
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id, save_graphs=False)
```

### 构建模拟流程

使用SPONGE中定义的计算力模块和计算能量模块，通过多次迭代进行分子动力学过程演化，使得体系达到我们所需要的平衡态，并记录每一个模拟步骤中得到的能量等数据。为了方便起见，本教程的计算迭代次数设置为`1`，其模拟流程构建代码如下：

```python
from mindsponge.md.simulation import Simulation
from mindspore import Tensor

if __name__ == "__main__":
    simulation = Simulation(args_opt)

    start = time.time()
    compiler = args_opt.o
    save_path = args_opt.o
    simulation.main_initial()
    for steps in range(simulation.md_info.step_limit):
        print_step = steps % simulation.ntwx
        if steps == simulation.md_info.step_limit - 1:
            print_step = 0
        temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene, _ = simulation(Tensor(steps), Tensor(print_step))
        # compute energy and temperature
```

### 运行脚本

执行以下命令，启动训练脚本`main.py`进行训练：

```text
python main.py --i /path/NVT_290_10ns.in \
               --amber_parm /path/WATER_ALA.parm7 \
               --c /path/WATER_ALA_350_cool_290.rst7 \
               --o /path/ala_NVT_290_10ns.out
```

- -`i` 为MD模拟的属性文件，控制模拟过程
- -`amber_parm` 为MD模拟体系的拓扑文件
- -`c` 为我们输入的初始坐标文件
- -`o` 为我们模拟输出的记录文件，其记录了输出每步的能量等信息
- -`path` 为文件所在的路径，在本教程中为`data/polypeptide`

训练过程中，使用属性文件（后缀为`.in`的文件）、拓扑文件（后缀为`.param7`的文件）以及坐标文件（后缀为`.rst7`的文件），通过在指定温度下进行模拟，计算力和能量，进行分子动力学过程演化。

### 运行结果

训练结束后，可以得到输出文件`ala_NVT_290_10ns.out`，体系能量变化被记录在了该文件中，可以查看模拟体系的热力学信息。查看`ala_NVT_290_10ns.out`可以看到如下内容：

```text
_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ _ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_
      1 0.788   -5836.521         48.745       0.891         14.904      9.041    194.479  763.169    -6867.750
   ...
```

其中记录了模拟过程中输出的各类能量， 分别是迭代次数（_steps_），温度（_TEMP_），总能量（_TOT_POT_E_），键长（_BOND_ENE_），键角（_ANGLE_ENE_），二面角相互作用（_DIHEDRAL_ENE_），非键相互作用，其包含静电力及Leonard-Jones相互作用。

## 性能描述

| Parameter                 |   GPU |
| -------------------------- |---------------------------------- |
| Resource                   | GPU (Tesla V100 SXM2); memory 16 GB
| Upload date              |
| MindSpore version          | 1.2
| Training parameter        | step=1
| Output                    | numpy file
| Speed                      | 5.0 ms/step
| Total time                 | 5.7 s
| Script                    | [Link](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/mindsponge/scripts)

## MindSpore.Numpy方式运行SPONGE

除了以Cuda核函数执行的方式运行SPONGE之外，现在我们同时支持以MindSpore原生表达的方式运行SPONGE。计算能量，坐标和力的Cuda核函数均被替换成了Numpy的语法表达，同时拥有MindSpore强大的加速能力。

Sponge-Numpy现在同样支持丙氨酸三肽水溶液体系，如果需要运行Sponge-Numpy，可以使用如下命令：

```shell
python main_numpy.py --i /path/NVT_290_10ns.in \
               --amber_parm /path/WATER_ALA.parm7 \
               --c /path/WATER_ALA_350_cool_290.rst7 \
               --o /path/ala_NVT_290_10ns.out
```

或者直接在 MindSPONGE / examples / polypeptide / scripts 目录下执行：

```shell
bash run_numpy.sh
```

其余步骤均与[丙氨酸三肽水溶液](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/examples/polypeptide/case_polypeptide.md)保持一致。

## MindSPONGE-Numpy运行机制

为了更充分地利用MindSpore的强大特性，以及更好地展示分子动力学算法的运作机制, SPONGE中的Cuda核函数被重构为Numpy语法的脚本，并封装在[md/functions](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/mindsponge/md/functions)模块之中。

MindSpore.Numpy计算套件包含一套完整的符合Numpy规范的接口，使得开发者可以Numpy的原生语法表达MindSpore的模型，同时拥有MindSpore的加速能力。MindSpore.Numpy是建立在MindSpore基础算子(mindspore.ops)之上的一层封装，以MindSpore张量为计算单元,因此可以与其他MindSpore特性完全兼容。更多介绍请参考[这里](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.numpy.html)。

### CUDA核函数与MindSpore的映射及迁移

在丙氨酸三肽水溶液案例中，所有的Cuda核函数均完成了MindSpore改写并完成了精度验证。对于Cuda算法的Numpy迁移，现在提供一个计算LJ Energy的案例供参考：

```cuda
for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength[0].x * int_x;
      dr.y = boxlength[0].y * int_y;
      dr.z = boxlength[0].z * int_z;

      dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
      if (dr2 < cutoff_square) {
        dr_2 = 1. / dr2;
        dr_4 = dr_2 * dr_2;
        dr_6 = dr_4 * dr_2;

        y = (r2.lj_type - r1.lj_type);
        x = y >> 31;
        y = (y ^ x) - x;
        x = r2.lj_type + r1.lj_type;
        r2.lj_type = (x + y) >> 1;
        x = (x - y) >> 1;
        atom_pair_lj_type = (r2.lj_type * (r2.lj_type + 1) >> 1) + x;

        dr_2 = (0.083333333 * lj_type_A[atom_pair_lj_type] * dr_6 - 0.166666666 * lj_type_B[atom_pair_lj_type]) * dr_6;
        ene_lin = ene_lin + dr_2;
      }
    }
    atomicAdd(&lj_ene[atom_i], ene_lin);
```

以上代码首先计算了当前分子与其邻居的距离，对于距离小于`cutoff_square`的分子对，进行后续的能量计算，并且累加到当前分子之上，作为该分子累积能量的一部分。因此，Mindore.Numpy版本的迁移分为两部分：

- 理解Cuda核函数算法
- 进行Numpy拆分以及映射

重构之后的Numpy脚本如下：

```python
nl_atom_serial_crd = uint_crd[nl_atom_serial]
r2_lj_type = atom_lj_type[nl_atom_serial]
crd_expand = np.expand_dims(uint_crd, 1)
crd_d = get_periodic_displacement(nl_atom_serial_crd, crd_expand, scaler)
crd_2 = crd_d ** 2
crd_2 = np.sum(crd_2, -1)
nl_atom_mask = get_neighbour_index(atom_numbers, nl_atom_serial.shape[1])
mask = np.logical_and((crd_2 < cutoff_square), (nl_atom_mask < np.expand_dims(nl_atom_numbers, -1)))
dr_2 = 1. / crd_2
dr_6 = np.power(dr_2, 3.)
r1_lj_type = np.expand_dims(atom_lj_type, -1)
x = r2_lj_type + r1_lj_type
y = np.absolute(r2_lj_type - r1_lj_type)
r2_lj_type = (x + y) // 2
x = (x - y) // 2
atom_pair_lj_type = (r2_lj_type * (r2_lj_type + 1) // 2) + x
dr_2 = (0.083333333 * lj_A[atom_pair_lj_type] * dr_6 - 0.166666666 * lj_B[atom_pair_lj_type]) * dr_6
ene_lin = np.where(mask, dr_2, zero_tensor)
ene_lin = np.sum(ene_lin, -1)
return ene_lin
```

具体步骤如下：

- 将Cuda中的索引取值改写为Numpy的fancy index索引取值。
- 建立一个掩码矩阵，将所有距离大于`cutoff_square`的计算屏蔽。
- 将所有元素级的运算变换为可以广播的Numpy的矩阵计算，中间可能涉及矩阵的形状变换。

### 使用图算融合/算子自动生成进行加速

为了获得成倍的加速收益，MindSPONGE-Numpy默认开启[图算融合](https://www.mindspore.cn/docs/zh-CN/master/design/enable_graph_kernel_fusion.html)以及[自动算子生成](https://gitee.com/mindspore/akg)。这两个加速组件可以为模型提供3倍（甚至更多）的性能提升, 使得MindSPONGE-Numpy达到与原版本性能相近的程度.

在模型脚本中添加如下两行代码即可获得图算融合加速：
(examples/polypeptide/src/main_numpy.py):

```python
# Enable Graph Mode, with GPU as backend, and allow Graph Kernel Fusion
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id, enable_graph_kernel=True)
# Make fusion rules for specific operators
context.set_context(graph_kernel_flags="--enable_expand_ops=Gather --enable_cluster_ops=TensorScatterAdd --enable_recompute_fusion=false")
```
