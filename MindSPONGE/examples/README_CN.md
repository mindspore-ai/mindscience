# MindSPONGE

- [简介](#简介)
- [数据](#数据)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本介绍](#脚本介绍)
    - [脚本和示例代码](#脚本和示例代码)
    - [模拟过程](#模拟过程)
- [结果](#结果)

## 简介

MindSPONGE包含了分子模拟过程中相关的功能函数以及分子模拟案例集合，其中包含了生物、材料、制药领域中的不同的分子体系的模拟。分子建模中，包含了基于传统分子模拟方法的相关案例，也会在后期包含AI+分子模拟的案例，详情请查看[案例](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/mindsponge/examples)。欢迎大家积极参与和关注。

下面的案例将展示如何在`GPU`上，使用MindSPONGE快速进行分子模拟。

## 数据

该案例中使用了三个不同的输入文件，分别是属性文件`NVT_290_10ns.in`，拓扑文件`WATER_ALA.parm7`，以及坐标文件 `WATER_ALA_350_cool_290.rst7`。三个输入文件都在`data/polypeptide`文件夹中。

拓扑文件和坐标文件可以由开源工具`AmberTools`中的`tleap`([链接](<http://ambermd.org/GetAmber.php>))生成，更多细节，请查看案例完整教程：

- [MindSPONGE 教程](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/mindsponge/examples/case_polypeptide.md)

![ALA Aqueous System](https://images.gitee.com/uploads/images/2021/0323/184453_4bd9b1a6_8142020.png "图片1.png")

## 环境要求

- 硬件设备: `GPU`
    - MindSPONGE现有案例目前只支持 `GPU`设备.
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 更多信息，请查看详细介绍:
    - [MindSPONGE 教程](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/mindsponge/examples)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## 快速入门

安装完成MindSpore后, 运行如下命令:

```shell
python main_poly.py --i /path/NVT_290_10ns.in \
               --amber_parm /path/WATER_ALA.parm7 \
               --c /path/WATER_ALA_350_cool_290.rst7 \
               --o /path/ala_NVT_290_10ns.out
```

`path` 是存储输入文件的具体路径，在本案例中，该路径为`data/polypeptide`。

## 脚本介绍

### 脚本和示例代码

```shell
├── MindSPONGE
    ├── main.py                                  # launch Simulation
    ├── main_poly.py                             # launch Simulation for polypeptide
    ├── md
        ├── space
            ├── md_information.py                    # save md information module
            ├── system_information.py                # subclass of md information
        ├── partition
            ├── neighbor_list.py                     # neighbor_list module
        ├── control
            ├── langevin_liujian_md.py               # langevin_liujian_md module
            ├── mc_baro.py                           # Monte Carlo pressure control method
            ├── bd_baro.py                           # Berendsen pressure control method
            ├── crd_molecular_map.py                 # molecular map module
        ├── potential
            ├── bond.py                              # bond module
            ├── angle.py                             # angle module
            ├── dihedral.py                          # dihedral module
            ├── nb14.py                              # nb14 module
            ├── lennard_jones.py                     # lennard_jones module
            ├── particle_mesh_ewald.py               # particle_mesh_ewald module
            ├── restrain.py                          # restrain module
            ├── simple_constrain.py                  # simple constrain module
            ├── vatom.py                             # virtual atoms
        ├── simulation.py                        # MindSPONGE simulation
        ├── simulation_poly.py                   # MindSPONGE simulation for polypeptide
```

### 模拟过程

启动模拟过程，运行如下命令：

```shell
python main_poly.py --i /path/NVT_290_10ns.in \
               --amber_parm /path/WATER_ALA.parm7 \
               --c /path/WATER_ALA_350_cool_290.rst7 \
               --o /path/ala_NVT_290_10ns.out
```

## 结果

模拟结束后，模拟结果会存储在指定的`.out`文件中，在这里为`ala_NVT_290_10ns.out`。其存储的内容为：

```text
_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ _ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_
      0 0.000   -5713.804         0.037       0.900         14.909      9.072    194.477  765.398    -6698.648
   ...
```

存储结果记录了模拟过程中输出的各类信息，包含步骤（_steps_），温度（_TEMP_），总能量（_TOT_POT_E_），键长（_BOND_ENE_），键角（_ANGLE_ENE_），二面角相互作用（_DIHEDRAL_ENE_），非键相互作用，其包含静电力及Leonard-Jones相互作用。
