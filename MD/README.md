# 多尺度分子动力学模拟套件

## 概述

多尺度分子动力学模拟套件基于国产深度学习框架MindSpore开发，提供了SPONGE分子动力学模拟软件的Python接口实现。套件涵盖蛋白质能量极小化、模拟系统构建、分子动力学模拟、增强采样等多种应用场景。

## 应用案例

### 1. SPONGE P01 - 蛋白质能量极小化问题

**路径**: `MD/tutorial_p01.py`

**功能描述**: 加载PDB文件（case1.pdb）并重建氢原子，使用AMBER.FF14SB力场进行能量计算，采用最速下降法进行能量最小化。运行500步优化，每10步保存一次轨迹，输出H5MD格式文件。

**主要文件**:
- `tutorial_p01.py` - 主程序

**关键技术**:
- `Protein` - 蛋白质结构加载
- `ForceField` - 力场定义
- `SteepestDescent` - 最速下降优化器
- `WriteH5MD` - H5MD轨迹输出

---

### 2. SPONGE B01 - 创建模拟系统问题

**路径**: `MD/tutorial_b01.py`

**功能描述**: 手动创建简单的水分子模拟系统，定义原子坐标和键连接关系，手动设置键能和角势能参数。使用Berendsen恒温器控制温度（300K），采用Leap Frog积分器运行1000步MD模拟，保存速度和力信息到H5MD文件。

**主要文件**:
- `tutorial_b01.py` - 主程序

**关键技术**:
- `Molecule` - 分子定义
- `BondEnergy` - 键能
- `AngleEnergy` - 角势能
- `UpdaterMD` - MD更新器
- `VelocityGenerator` - 速度生成器

---

### 3. SPONGE B06 - 蛋白质分子的最小化和MD问题

**路径**: `MD/tutorial_b06.py`

**功能描述**: 加载蛋白质PDB文件（case2.pdb）并重建氢原子，使用AMBER.FF14SB力场。采用两阶段模拟：第一阶段500步能量最小化，第二阶段2000步MD模拟（温度300K）。使用Langevin恒温器和Velocity Verlet积分器，输出H5MD格式轨迹文件。

**主要文件**:
- `tutorial_b06.py` - 主程序

**关键技术**:
- `Protein` - 蛋白质结构
- `WithEnergyCell` - 能量计算单元
- `UpdaterMD` - MD更新器
- `SteepestDescent` - 最速下降优化
- `change_optimizer` - 优化器切换

---

### 4. SPONGE B08 - 仅移动氢原子来最小化蛋白质的能量问题

**路径**: `MD/tutorial_b08.py`

**功能描述**: 加载蛋白质PDB文件（case2.pdb），使用AMBER.FF99SB力场。核心特性是使用MaskedDriven限制重原子，仅移动氢原子进行能量最小化。采用动态学习率（指数衰减）的最速下降优化，使用邻居列表加速计算，运行500步优化，保存最终PDB文件。

**主要文件**:
- `tutorial_b08.py` - 主程序

**关键技术**:
- `MaskedDriven` - 掩码驱动器
- `WithForceCell` - 力计算单元
- `NeighbourList` - 邻居列表
- `ExponentialDecayLR` - 指数衰减学习率
- `heavy_atom_mask` - 重原子掩码

---

### 5. SPONGE A01 - 集体变量、度量和分析问题

**路径**: `MD/tutorial_a01.py`

**功能描述**: 加载丙氨酸二肽PDB文件，定义两个集体变量（CV）：phi和psi扭转角。使用metrics功能实时监测和分析CV。采用两阶段模拟：第一阶段100步能量最小化，第二阶段1000步MD模拟（温度300K）。使用Langevin恒温器和Velocity Verlet积分器，可通过`analyse()`方法获取CV值。

**主要文件**:
- `tutorial_a01.py` - 主程序

**关键技术**:
- `Torsion` - 扭转角定义
- `metrics` - 度量监测
- `analyse()` - 分析方法
- `VelocityVerlet` - Velocity Verlet积分器
- `Langevin` - Langevin恒温器

---

### 6. SPONGE A03 - 能量包装和综合淬火采样（ITS）问题

**路径**: `MD/tutorial_a03.py`

**功能描述**: 加载丙氨酸二肽PDB文件，使用AMBER.FF14SB力场。核心特性是使用ITS（Integrated Tempering Sampling）增强采样方法。ITS参数：模拟温度300K，温度范围270K-670K，温度bin数200，更新频率100步，非线性温度分布。定义phi和psi扭转角作为监测变量。采用两阶段模拟：第一阶段100步能量最小化，第二阶段1000步ITS-MD模拟。

**主要文件**:
- `tutorial_a03.py` - 主程序

**关键技术**:
- `ITS` - 综合淬火采样
- `WithEnergyCell` - 能量计算单元
- `wrapper` - 能量包装
- `Torsion` - 扭转角定义
- 增强采样技术

---

## 技术特点

1. **分类体系**:
   - **P系列**（Protein）：蛋白质相关基础操作
   - **B系列**（Basic）：基础功能演示
   - **A系列**（Advanced）：高级功能应用

2. **力场支持**:
   - AMBER.FF14SB：主要用于蛋白质模拟
   - AMBER.FF99SB：用于特定优化场景

3. **模拟流程**: 多数案例采用"先最小化，后MD"的两阶段策略

4. **输出格式**: 统一使用H5MD格式保存轨迹数据

5. **增强采样**: A03展示了ITS增强采样技术，适用于复杂势能面探索

## 案例路径对照

| 案例名称 | 实际路径 |
|---------|---------|
| P01 蛋白质能量极小化 | `./tutorial_p01.py` |
| B01 创建模拟系统 | `./tutorial_b01.py` |
| B06 蛋白质分子最小化和MD | `./tutorial_b06.py` |
| B08 仅移动氢原子最小化能量 | `./tutorial_b08.py` |
| A01 集体变量、度量和分析 | `./tutorial_a01.py` |
| A03 能量包装和ITS | `./tutorial_a03.py` |

完整教程路径：`Biology/MindSPONGE/tutorials/`

## 运行环境

- MindSpore >= 2.0
- MindSPONGE
- Python >= 3.7

## 使用方式

```bash
# 运行蛋白质能量极小化
python tutorial_p01.py

# 运行模拟系统创建
python tutorial_b01.py

# 运行蛋白质最小化和MD
python tutorial_b06.py

# 运行氢原子优化
python tutorial_b08.py

# 运行集体变量分析
python tutorial_a01.py

# 运行ITS增强采样
python tutorial_a03.py
```

## 目录结构

```
MD/
├── README.md
├── tutorial_p01.py    # 蛋白质能量极小化
├── tutorial_b01.py    # 创建模拟系统
├── tutorial_b06.py    # 蛋白质最小化和MD
├── tutorial_b08.py    # 氢原子优化
├── tutorial_a01.py    # 集体变量分析
└── tutorial_a03.py    # ITS增强采样
```

## 相关资源

- MindSPONGE完整教程: `Biology/MindSPONGE/tutorials/`
- SPONGE核心代码: `Biology/MindSPONGE/src/sponge/`
