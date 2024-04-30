# MindSpore Chemistry Release Notes

[View English](./RELEASE.md)

MindSpore Chemistry是一个基于MindSpore构建的化学套件，致力于高效使能AI与化学的联合创新，践行AI与化学结合的全新科学研究范式。

## MindSpore Chemistry 0.1.0 Release Notes

### 主要特性

* 提供**分子生成案例**：提供了高熵合金组分设计任务的案例，基于主动学习流程利用AI模型分阶段进行高熵合金的组分生成、筛选，可以辅助化学家进行更加高效的材料设计工作。
* 提供**分子预测案例**：提供了有机分子能量预测的案例，基于等变计算库构建了NequIP和Allegro模型，根据分子体系中原子数与原子位置等信息计算分子体系能量。
* 提供**电子结构预测案例**：提供了DeephE3nn模型案例，基于E3的等变神经网络，利用原子的结构去预测其哈密顿量。
* 提供**晶体材料性质预测案例**：提供了Matformer模型，基于图神经网络和Transformer架构的深度学习模型，用于预测晶体材料的各种性质。
* 提供**图计算相关库**：提供了图计算相关接口模块，如图数据集处理、图汇聚操作等。
* 提供**等变计算库**：提供了Irreps、Spherical Harmonics等底层等变计算接口，同时也提供了等变激活层、等变线性层等基于底层等变接口构建的高阶等变神经网络层接口，旨在为用户提供方便调用并构建等变神经网络的功能。

### 贡献者

感谢以下开发者做出的贡献:

yufan, wangzidong, liuhongsheng, gongyue, gengchenhua, linghejing, yanchaojie, suyun, wujian, caowenbin
