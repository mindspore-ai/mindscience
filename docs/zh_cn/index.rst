MindSpore Science 文档
======================

简介
----

将 AI 与科学计算相结合（即 AI+科学计算），是指利用人工智能技术（如机器学习、深度学习等）对科学难题进行计算和分析。这种结合突破了传统数学模型和算法的局限，能够借助 AI 强大的计算能力，探索未知领域，并提升计算效率与准确性。AI+科学计算的能力主要来源于大数据的支持和算法的优化，通过不断学习和迭代，AI 可以更好地应对复杂的科学计算问题。

MindSpore Science 是基于 MindSpore 融合架构打造的高性能科学计算行业套件，提供业界领先的数据集、SOTA 模型、常用模型接口和前后处理工具。该套件面向 Ascend 深度优化，加速科学行业应用开发。

支持特性
--------

MindScience-Core 是 MindScience 的核心组件，为开发者提供易用性强、昇腾亲和的常用模型接口，加速 AI4Science 模型开发，主要提供以下几个能力：

常用模型接口：

- 图计算：由于图的拓扑结构天然适配网格、分子结构等数据形态，AI4Science 领域的模型大量采用以 GNN（Graph Neural Network，图神经网络）为核心的架构。MindScience 为此提供了一套功能完备、与 Ascend 平台深度亲和的 GNN 接口，同时保持与业界主流框架接口的兼容性，有效提升了相关模型的开发效率。

- 等变计算：等变计算库 e3nn 专为处理具有对称性的数据场景设计，尤其适配分子、晶体等包含旋转、平移等对称属性的结构。它提供了一套完整的等变神经网络构建工具，能精准捕捉数据的对称性特征，研究者可高效开发符合对称性原理的模型，显著提升分子模拟、材料科学等领域的任务表现。

- 物理信息神经网络：物理信息神经网络库 PINNs（Physics-Informed Neural Networks）专为融合物理规律与数据驱动建模设计，尤其适用于流体力学、热传导等偏微分方程（PDE）求解场景。MindScience 的 pde 模块提供了将物理守恒律、边界条件等约束嵌入神经网络的完整工具链，研究者无需依赖大量标注数据，即可高效构建符合物理规律的模型，大幅提升工程仿真、科学计算等领域的求解效率与泛化能力。

- 可微分求解器：可微分求解器是一类融合数值计算与自动微分能力的专用工具，核心用于高效求解常微分方程（ODE）、偏微分方程（PDE）等科学与工程问题。它能在输出数值解的同时，反向传播计算梯度信息，完美兼容 MindSpore 深度学习框架。借助这一特性，研究者可将物理方程的求解过程嵌入神经网络训练流程，为科学计算与深度学习的结合提供关键支撑，广泛应用于动力学模拟、控制策略优化等 AI4Science 场景。

- 科学计算算子：科学计算算子是支撑 AI4Science 模型构建的核心基础组件，算子经过与 Ascend 结合的深度优化，实现复杂科学计算任务的高效执行。

- 分布式并行：分布式并行是应对 AI4Science 大规模任务的核心高效计算方案，通过将海量数据、复杂模型或计算任务拆分到多节点、多设备集群中协同处理，突破单设备的算力与内存瓶颈，为 AI4Science 向更复杂、更精细的研究方向拓展提供了关键算力保障。

典型模型支持：
--------------

.. raw:: html

    <details>
    <summary><strong>MindChem</strong></summary>

.. list-table::
   :header-rows: 1

   * - 应用名称
     - 简介
     - 学习类型
   * - `DiffCSP <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/diffcsp>`__
     - 晶体结构生成
     - 扩散模型/生成式学习
   * - `NequIP <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/nequip>`__
     - 分子力场
     - 监督学习
   * - `Orb <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/orb>`__
     - 分子力场
     - 监督学习
   * - `MatFormer <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/matformer>`__
     - 材料性质预测
     - 监督学习
   * - `DeepHE3nn <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/deephe3nn>`__
     - 哈密顿量预测
     - 监督学习
   * - `CrystalFlow <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindChem/applications/crystalflow>`__
     - 晶体结构生成
     - 扩散模型/生成式学习

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><strong>MindEarth</strong></summary>

.. list-table::
   :header-rows: 1

   * - 应用名称
     - 简介
     - 学习类型
   * - `LeadFormer <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEarth/applications/leadformer>`__
     - 北极海冰冰间水道预测
     - 监督学习

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><strong>MindEnergy</strong></summary>

.. list-table::
   :header-rows: 1

   * - 应用名称
     - 简介
     - 学习类型
   * - `DAE-PINN <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEnergy/applications/dae_pinn>`__
     - 物理信息神经网络
     - 监督+物理约束
   * - `DeepONet-Grig-UQ <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEnergy/applications/deeponet_grig_uq>`__
     - 电网故障后轨迹预测
     - 监督学习
   * - `PowerFlowNet <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEnergy/applications/powerflownet>`__
     - 潮流计算
     - 监督学习

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><strong>MindFlow</strong></summary>

.. list-table::
   :header-rows: 1

   * - 应用名称
     - 简介
     - 学习类型
   * - `Acoustic <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/acoustic>`__
     - 声学模拟
     - 自动迭代学习
   * - `P2C2Net <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/p2c2net>`__
     - 物理建模求解二维Burgers方程组
     - 监督学习
   * - `FNO2D <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/fno2d>`__
     - 傅里叶神经算子求解二维Navier-Stokes方程组
     - 监督学习

.. raw:: html

    </details>

.. raw:: html

    <details>
    <summary><strong>MindSPONGE</strong></summary>

.. list-table::
   :header-rows: 1

   * - 应用名称
     - 简介
     - 学习类型
   * - `AlphaFold3 <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindSPONGE/applications/af3>`__
     - 蛋白质结构预测
     - 监督学习
   * - `ProteinMPNN <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindSPONGE/applications/proteinmpnn>`__
     - 蛋白质序列设计
     - 监督学习
   * - `RFdiffusion <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindSPONGE/applications/rfdiffusion>`__
     - 蛋白质扩散生成
     - 扩散模型/生成式学习
   * - `Protenix <https://atomgit.com/mindspore-lab/mindscience/blob/master/MindSPONGE/applications/protenix>`__
     - 蛋白质功能预测
     - 监督学习

.. raw:: html

    </details>

.. toctree::
   :maxdepth: 1
   :caption: 组件介绍
   :hidden:

   MindChem
   MindEarth
   MindEnergy
   MindFlow
   MindSPONGE

.. toctree::
   :maxdepth: 1
   :caption: 安装指南
   :hidden:

   install.md

.. toctree::
   :maxdepth: 1
   :caption: 快速开始
   :hidden:

   quick_start.md

.. toctree::
   :maxdepth: 1
   :caption: API参考
   :hidden:

   api/mindscience.common
   api/mindscience.data
   api/mindscience.diffuser
   api/mindscience.distributed
   api/mindscience.e3nn
   api/mindscience.gnn
   api/mindscience.models
   api/mindscience.pde
   api/mindscience.sciops
   api/mindscience.solvers
   api/mindscience.utils

.. toctree::
   :maxdepth: 1
   :caption: 贡献指南
   :hidden:

   CONTRIBUTING

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE

