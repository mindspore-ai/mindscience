sponge.core.AnalysisCell
===========================

.. py:class:: sponge.core.AnalysisCell(system: Molecule, potential: PotentialCell, neighbour_list: NeighbourList = None)

    用于分析的模块。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 模拟系统。
        - **potential** (:class:`sponge.potential.PotentialCell`) - 势能。
        - **neighbour_list** (:class:`sponge.partition.NeighbourList`, 可选) - 邻居列表。默认: ``None``。

    输入：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, A, D)` 。数据类型为 float。这里的 :math:`B` 是batch size， :math:`A` 是原子数量，而 :math:`D` 是模拟系统的空间维度，通常为3。
        - **pbc_box** (Tensor) - PBC边界条件。shape为 :math:`(B, D)` 。数据类型为 float。
        - **energy** (Tensor) - 能量。shape为 :math:`(B, 1)` 。数据类型为 float。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 。数据类型为 float。
        - **potentials** (Tensor, 可选) - 力场来源的原始势能。shape为 :math:`(B, U)` 。这里的 :math:`U` 是势能的数量。数据类型为 float。默认： ``0``。
        - **total_bias** (Tensor, 可选) - 用于重加权的总偏置势。shape为 :math:`(B, 1)` 。数据类型为 float。默认： ``0``。
        - **biases** (Tensor, 可选) - 初始偏置势。shape为 :math:`(B, V)` 。这里的 :math:`V` 是偏置势的数量。数据类型为 float。默认： ``0``。

    输出：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, A, D)` 。数据类型为 float。
        - **pbc_box** (Tensor) - PBC边界条件。shape为 :math:`(B, D)` 。数据类型为 float。
        - **energy** (Tensor) - 模拟系统的总势能。shape为 :math:`(B, 1)` 。数据类型为 float。
        - **force** (Tensor) - 模拟系统中每个原子的力。shape为 :math:`(B, A, D)` 。数据类型为 float。
        - **potentials** (Tensor) - 来自力场的势能。shape为 :math:`(B, U)` 。数据类型为 float。
        - **total_bias** (Tensor) - 用于重加权的总偏置势。shape为 :math:`(B, 1)` 。数据类型为 float。
        - **biases** (Tensor) - 偏置势。shape为 :math:`(B, V)` 。数据类型为 float。
