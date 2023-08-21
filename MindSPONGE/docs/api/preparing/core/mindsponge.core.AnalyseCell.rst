mindsponge.core.AnalyseCell
===========================

.. py:class:: mindsponge.core.AnalyseCell(system: Molecule, potential: PotentialCell, neighbour_list: NeighbourList = None, calc_energy: bool = False, calc_forces: bool = False)

    分析的核心层。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **potential** (PotentialCell) - 势能。
        - **neighbour_list** (NeighbourList) - 邻居列表。默认值："None"。
        - **calc_energy** (bool) - 是否计算能量。默认值： ``False`` 。
        - **calc_forces** (bool) - 是否计算力。默认值： ``False`` 。

    输出：
        - Tensor。能量。
        - Tensor。力。
        - Tensor。坐标。
        - Tensor。PBC box。