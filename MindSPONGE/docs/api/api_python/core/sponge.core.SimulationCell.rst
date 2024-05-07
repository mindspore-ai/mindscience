sponge.core.SimulationCell
==============================

.. py:class:: sponge.core.SimulationCell(system: Molecule, potential: PotentialCell, bias: Bias = None, cutoff: float = None, neighbour_list: NeighbourList = None, wrapper: EnergyWrapper = None)

    用于仿真的单元，等同于 :class:`sponge.core.WithEnergyCell`。

    .. note::
        此单元将在未来的版本中被移除，请改用 :class:`sponge.core.WithEnergyCell`。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 仿真系统。
        - **potential** (:class:`sponge.potential.PotentialCell`) - 势能函数单元。
        - **bias** (`sponge.potential.Bias`, 可选) - 偏置势能函数单元。默认值： ``None``。
        - **cutoff** (float, 可选) - 邻居列表的截断距离。如果未指定，则将使用势能的截断值。默认值： ``None``。
        - **neighbour_list** (:class:`sponge.partition.NeighbourList`, 可选) - 邻居列表。默认值： ``None``。
        - **wrapper** (`sponge.sampling.wrapper.EnergyWrapper`, 可选) - 网络用于包装和处理势能及偏置。默认值： ``None``。