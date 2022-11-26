mindsponge.core.SimulationCell
==============================

.. py:class:: mindsponge.core.SimulationCell(system: Molecule, potential: PotentialCell, cutoff: float = None, neighbour_list: NeighbourList = None, wrapper: EnergyWrapper = "sum", bias: Bias = None)

    模拟的核心层。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **potential** (PotentialCell) - 势能。
        - **cutoff** (float) - 中止距离。默认值："None"。
        - **neighbour_list** (NeighbourList) - 邻居列表。默认值："None"。
        - **wrapper** (EnergyWrapper) - 网络来包装和处理电位和偏置。默认值："sum"。
        - **bias** (Bias) - 偏置势能。默认值："None"。

    .. py:method:: get_neighbour_list()

        获得邻居列表。

        返回：
            - Tensor。邻居的索引。
            - Tensor。邻居的mask。

    .. py:method:: set_pbc_grad(grad_box: bool)

        设置是否计算PBC box的梯度。

        参数：
            - **grad_box** (bool) - 是否计算PBC box的梯度。

    .. py:method:: update_neighbour_list()

        更新邻居列表。

        返回：
            Tensor。邻居列表。