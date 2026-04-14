sponge.core.WithForceCell
==============================

.. py:class:: sponge.core.WithForceCell(self, system: Molecule, force: ForceCell, neighbour_list: NeighbourList = None, modifier: ForceModifier = None)

    用于封装带有原子力函数的仿真系统的单元。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 仿真系统。
        - **force** (`sponge.potential.ForceCell`) - 原子力计算单元。
        - **neighbour_list** (:class:`sponge.partition.NeighbourList`, 可选) - 邻居列表。默认值： ``None``。
        - **modifier** (`sponge.sampling.modifier.ForceModifier`, 可选) - 力修正器。默认值： ``None``。

    输入：
        - **energy** (Tensor) - 势能。shape为 :math:`(B, 1)` 。数据类型为float。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 。数据类型float。
        - **virial** (Tensor) - 维里。shape为 :math:`(B, D)` 。数据类型为float。

    输出：
        - **energy** (Tensor) - 仿真系统的总势能。shape为 :math:`(B, 1)` 。数据类型为float。这里的 :math:`B` 是batch size，即仿真中的walker的数量。
        - **force** (Tensor) - 作用于仿真系统每个原子的力。shape为 :math:`(B, A, D)` 。数据类型为float。这里的 :math:`B` 是batch size， :math:`A` 是原子数量， :math:`D` 是仿真系统的空间维度，通常为3。
        - **virial** (Tensor) - 维里。shape为 :math:`(B, D)` 。数据类型为float。

    .. py:method:: cutoff()
        :property:

        邻居列表的截断距离。

        返回：
            Tensor，截断距离。

    .. py:method:: energy_unit()
        :property:

        能量单位。

        返回：
            str，能量单位。

    .. py:method:: get_neighbour_list()

        获取邻居列表。

        返回：
            - neigh_idx，系统中每个原子邻近原子的目录。shape为 :math:`(B, A, N)` 的Tensor，数量类型为int。
            - neigh_mask，neigh_idx的掩码。shape为 :math:`(B, A, N)` 的Tensor，数量类型为bool。

    .. py:method:: length_unit()
        :property:

        长度单位。

        返回：
            str，长度单位。

    .. py:method:: neighbour_list_pace()
        :property:

        邻居列表的更新步长。

        返回：
            int，更新步长。

    .. py:method:: set_pbc_grad(grad_box: bool)

        设置是否计算PBC box的梯度。

        参数：
            - **grad_box** (bool) - 是否计算PBC box的梯度。

    .. py:method:: update_modifier(step: int)

        更新修饰器。

        参数：
            - **step** (int) - 当前仿真步数，当步数整除更新频率余数为0时，更新修饰。

    .. py:method:: update_neighbour_list()

        更新邻居列表。

        参数：
            - **coordinate** (Tensor) - 位置坐标。shape为 :math:`(B, A, D)` 的Tensor。这里的 :math:`B` 是batch size， :math:`A` 是原子数量， :math:`D` 是仿真系统的空间维度，通常为3。数据类型为float。
            - **pbc_box** (Tensor) - 周期性边界条件(PBC)盒子。shape为 :math:`(B, D)` 的Tensor。数据类型为浮点型。

        返回：
            - neigh_idx，系统中每个原子邻近原子的目录。shape为 :math:`(B, A, N)` 的Tensor，数量类型为int。
            - neigh_mask，neigh_idx的掩码。shape为 :math:`(B, A, N)` 的Tensor，数量类型为bool。
