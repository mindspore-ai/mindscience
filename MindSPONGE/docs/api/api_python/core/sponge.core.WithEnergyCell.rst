sponge.core.WithEnergyCell
==============================

.. py:class:: sponge.core.WithEnergyCell(system: Molecule, potential: PotentialCell, bias: Union[Bias, List[Bias]] = None, cutoff: float = None, neighbour_list: NeighbourList = None, wrapper: EnergyWrapper = None, **kwargs)

    用势能函数封装仿真系统的神经网络层。
    该神经网络层用于计算并返回系统在当前坐标处的势能值。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 仿真系统。
        - **potential** (:class:`sponge.potential.PotentialCell`) - 势能函数层。
        - **bias** (Union[`sponge.potential.Bias`, List[`sponge.potential.Bias`]], 可选) - 偏置势函数层。默认值： ``None``。
        - **cutoff** (float, 可选) - 邻居列表的截断距离。如果为 ``None``，则将其赋值为势能的截止值。默认值： ``None``。
        - **neighbour_list** (:class:`sponge.partition.NeighbourList`, 可选) - 邻居列表。默认值： ``None``。
        - **wrapper** (`sponge.sampling.wrapper.EnergyWrapper`, 可选) - 包裹和处理势能和偏置势的网络。默认值： ``None``。
        - **kwargs** (dict) - 关键字参数。

    输入：
        - **\*inputs** (Tuple(Tensor)) - :class:`sponge.core.WithEnergyCell` 的输入Tensor tuple。

    输出：
        - **energy** (Tensor) - 整个系统的势能。shape为 :math:`(B, 1)` 。数据类型为float。

    .. py:method:: bias()
        :property:

        整体偏置势的Tensor。

        返回：
            Tensor，shape为 :math:`(B, 1)` ，数据类型为float。

    .. py:method:: bias_names()
        :property:

        偏置势能的名字。

        返回：
            list[str]，偏置势能的名字列表。

    .. py:method:: bias_pace(index: int = 0)

        偏置势的更新频率。

        参数：
            - **index** (int) - 偏置势的目录。默认值：0。

        返回：
            int，更新频率。

    .. py:method:: biases()
        :property:

        偏置势分量的Tensor。

        返回：
            偏置势分量的Tensor。shape为 :math:`(B, V)` ，数据类型为float。

    .. py:method:: calc_biases()

        计算偏置势项。

        返回：
            偏置势项，shape为 :math:`(B, V)` 的Tensor。数据类型为float。

    .. py:method:: calc_energies()

        计算势能的能量项。

        返回：
            能量项，shape为 :math:`(B, U)` 的Tensor。数据类型为float。

    .. py:method:: cutoff()
        :property:

        邻居列表的截断距离。

        返回：
            Tensor，截断距离。

    .. py:method:: energies()
        :property:

        势能分量的Tensor。

        返回：
            势能分量的Tensor，shape为 :math:`(B, U)` ，数据类型为float。

    .. py:method:: energy_names()
        :property:

        能量项的名字。

        返回：
            list[str]，能量项的名字列表。

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

    .. py:method:: num_biases()
        :property:

        偏置势能 :math:`V` 的数量。

        返回：
            int，偏置势能的数量。

    .. py:method:: num_energies()
        :property:

        能量项 :math:`U` 的数量。

        返回：
            int，能量项的数量。

    .. py:method:: set_pbc_grad(grad_box: bool)

        设置是否计算PBC box的梯度。

        参数：
            - **grad_box** (bool) - 是否计算PBC box的梯度。

    .. py:method:: update_bias(step: int)

        更新偏置势。

        参数：
            - **step** (int) - 当前仿真步数，当步数整除更新频率余数为0时，更新偏置势。

    .. py:method:: update_neighbour_list()

        更新邻居列表。

        返回：
            - neigh_idx，系统中每个原子邻近原子的目录。shape为 :math:`(B, A, N)` 的Tensor，数量类型为int。
            - neigh_mask，neigh_idx的掩码。shape为 :math:`(B, A, N)` 的Tensor，数量类型为bool。

    .. py:method:: update_wrapper(step: int)

        更新能量包装器。

        参数：
            - **step** (int) - 当前仿真步数，当步数整除更新频率余数为0时，更新能量包装器。
