sponge.core.WithEnergyCell
==============================

.. py:class:: sponge.core.WithEnergyCell(system: Molecule, potential: PotentialCell, bias: Union[Bias, List[Bias]] = None, cutoff: float = None, neighbour_list: NeighbourList = None, wrapper: EnergyWrapper = None)

    用势能函数封装仿真系统的神经网络层。
    该神经网络层用于计算并返回系统在当前坐标处的势能值。

    参数:
        - **system** (Molecule) - 仿真系统。
        - **potential** (PotentialCell) - 势能函数层。
        - **bias** (Union[Bias, List[Bias]]) - 偏差势能函数层。默认值："None"。
        - **cutoff** (float) - 邻居列表的截断距离。如果为None，则将其赋值为势能的截止值。默认值："None"。
        - **neighbour_list** (NeighbourList) - 邻居列表。默认值："None"。
        - **wrapper** (EnergyWrapper) - 包裹和处理势和偏差的网络。默认值："None"。

    输入:
        - **\*inputs** (Tuple(Tensor)) - 'WithEnergyCell'的输入Tensor对。

    输出:
        整个系统的势能, shape为 `(B, 1)` 的Tensor。数据类型为float。

    .. py:method:: cutoff()

        邻居列表的截断距离。

        返回：
            Tensor，截断距离。

    .. py:method:: neighbour_list_pace()

        邻居列表的更新步长。

        返回：
            int，更新步长。

    .. py:method:: length_unit()

        长度单位。

        返回：
            str，长度单位。

    .. py:method:: energy_unit()

        能量单位，

        返回：
            str，能量单位

    .. py:method:: num_energies()

        能量项 :math:`U` 的数量。

        返回：
            int，能量项的数量。

    .. py:method:: num_biases()

        偏置势能 :math:`V` 的数量。

        返回：
            int，偏置势能的数量。

    .. py:method:: energy_names()

        能量项的名字。

        返回：
            list[str]，能量项的名字列表。

    .. py:method:: bias_names()

        偏置势能的名字。

        返回：
            list[str]，偏置势能的名字列表。

    .. py:method:: energies()

        势能分量的Tensor。

        返回：
            势能分量的Tensor，shape为 `(B, U)` ，数据类型为float。

    .. py:method:: biases()

        偏置势分量的Tensor。

        返回：
            偏置势分量的Tensor。shape为  `(B, V)` ，数据类型为float。

    .. py:method:: bias()

        整体偏置势的Tensor。

        返回：
            Tensor，shape为 `(B, 1)` ，数据类型为float。

    .. py:method:: bias_pace(index=0)

        偏置势的更新频率。

        参数：
            - **index** (int) - 偏置势的目录。默认值：0。

        返回：
            int，更新频率。

    .. py:method:: set_pbc_grad(grad_box)

        设置是否计算PBC box的梯度。

        参数：
            - **grad_box** (bool) - 是否计算PBC box的梯度。

    .. py:method:: update_neighbour_list()

        更新邻居列表。

        返回：
            - neigh_idx，系统中每个原子邻近原子的目录。shape为 `(B, A, N)` 的Tensor，数量类型为int。
            - neigh_mask，neigh_idx的掩码。shape为 `(B, A, N)` 的Tensor，数量类型为bool。

    .. py:method:: update_bias(step)

        更新偏置势。

        参数：
            - **step** (int) - 当前仿真步数，当步数整除更新频率余数为0时，更新偏置势。

    .. py:method:: update_wrapper(step)

        更新能量包装器。

        参数：
            - **step** (int) - 当前仿真步数，当步数整除更新频率余数为0时，更新能量包装器。

    .. py:method:: get_neighbour_list()

        获取邻居列表。

        返回：
            - neigh_idx，系统中每个原子邻近原子的目录。shape为 `(B, A, N)` 的Tensor，数量类型为int。
            - neigh_mask，neigh_idx的掩码。shape为 `(B, A, N)` 的Tensor，数量类型为bool。

    .. py:method:: calc_energies()

        计算势能的能量项。

        返回：
            能量项，shape为 `(B, U)` 的Tensor。数据类型为float。

    .. py:method:: calc_biases()

        计算偏置势项。

        返回：
            偏置势项，shape为 `(B, V)` 的Tensor。数据类型为float。