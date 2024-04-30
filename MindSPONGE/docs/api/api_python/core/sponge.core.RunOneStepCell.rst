sponge.core.RunOneStepCell
==============================

.. py:class:: sponge.core.RunOneStepCell(energy: :class:`sponge.energy.WithEnergyCell` = None, force: :class:`sponge.force.WithForceCell` = None, optimizer: :class:`mindspore.nn.optim.Optimizer` = None, steps: int = 1, sens: float = 1.0, **kwargs)

    运行一步模拟的神经网络层。这一层包裹了 `energy` ， `force` 和 `optimizer` 。在construct函数里将会生成一张反向图来更新仿真系统的原子坐标。

    参数：
        - **energy** ( :class:`sponge.energy.WithEnergyCell`) - 包含了有势能函数的模拟系统的神经网络层。默认值：``None``。该神经网络层用于计算并返回系统在当前坐标处的势能值。
        - **force** ( :class:`sponge.force.WithForceCell`) - 包含了有原子力函数的模拟系统的神经网络层。默认值：``None``。该神经网络层用于计算并返回系统在当前坐标处的力值。
        - **optimizer** ( :class:`mindspore.nn.optim.Optimizer`, 可选) - 模拟的优化器。默认值： ``None``。
        - **steps** (int, 可选) - 模拟的步数。默认值： ``1``。
        - **sens** (float, 可选) - 作为反向传播的输入要填充的缩放数。默认值： ``1.0``。
        - **kwargs** (dict) - 关键字参数。

    输入：
        - **\*inputs** (Tuple(Tensor)) - :class:`sponge.energy.WithEnergyCell` 的输入Tensors的tuple。

    输出：
        - **energy** (Tensor) - 能量。shape为 :math:`(B, 1)` 。数据类型为float。这里的B是batch size。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 。数据类型为float。这里的 :math:`B` 是batch size， :math:`A` 是原子数量， :math:`D` 是空间维度，通常为3。

    .. py:method:: bias()
        :property:

        整个偏置势的Tensor。

        返回：
            Tensor，shape为 `(B, 1)` ，数据类型为float。

    .. py:method:: bias_function()
        :property:

        偏置势函数的网络层。

        返回：
            Cell，偏置势函数。

    .. py:method:: bias_names()
        :property:

        偏置势能的名字。

        返回：
            list[str]，偏置势能的名字列表。

    .. py:method:: biases()
        :property:

        偏置势的组成部分的Tensor。

        返回：
            Tensor，shape为 `(B, V)` ，数据类型为float。

    .. py:method:: energies()
        :property:

        势能组成部分的Tensor。

        返回：
            Tensor，shape为 `(B, U)` ，数据类型为float。

    .. py:method:: energy_cutoff()
        :property:

        `WithEnergyCell` 中邻居列表的截断距离。

        返回：
            Tensor， `WithEnergyCell` 中邻居列表的截断距离。

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

    .. py:method:: force_cutoff()
        :property:

        `WithForceCell` 中邻居列表的截断距离。

        返回：
            Tensor， `WithForceCell` 中邻居列表的截断距离。

    .. py:method:: length_unit()
        :property:

        长度单位。

        返回：
            str，长度单位。

    .. py:method:: neighbour_list_pace()
        :property:

        更新邻居列表所需的step。

        返回：
            int，更新邻居列表所需的step数。

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

    .. py:method:: set_pbc_grad(value: bool)

        设定是否计算周期性边界条件箱的梯度。

        参数：
            - **value** (bool) - 用于判断是否计算周期性边界条件箱的梯度的标志符。

    .. py:method:: set_steps(steps: int)

        设置JIT的步数。

        参数：
            - **steps** (int) - JIT的步数。

    .. py:method:: update_bias(step: int)

        更新偏置势。

        参数：
            - **step** (int) - 更新偏置势的仿真step。

    .. py:method:: update_modifier(step: int)

        更新力修饰器。

        参数：
            - **step** (int) - 更新力修饰器的仿真step。

    .. py:method:: update_neighbour_list()

        更新邻居列表。

    .. py:method:: update_wrapper(step: int)

        更新能量包。

        参数：
            - **step** (int) - 更新能量包的仿真step。