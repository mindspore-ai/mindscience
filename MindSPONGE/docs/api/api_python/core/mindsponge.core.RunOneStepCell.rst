mindsponge.core.RunOneStepCell
==============================

.. py:class:: mindsponge.core.RunOneStepCell(energy: WithEnergyCell = None, force: WithForceCell = None, optimizer: Optimizer = None, steps: int = 1, sens: float = 1.0,)

    运行一步模拟的神经网络层。这一层包裹了 `energy` ， `force` 和 `optimizer` 。在construct函数里将会生成一张反向图来更新仿真系统的原子坐标。

    参数：
        - **energy** (WithEnergyCell) - 包含了有势能函数的模拟系统的神经网络层。默认值："None"。该神经网络层用于计算并返回系统在当前坐标处的势能值。
        - **force** (WithForceCell) - 包含了有原子力函数的模拟系统的神经网络层。默认值："None"。该神经网络层用于计算并返回系统在当前坐标处的力值。
        - **optimizer** (Optimizer) - 模拟的优化器。默认值："None"。
        - **steps** (int) - 模拟的步数。默认值：1.0。
        - **sens** (float) - 作为反向传播的输入要填充的缩放数。默认值：1.0。

    输入：
        - **\*inputs** (Tuple(Tensor)) - `WithEnergyCell` 的输入Tensors的tuple。

    输出：
        - 整体的势能，shape为 `(B, 1)` 的Tensor，数据类型为float。
        - 原子力，shape为 `(B, A, D)` 的Tensor，数据类型为float。

    .. py:method:: neighbour_list_pace()

        更新邻居列表所需的step。

        返回：
            int，更新邻居列表所需的step数。

    .. py:method:: energy_cutoff()

        `WithEnergyCell` 中邻居列表的截断距离。

        返回：
            Tensor， `WithEnergyCell` 中邻居列表的截断距离。

    .. py:method:: force_cutoff()

        `WithForceCell` 中邻居列表的截断距离。

        返回：
            Tensor， `WithForceCell` 中邻居列表的截断距离。

    .. py:method:: length_unit()

        长度单位。

        返回：
            str，长度单位。

    .. py:method:: energy_unit()

        能量单位。

        返回：
            str，能量单位。

    .. py:method:: num_energies()

        能量项 :math:`U` 的数量。

        返回：
            int，能量项的数量。

    .. py:method:: energy_names()

        能量项的名字。

        返回：
            list[str]，能量项的名字列表。

    .. py:method:: bias_names()

        偏置势能的名字。

        返回：
            list[str]，偏置势能的名字列表。

    .. py:method:: num_biases()

        偏置势能 :math:`V` 的数量。

        返回：
            int，偏置势能的数量。

    .. py:method:: energies()

        势能组成部分的Tensor。

        返回：
            Tensor，shape为 `(B, U)` ，数据类型为float。

    .. py:method:: biases()

        偏置势的组成部分的Tensor。

        返回：
            Tensor，shape为 `(B, V)` ，数据类型为float。

    .. py:method:: bias()

        整个偏置势的Tensor。

        返回：
            Tensor，shape为 `(B, 1)` ，数据类型为float。

    .. py:method:: bias_function()

        偏置势函数的网络层。

        返回：
            Cell，偏置势函数。

    .. py:method:: update_neighbour_list()

        更新邻居列表。

    .. py:method:: update_bias(step)

        更新偏置势。

        参数：
            - **step** (int) - 更新偏置势的仿真step。

    .. py:method:: update_wrapper(step)

        更新能量包。

        参数：
            - **step** (int) - 更新能量包的仿真step。

    .. py:method:: update_modifier(step)

        更新力修饰器。

        参数：
            - **step** (int) - 更新力修饰器的仿真step。

    .. py:method:: set_pbc_grad(value)

        设定是否计算周期性边界条件箱的梯度。

        参数：
            - **value** (bool) - 用于判断是否计算周期性边界条件箱的梯度的标志符。

    .. py:method:: set_steps(step)

        设置JIT的步数。

        参数：
            - **steps** (int) - JIT的步数。

    .. py:method:: run_one_step(*inputs)

        运行单步模拟。

        参数：
            - **/*inputs** (Tuple(Tensor)) - `WithEnergyCell` 的输入Tensors的tuple。

        返回：
          - 整体的势能，shape为 `(B, 1)` 的Tensor，数据类型为float。
          - 原子力，shape为 `(B, A, D)` 的Tensor，数据类型为float。