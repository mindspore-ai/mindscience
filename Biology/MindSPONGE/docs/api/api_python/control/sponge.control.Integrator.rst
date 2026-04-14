sponge.control.Integrator
=============================

.. py:class:: sponge.control.Integrator(system: :class:`sponge.system.Molecule`, thermostat: :class:`sponge.control.Thermostat` = None, barostat: :class:`sponge.control.Barostat` = None, constraint: Union[:class:`sponge.control.Constraint`, List[:class:`sponge.control.Constraint`]] = None, **kwargs)

    :class: `sponge.control.Integrator` 是在分子模拟的过程中，用以控制原子坐标和速度的模块。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 模拟体系。
        - **thermostat** ( :class:`sponge.control.Thermostat`, 可选) - 用于温度耦合的恒温器。默认值为 ``None``。
        - **barostat** (:class:`sponge.control.Barostat`, 可选) - 用于压力耦合的气压调节器。默认值为 ``None``。
        - **constraint** (Union[ :class:`sponge.control.Constraint`, List[ :class:`sponge.control.Constraint`]], 可选) - 约束算法。默认值为 ``None``。
        - **kwargs** (dict) - 关键字参数。

    输入：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。这里 :math:`B` 是分子模拟中walker的数目， :math:`A` 是原子数目， :math:`D` 是模拟系统的空间维数，通常为3。
        - **velocity** (Tensor) - 速度。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **energy** (Tensor) - 能量。shape为 :math:`(B, 1)` 的Tensor。数据类型是float。
        - **kinetics** (Tensor) - 动能。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **virial** (Tensor) - 维里应力。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **pbc_box** (Tensor) - 周期性边界条件盒子。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **step** (int) - 模拟步数。默认值： ``0``。

    输出：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **velocity** (Tensor) - 速度。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **energy** (Tensor) - 能量。shape为 :math:`(B, 1)` 的Tensor。数据类型是float。
        - **kinetics** (Tensor) - 动能。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **virial** (Tensor) - 维里。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **pbc_box** (Tensor) - 周期性边界条件盒子。shape为 :math:`(B, D)` 的Tensor。数据类型是float。

    .. py:method:: add_constraint(constraint)

        给积分器添加约束算法。

        参数：
            - **constraint** ( :class:`sponge.control.Constraint`) - 约束。

    .. py:method:: get_name()

        获取积分器的名称。

        返回：
            - str。 积分器的名称。

    .. py:method:: set_barostat(barostat)

        给积分器添加恒压器算法。

        参数：
            - **barostat** ( :class:`sponge.control.Barostat`) - 恒压器。

    .. py:method:: set_constraint(constraint: Union[:class:`sponge.control.Constraint`, List[:class:`sponge.control.Constraint`]], num_constraints: int = 0)

        给积分器设置约束算法。

        参数：
            - **constraint** (Union[:class:`sponge.control.Constraint`, List[:class:`sponge.control.Constraint`]]) - 约束。
            - **num_constraints** (int, 可选) - 约束的数目。默认值： ``0``。

    .. py:method:: set_degrees_of_freedom(dofs)

        设置自由度。

        参数：
            - **dofs** (int) - 自由度。

    .. py:method:: set_thermostat(thermostat)

        给积分器添加恒温器算法。

        参数：
            - **thermostat** (:class:`sponge.control.Thermostat`) - 恒温器。

    .. py:method:: set_time_step(dt)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步所需时间。