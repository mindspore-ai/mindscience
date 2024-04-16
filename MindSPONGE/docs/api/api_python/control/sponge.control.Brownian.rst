sponge.control.Brownian
===========================

.. py:class:: sponge.control.Brownian(system: :class:`sponge.system.Molecule`, temperature: float = 300, friction_coefficient: float = 1e3, **kwargs)

    布朗积分器。

    参数：
        - **system** ( :class:`sponge.system.Molecule`) - 模拟系统。
        - **temperature** (float, 可选) - 模拟温度，单位K。默认值： ``300.0``。
        - **friction_coefficient** (float, 可选) - 摩擦系数，单位(amu/ps)。默认值： ``1e3``。
        - **kwargs** (dict)- 关键字参数。

    输入：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。这里 :math:`B` 是分子模拟中walker的数目， :math:`A` 是原子数目， :math:`D` 是模拟系统的空间维数，通常为3。
        - **velocity** (Tensor) - 速度。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **energy** (Tensor) - 能量。shape为 :math:`(B, 1)` 的Tensor。数据类型是float。
        - **kinetics** (Tensor) - 动能。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **virial** (Tensor) - 维里。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
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

    .. py:method:: set_thermostat(thermostat: None = None)

        给积分器设置恒温器算法。

        参数：
            - **thermostat** (None) - 设置恒温器算法，:class:`sponge.control.Brownian` 积分器中恒温器算法只能是 ``None``。默认值： ``None``。

    .. py:method:: set_barostat(barostat: None = None)

        给积分器设置恒压器算法。

        参数：
            - **barostat** (None) - 设置恒压器算法，:class:`sponge.control.Brownian` 积分器中恒压器算法只能是 ``None``。默认值： ``None``。

    .. py:method:: set_constraint(constraint: None = None, num_constraints: int = 0)

        给积分器设置约束算法。

        参数：
            - **constraint** (None) - 设置约束算法， :class:`sponge.control.Brownian`积分器中约束算法只能是 ``None``。默认值： ``None``。
            - **num_constraints** (int) - 约束数目。默认值： ``0``。

    .. py:method:: set_time_step(dt)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步所需时间。

    .. py:method:: temperature()
        :property:

        返回模拟温度。

        返回：
            - Tensor，模拟温度。