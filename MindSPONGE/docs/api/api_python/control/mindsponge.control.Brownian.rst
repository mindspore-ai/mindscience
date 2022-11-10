mindsponge.control.Brownian
===========================

.. py:class:: mindsponge.control.Brownian(system, temperature=300.0, friction_coefficient=1e3)

    布朗积分器。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **temperature** (float) - 模拟温度 T，单位K。默认值：300.0。
        - **friction_coefficient** (float) - 摩擦系数g，单位(amu/ps)。默认值：1e3。

    输出：
        - Tensor。坐标，shape(B, A, D)，数据类型为float。
        - Tensor。速度，shape(B, A, D)，数据类型为float。
        - Tensor。力，shape(B, A, D)，数据类型为float。
        - Tensor。能量，shape(B, 1)，数据类型为float。
        - Tensor。动力学，shape(B, D)，数据类型为float。
        - Tensor。维里，shape(B, D)，数据类型为float。
        - Tensor。周期性边界条件box，shape(B, D)，数据类型为float。

    .. py:method:: set_thermostat(thermostat)

        给积分器设置恒温器算法。

        参数：
            - **thermostat** (None) - 设置恒温器算法。

    .. py:method:: set_barostat(barostat)

        给积分器设置恒压器算法。

        参数：
            - **barostat** (None) - 设置恒压器算法。

    .. py:method:: set_constraint(constraint)

        给积分器设置约束算法。

        参数：
            - **constraint** (None) - 设置约束算法。

    .. py:method:: set_time_step(dt)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步时间所需时间。