mindsponge.control.Thermostat
=============================

.. py:class:: mindsponge.control.Thermostat(system, temperature=300, control_step=1, time_constant=4)

    温度耦合的恒温控制器。

    参数：
        - **system** (Molecule) - 模拟体系。
        - **thermostat** (Thermostat) - 用于温度耦合的恒温器。
        - **barostat** (Barostat) - 用于压力耦合的气压调节器。
        - **constraint** (Constraint) - 约束算法。

    输出：
        - Tensor。坐标，shape(B, A, D)。
        - Tensor。速度，shape(B, A, D)。
        - Tensor。力，shape(B, A, D)。
        - Tensor。能量，shape(B, 1)。
        - Tensor。动力学，shape(B, D)。
        - Tensor。维里，shape(B, D)。
        - Tensor。周期性边界条件box，shape(B, D)。

    .. py:method:: kinetics()

        参考动力学。

        返回：
            Tensor。参考动力学。

    .. py:method:: set_degrees_of_freedom(dofs)

        设置自由度。

        参数：
            - **dofs** (int) - 自由度。

    .. py:method:: temperature()

        参考温度。

        返回：
            Tensor。参考温度。

    .. py:method:: velocity_scale(sim_kinetics, ref_kinetics, ratio=1)

        计算温度耦合的速度范围因子。

        参数：
            - **sim_kinetics** (Tensor) - 模拟动力学。
            - **ref_kinetics** (Tensor) - 参考动力学。
            - **ratio** (float) - lambda的度。默认值：1。

        返回：
            Tensor。速度范围因子。