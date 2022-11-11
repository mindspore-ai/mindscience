mindsponge.control.Thermostat
=============================

.. py:class:: mindsponge.control.Thermostat(system, temperature=300.0, control_step=1, time_constant=4.0)

    温度耦合的恒温控制器。

    参数：
        - **system** (Molecule) - 模拟体系。
        - **temperature** (float) - 温度耦合参考温度P_ref (bar)。单位为k。默认值：300.0。
        - **control_step** (int) - 控制器执行的步骤间隔。默认值：1。
        - **time_constant** (float) - 温度耦合的时间常数 \tau_T。单位为ps。默认值：4.0。

    输出：
        - Tensor。坐标，shape(B, A, D)，数据类型为float。
        - Tensor。速度，shape(B, A, D)，数据类型为float。
        - Tensor。力，shape(B, A, D)，数据类型为float。
        - Tensor。能量，shape(B, 1)，数据类型为float。
        - Tensor。动力学，shape(B, D)，数据类型为float。
        - Tensor。维里，shape(B, D)，数据类型为float。
        - Tensor。周期性边界条件box，shape(B, D)，数据类型为float。

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

    .. py:method:: velocity_scale(sim_kinetics, ref_kinetics, ratio=1.0)

        计算温度耦合的速度范围因子。

        参数：
            - **sim_kinetics** (Tensor) - 模拟动力学。
            - **ref_kinetics** (Tensor) - 参考动力学。
            - **ratio** (float) - lambda的度。默认值：1.0。

        返回：
            Tensor。速度范围因子。