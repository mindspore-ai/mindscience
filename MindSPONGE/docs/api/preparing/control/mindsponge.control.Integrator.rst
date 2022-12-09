mindsponge.control.Integrator
=============================

.. py:class:: mindsponge.control.Integrator(system, thermostat, barostat, constraint)

    模拟积分器。

    参数：
        - **system** (Molecule) - 模拟体系。
        - **thermostat** (Thermostat) - 用于温度耦合的恒温器。
        - **barostat** (Barostat) - 用于压力耦合的气压调节器。
        - **constraint** (Constraint) - 约束算法。

    .. py:method:: add_constraint(constraint)

        给积分器添加约束算法。

        参数：
            - **constraint** (Constraint) - 约束。

    .. py:method:: set_barostat(barostat)

        给积分器添加恒压器算法。

        参数：
            - **barostat** (Barostat) - 恒压器。

    .. py:method:: set_constraint(constraint)

        给积分器设置约束算法。

        参数：
            - **constraint** (Constraint) - 约束。

    .. py:method:: set_degrees_of_freedom(dofs)

        设置自由度。

        参数：
            - **dofs** (int) - 自由度。

    .. py:method:: set_thermostat(thermostat)

        给积分器添加恒温器算法。

        参数：
            - **thermostat** (Thermostat) - 恒温调节器。

    .. py:method:: set_time_step(dt)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步时间所需时间。