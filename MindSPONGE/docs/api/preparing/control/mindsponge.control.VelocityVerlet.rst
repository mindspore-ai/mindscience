mindsponge.control.VelocityVerlet
=================================

.. py:class:: mindsponge.control.VelocityVerlet(system, thermostat, barostat, constraint)

    基于"middle scheme"的温度verlet积分器。

    参考文献：
        `Zhang, Z.; Liu, X.; Chen, Z.; Zheng, H.; Yan, K.; Liu, J.
        A Unified Thermostat Scheme for Efficient Configurational Sampling for
        Classical/Quantum Canonical Ensembles via Molecular Dynamics [J].
        The Journal of Chemical Physics, 2017, 147(3): 034109.
        <https://aip.scitation.org/doi/abs/10.1063/1.4991621>`_。

    参数：
        - **system** (Molecule) - 模拟体系。
        - **thermostat** (Thermostat) - 用于温度耦合的恒温器。默认值："None"。
        - **barostat** (Barostat) - 用于压力耦合的气压调节器。默认值："None"。
        - **constraint** (Constraint) - 约束算法。默认值："None"。

    输出：
        - Tensor。坐标，shape(B, A, D)，数据类型为float。
        - Tensor。速度，shape(B, A, D)，数据类型为float。
        - Tensor。力，shape(B, A, D)，数据类型为float。
        - Tensor。能量，shape(B, 1)，数据类型为float。
        - Tensor。动力学，shape(B, D)，数据类型为float。
        - Tensor。维里，shape(B, D)，数据类型为float。
        - Tensor。周期性边界条件box，shape(B, D)，数据类型为float。

    .. py:method:: set_velocity_half(velocity_half, success=True)

        在前半步设置速度。

        参数：
            - **velocity_half** (Tensor) - 前半步的速度。
            - **success** (bool) - 是否速度被成功设定。默认值： ``True`` 。