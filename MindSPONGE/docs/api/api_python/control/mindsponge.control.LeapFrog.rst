mindsponge.control.LeapFrog
===========================

.. py:class:: mindsponge.control.LeapFrog(system, thermostat, barostat, constraint)

    基于"middle scheme"的蛙跳积分器。

    参考文献：
        Zhang, Z.; Yan, K; Liu, X.; Liu, J..
        A Leap-Frog Algorithm-based Efficient Unified Thermostat Scheme for Molecular Dynamics [J].
        Chinese Science Bulletin, 2018, 63(33): 3467-3483.

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