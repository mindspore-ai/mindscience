mindsponge.control.LeapFrog
===========================

.. py:class:: mindsponge.control.LeapFrog(system, thermostat=None, barostat=None, constraint=None)

    基于"middle scheme"的蛙跳积分器。

    参考文献：
        `Zhang, Z.; Yan, K; Liu, X.; Liu, J..
        A Leap-Frog Algorithm-based Efficient Unified Thermostat Scheme for Molecular Dynamics [J].
        Chinese Science Bulletin, 2018, 63(33): 3467-3483.
        <https://www.sciengine.com/CSB/doi/10.1360/N972018-00908;JSESSIONID=ef65e0fb-a95f-4ba0-be14-a10b68b08aff>`_。

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