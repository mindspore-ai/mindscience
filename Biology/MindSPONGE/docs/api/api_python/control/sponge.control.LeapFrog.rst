sponge.control.LeapFrog
===========================

.. py:class:: sponge.control.LeapFrog(system: :class:`sponge.system.Molecule`, thermostat: :class:`sponge.control.Thermostat` = None, barostat: :class:`sponge.control.Barostat` = None, constraint: :class:`sponge.control.Constraint` = None, **kwargs)

    基于middle scheme的蛙跳积分器，是 :class:`sponge.control.Integrator` 的一个子类。

    参考文献 Zhang, Z.; Yan, K; Liu, X.; Liu, J..
    A Leap-Frog Algorithm-based Efficient Unified Thermostat Scheme for Molecular Dynamics.
    Chinese Science Bulletin, 2018, 63(33).

    参数：
        - **system** ( :class:`sponge.system.Molecule`) - 模拟体系。
        - **thermostat** ( :class:`sponge.control.Thermostat`, 可选) - 用于温度耦合的恒温器。默认值： ``None``。
        - **barostat** ( :class:`sponge.control.Barostat`, 可选) - 用于压力耦合的气压调节器。默认值： ``None``。
        - **constraint** ( :class:`sponge.control.Constraint`, 可选) - 约束算法。默认值： ``None``。
        - **kwargs** - 关键字参数。

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