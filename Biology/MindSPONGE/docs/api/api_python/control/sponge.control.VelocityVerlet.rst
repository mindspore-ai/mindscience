sponge.control.VelocityVerlet
=================================

.. py:class:: sponge.control.VelocityVerlet(system: :class:`sponge.system.Molecule`, thermostat: :class:`sponge.control.Thermostat` = None, barostat: :class:`sponge.control.Barostat` = None, constraint: :class:`sponge.control.Constraint` = None, **kwargs)

    基于middle scheme的速度verlet积分器。

    参考文献 Zhang, Z.; Liu, X.; Chen, Z.; Zheng, H.; Yan, K.; Liu, J.
    A Unified Thermostat Scheme for Efficient Configurational Sampling for
    Classical/Quantum Canonical Ensembles via Molecular Dynamics.
    The Journal of Chemical Physics, 2017, 147(3).

    参数：
        - **system** ( :class:`sponge.system.Molecule`) - 模拟体系。
        - **thermostat** ( :class:`sponge.control.Thermostat`, 可选) - 用于温度耦合的恒温器。默认值： ``None``。
        - **barostat** ( :class:`sponge.control.Barostat`, 可选) - 用于压力耦合的气压调节器。默认值： ``None``。
        - **constraint** ( :class:`sponge.control.Constraint`, 可选) - 约束算法。默认值： ``None``。

    输入：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, , D)` 的Tensor。数据类型是float。这里 :math:`B` 是分子模拟中walker的数目， :math:`A` 是原子数目， :math:`D` 是模拟系统的空间维数，通常为3。
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

    .. py:method:: set_velocity_half(velocity_half)

        在前半步设置速度。

        参数：
            - **velocity_half** (Tensor) - 前半步的速度。
        
        返回：
            bool。是否成功设置速度。
