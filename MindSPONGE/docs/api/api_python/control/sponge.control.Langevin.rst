sponge.control.Langevin
===========================

.. py:class:: sponge.control.Langevin(system: :class:`sponge.system.Molecule`, temperature: float = 300.0, control_step=1, time_constant=0.5, seed=0, seed2=0, **kwargs)

    Langevin恒温控制器。

    参考文献 Goga, N.; Rzepiela, A. J.; de Vries, A. H.; Marrink, S. J.; Berendsen, H. J. C..
    Efficient Algorithms for Langevin and DPD Dynamics.
    Journal of Chemical Theory and Computation, 2012, 8(10)

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 模拟系统。
        - **temperature** (float, 可选) - 温度耦合的参考温度 :math:`T_ref` ，单位为开尔文。默认值： ``300.0``。
        - **control_step** (int, 可选) - 控制器执行的步骤间隔。默认值： ``1``。
        - **time_constant** (float, 可选) - 温度耦合的时间常数 :math:`\tau_T` 。单位为皮秒。默认值： ``0.2``。
        - **seed** (int, 可选) - 标准常态的随机种子。默认值： ``0``。
        - **seed2** (int, 可选) - 标准常态的随机种子2。默认值： ``0``。
        - **kwargs** - 关键字参数。

    输入：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。这里 :math:`B` 是分子模拟中walker的数目， :math:`A` 是原子数目， :math:`D` 是模拟系统的空间维数，通常为3。
        - **velocity** (Tensor) - 速度。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **energy** (Tensor) - 能量。shape为 :math:`(B, 1)` 的Tensor。数据类型是float。
        - **kinetics** (Tensor) - 动能。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **virial** (Tensor) - 维里应力。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
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

    .. py:method:: set_time_step(dt)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步时间所需时间。