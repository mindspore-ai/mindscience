sponge.control.AndersenBarostat
====================================

.. py:class:: sponge.control.AndersenBarostat(system: :class:`sponge.system.Molecule`, pressure: float = 1, anisotropic: bool = False, control_step: int = 1, compressibility: float = 4.6e-5, time_constant: float = 1., **kwargs)

    Andersen（弱耦合）气压调节器，是 :class:`sponge.control.Barostat` 的子类。

    参考文献 Andersen, Hans Christian.,
    Molecular dynamics simulations at constant pressure and/or temperature.,
    Journal of Chemical Physics, 1980.

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 模拟系统。
        - **pressure** (float, 可选) - 压力耦合参考压力 :math:`P_ref` 。 单位是 :math:`bar` 。默认值： ``1.0``。
        - **anisotropic** (bool, 可选) - 是否执行各向异性压力控制。默认值： ``False`` 。
        - **control_step** (int, 可选) - 控制器执行的步骤间隔。默认值： ``1``。
        - **compressibility** (float, 可选) - 等温压缩率 :math:`\beta` 。单位是 :math:`bar^{-1}` 。默认值： ``4.6e-5``。
        - **time_constant** (float, 可选) - 压力耦合的时间常数 :math:`\tau_p` 。单位是皮秒。默认值： ``1.0``。
        - **kwargs** (dict) - 关键字参数。

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