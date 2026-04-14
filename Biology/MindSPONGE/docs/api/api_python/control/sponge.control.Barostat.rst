sponge.control.Barostat
===========================

.. py:class:: sponge.control.Barostat(system: :class:`sponge.system.Molecule`, pressure: float = 1, anisotropic: bool = False, control_step: int = 1, compressibility: float = 4.6e-5, time_constant: float = 1., **kwargs)

    MindSPONGE中压力耦合器的气压调节器的基类，是 :class:`sponge.controller.Controller` 的基类。

    :class:`sponge.controller.Barostat` 模块用于压力耦合，它的功能是在模拟过程中控制原子的坐标和周期性边界条件盒子（PBC box）的大小。

    参数：
        - **system** (:class: `sponge.system.Molecule`) - 模拟系统。
        - **pressure** (float, 可选) - 压力耦合参考压力 :math:`P_ref` 。单位是 :math:`bar` 。默认值： ``1.0``。
        - **anisotropic** (bool, 可选) - 是否执行各向异性压力控制。默认值： ``False`` 。
        - **control_step** (int, 可选) - 控制器执行的步骤间隔。默认值： ``1``。
        - **compressibility** (float, 可选) - 等温压缩率 :math:`\beta` 。单位是 :math:`bar^{-1}` 。默认值： ``4.6e-5``。
        - **time_constant** (float, 可选) - 压力耦合的时间常数。默认值： ``1.0``。
        - **kwargs** (dict) - 关键字参数。

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

    .. py:method:: compressibility()

        等温压缩率。

        返回：
            Tensor，等温压缩率。

    .. py:method:: pressure()
        :property:

        参考压力。

        返回：
            Tensor。参考压力。
    
    .. py:method:: pressure_scale(sim_press: Tensor, ref_press: Tensor, ratio: float=1.0)

        计算压力耦合器的坐标缩放因子。

        参数：
            - **sim_press** (Tensor) - 模拟压力。
            - **ref_press** (Tensor) - 参考压力。
            - **ratio** (float) - 压力耦合的比率。默认值： ``1.0``。

        返回：
            Tensor，压力耦合的坐标缩放因子。

    .. py:method:: reconstruct_pressure(pressure: Union[float, ndarray, Tensor, List[float]])

        重置参考压力。

        参数：
            - **pressure** (Union[float, ndarray, Tensor, List[float]]) - 压力。

        返回：
            :class:`sponge.control.Barostat`。当前的压力耦合器。
    
    .. py:method:: set_pressure(pressure: Union[float, ndarray, Tensor, List[float]])
    
        设置参考压力。
        参考压力数组的形状必须与当前压力数组的形状相同。
    
        参数：
            - **pressure** (Union[float, ndarray, Tensor, List[float]]) - 参考压力。
    
        返回：
            Tensor，参考压力。