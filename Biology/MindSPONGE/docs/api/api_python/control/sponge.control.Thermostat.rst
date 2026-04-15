sponge.control.Thermostat
=============================

.. py:class:: sponge.control.Thermostat(system: :class:`sponge.system.Molecule`, temperature: Union[float, ndarray, Tensor, List[float]] = 300, control_step: int = 1, time_constant: float = 0.5, **kwargs)

    恒温控制器模块的基类，也是 :class:`sponge.control.Controller` 的子类。

    :class:`sponge.control.Thermostat` 模块用于温度耦合，其主要功能是在模拟过程中，控制原子速度和系统的动力学。

    参数：
        - **system** ( :class:`sponge.system.Molecule`) - 模拟体系。
        - **temperature** (Union[float, ndarray, Tensor], 可选) - 温度耦合的参考温度 :math:`T_{ref}` 。单位为开尔文。默认值： ``300.0``。
        - **control_step** (int, 可选) - 控制器执行的步骤间隔。默认值： ``1``。
        - **time_constant** (float, 可选) - 温度耦合的时间常数 :math:`\tau_T` 。单位为皮秒。默认值： ``0.5``。
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

    .. py:method:: get_ref_kinetics()

        获取参考动能。

        返回：
            Tensor。参考动能。

    .. py:method:: kinetics()
        :property:

        参考动能。

        返回：
            Tensor。参考动能。

    .. py:method:: reconstruct_temperature(temperature: Union[float, ndarray, Tensor, List[float]])

        重置参考温度。

        参数：
            - **temperature** (Union[float, ndarray, Tensor, List[float]]) - 温度。

    .. py:method:: set_degrees_of_freedom(dofs)

        设置自由度。

        参数：
            - **dofs** (int) - 自由度。

    .. py:method:: set_temperature(temperature: Union[float, ndarray, Tensor, List[float]])

        设置参考温度数值。温度数组的形状必须与当前温度数组的形状相同。

        参数：
            - **temperature** (Union[float, ndarray, Tensor, List[float]]) - 温度。

        返回：
            Tensor。温度。

    .. py:method:: temperature()
        :property:

        参考温度。

        返回：
            Tensor。参考温度。

    .. py:method:: velocity_scale(sim_kinetics, ref_kinetics, ratio=1.0)

        计算温度耦合的速度缩放因子。

        参数：
            - **sim_kinetics** (Tensor) - 模拟动能。Tensor形状为 :math:`(B, D)` 。这里 :math:`B` 是分子模拟中walker的数目， :math:`D` 是模拟系统的空间维数，通常为3。数据类型是float。
            - **ref_kinetics** (Tensor) - 参考动能。Tensor形状为 :math:`(B, D)` 。数据类型是float。
            - **ratio** (float, 可选) - 温度耦合的比率。默认值： ``1.0``。

        返回：
            Tensor。速度缩放因子。