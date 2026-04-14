sponge.control.Controller
=============================

.. py:class:: sponge.control.Controller(system: :class:`sponge.system.Molecule`, control_step: int = 1, **kwargs)

    MindSPONEG的控制器模块中的基类。

    在 :class:`sponge.optimizer.Updater` 中使用 `sponge.control.Controller` 用于控制仿真过程中的七个变量，包括坐标、速度、力、能量、动力学、维里和PBC box。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 模拟系统。
        - **control_step** (int, 可选) - 控制器执行的步骤间隔。默认值： ``1``。
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

    .. py:method:: boltzmann()
        :property:

        获取当前单元中的玻尔兹曼常数。

        返回：
            float，当前单元中的玻尔兹曼常数。

    .. py:method:: get_com(coordinate: Tensor, keepdims: bool = True)

        计算质心坐标。

        参数：
            - **coordinate** (Tensor) - 原子坐标。shape为 :math:`(B, A, D)` 。数据类型为float。
            - **keepdims** (bool) - 如果为 ``True``，在结果中保持第二根轴对应的维度且长度为1。默认值： ``True`` 。

        返回：
            Tensor，质心坐标。shape为 :math:`(B, A, D)` 或 :math:`(B, D)` 。数据类型为float。

    .. py:method:: get_com_velocity(velocity: Tensor, keepdims: bool = True)

        计算质心速度。

        参数：
            - **velocity** (Tensor) - 速度的Tensor。shape为 :math:`(B, A, D)` 。数据类型为float。
            - **keepdims** (bool) - 如果为True，在结果中保持第二根轴对应的维度且长度为1。默认值： ``True`` 。

        返回：
            Tensor，质心速度。shape为 :math:`(B, A, D)` 或 :math:`(B, D)` 。数据类型为float。

    .. py:method:: get_kinetics(velocity: Tensor)

        根据速度计算动能。

        参数：
            - **velocity** (Tensor) - 原子速度。shape为 :math:`(B, A, D)` 。数据类型为float。

        返回：
            Tensor，动能。shape为 :math:`(B, A, D)` 。数据类型为float。

    .. py:method:: get_pressure(kinetics: Tensor, virial: Tensor, pbc_box: Tensor)

        根据动力学，维里和周期性边界条件计算压力。

        参数：
            - **kinetics** (Tensor) - 动力学的Tensor。shape为 :math:`(B, D)` 。数据类型为float。
            - **virial** (Tensor) - 维里的Tensor。shape为 :math:`(B, D)` 。数据类型为float。
            - **pbc_box** (Tensor) - 周期性边界条件box的Tensor。shape为 :math:`(B, D)` 。数据类型为float。

        返回：
            Tensor，压力。shape为 :math:`(B, D)` 。数据类型为float。

    .. py:method:: get_temperature(kinetics: Tensor = None)

        根据速度计算温度。

        参数：
            - **kinetics** (Tensor) - 动能。shape为 :math:`(B, D)` 。数据类型为float。默认值： ``None``。

        返回：
            Tensor，温度。shape为 :math:`(B)` 。数据类型为float。

    .. py:method:: get_volume(pbc_box: Tensor)

        根据周期性边界条件box计算容积。

        参数：
            - **pbc_box** (Tensor) - 用于计算容积的周期性边界条件。shape为 :math:`(B, D)` 。数据类型为float。

        返回：
            Tensor，容积。shape为 :math:`(B)` 。数据类型为float。

    .. py:method:: set_degrees_of_freedom(dofs: int)

        设置自由度(DOFs)。

        参数：
            - **dofs** (int) - 自由度。

    .. py:method:: set_time_step(dt: float)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步时长。

    .. py:method:: update_coordinate(coordinate: Tensor)

        更新模拟系统的坐标。

        参数：
            - **coordinate** (Tensor) - 原子坐标。shape为 :math:`(B, A, D)` 。数据类型为float。

        返回：
            Tensor，更新后的坐标。shape为 :math:`(B, A, D)`。

    .. py:method:: update_pbc_box(pbc_box: Tensor)

        更新周期性边界条件box的参数。

        参数：
            - **pbc_box** (Tensor) - 周期性边界条件盒子（PBC box）。shape为 :math:`(B, D)` 。数据类型为float。

        返回：
            Tensor，更新后的PBC box。shape为 :math:`(B, D)` 。

