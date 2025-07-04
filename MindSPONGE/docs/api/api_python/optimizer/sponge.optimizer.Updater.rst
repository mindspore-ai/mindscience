sponge.optimizer.Updater
============================

.. py:class:: sponge.optimizer.Updater(system: Molecule, controller: Union[Controller, List[Controller]] = None, time_step: float = 1e-3, velocity: Union[Tensor, ndarray, List[float]] = None, weight_decay: float = 0.0, loss_scale: float = 1.0, **kwargs)

    MindSPONGE更新器的基类。是MindSpore中 `mindspore.nn.Optimizer` 的特殊子类。 :class:`sponge.optimizer.Updater` 更新仿真系统中的原子坐标。原子坐标的更新要求原子受力和原子速度。力是从外界传递而来，速度是 :class:`sponge.optimizer.Updater` 自己的参数。
    当使用周期性边界条件的时候， :class:`sponge.optimizer.Updater` 也能够通过仿真系统的维里更新周期性边界条件箱的尺寸。
    在通过一系列的 :class:`sponge.control.Controller` 控制器的优化过程中， :class:`sponge.optimizer.Updater` 控制着七个变量的值，分别是：坐标、速度、力、能量、动能、维里和周期性边界条件箱。如果传入超过一个 :class:`sponge.control.Controller` ，它们将按照队列顺序进行工作。

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 模拟系统。
        - **controller** (Union[:class:`sponge.control.Controller`, List[:class:`sponge.control.Controller`]], 可选) - 控制器或控制器列表来控制模拟系统中的七个变量（坐标、速度、力、能量、动能、维里和周期性边界条件箱）。默认值： ``None``。
        - **time_step** (float, 可选) - 单步时间。默认值： ``1e-3``。
        - **velocity** (Union[Tensor, ndarray, List[float]], 可选) - 原子速度的array，shape为 `(A, D)` 或 `(B, A, D)` ，这里 `B`是batch size， `A`是原子总数， `D`是模拟系统的维度，一般为3。数据类型为float。默认值： ``None``。
        - **weight_decay** (float, 可选) - 权重衰减值。默认值： ``0.0``。
        - **loss_scale** (float, 可选) - 梯度缩放系数。默认值： ``1.0``。
        - **kwargs** (dict) - 关键字参数。

    输入：
        - **energy** (Tensor) - 系统的能量。shape为 :math:`(B, A, D)` 。数据类型为float。
        - **force** (Tensor) - 系统的力。shape为 :math:`(B, A, D)` 。数据类型为float。
        - **virial** (Tensor， 可选) - 系统的维里。shape为 :math:`(B, A, D)` 。数据类型为float。默认值： ``None``。

    输出：
        - **success** (bool) - 是否成功完成当前优化单步并且移动到下一步。

    .. py:method:: boltzmann()
        :property:

        当前单位下的布尔兹曼常数。

        返回：
            float，当前单位下的布尔兹曼常数。

    .. py:method:: decay_and_scale_grad(force: Tensor, virial: Tensor = None)

        对力和维里进行权重衰减和梯度标度。

        参数：
            - **force** (Tensor) - 力的Tensor，数据类型为float。
            - **virial** (Tensor, 可选) - 维里的Tensor，数据类型为float。默认值：``None``。

        返回：
            - Tensor，权重衰减和梯度标度之后的力。
            - Tensor，权重衰减和梯度标度之后的维里。如果pbc_box是None，输出维里与输入保持一致。

    .. py:method:: get_dt()

        获取当前单步的学习率。

        返回：
            float。当前单步的学习率。

    .. py:method:: get_kinetics(velocity: Tensor)

        获取动能。

        参数：
            - **velocity** (Tensor) - 原子速度的Tensor，数据类型为float。

        返回：
            Tensor，系统中的动能。

    .. py:method:: get_pressure(kinetics: Tensor, virial: Tensor, pbc_box: Tensor)

        获得压力。

        参数：
            - **kinetics** (Tensor) - 动能的Tensor，数据类型为float。默认值： ``None``。
            - **virial** (Tensor) - 维里的Tensor，数据类型为float。默认值： ``None``。
            - **pbc_box** (Tensor) - 周期性边界条件箱的Tensor，数据类型为float。默认值： ``None``。

        返回：
            Tensor，系统的压力。

    .. py:method:: get_temperature(kinetics: Tensor = None)

        获取温度。

        参数：
            - **kinetics** (Tensor) - 动能的Tensor，数据类型为float。默认值： ``None``。

        返回：
            Tensor，系统的温度。

    .. py:method:: get_velocity()

        获取速度。

        返回：
            Tensor，系统中原子的速度。

    .. py:method:: next_step(success: bool = True)

        完成当前优化step并且进行到下一个step。

        参数：
            - **success** (bool) - 是否完成当前优化step并且移动到下一步。默认值： ``True`` 。

        返回：
            bool，是否成功完成当前优化step并且移动到下一步。

    .. py:method:: press_unit_scale()
        :property:

        压力的参考值。

        返回：
            float，压力的参考值。

    .. py:method:: set_degrees_of_freedom(dofs: int)

        设置系统的自由度。

        参数：
            - **dofs** (int) - 自由度。

    .. py:method:: set_step(step: int = 0)

        设置系统的当前步数。

        参数：
            - **step** (int) - 系统的当前步数。默认值： ``0``。

    .. py:method:: update_coordinate(coordinate: Tensor, success: bool = True)

        更新坐标的参数。

        参数：
            - **coordinate** (Tensor) - 原子的位置坐标的Tensor。数据类型为float。
            - **success** (bool) - 判断是否更新坐标。默认值： ``True`` 。

        返回：
            bool，是否成功更新了坐标的参数。

    .. py:method:: update_kinetics(kinetics: Tensor, success: bool = True)

        更新动能参数。

        参数：
            - **kinetics** (Tensor) - 动能的Tensor。数据类型为float。
            - **success** (bool) - 判断是否更新动能参数。默认值： ``True`` 。

        返回：
            bool。是否成功更新了动能参数。

    .. py:method:: update_pbc_box(pbc_box: Tensor, success: bool = True)

        更新周期性边界条件箱的参数。

        参数：
            - **pbc_box** (Tensor) - 周期性边界条件box的Tensor。数据类型为float。
            - **success** (bool) - 判断是否更新周期性边界条件箱的参数。默认值： ``True`` 。

        返回：
            bool，是否成功更新了周期性边界条件箱的参数。

    .. py:method:: update_pressure(pressure: Tensor, success: bool = True)

        更新压力参数。

        参数：
            - **pressure** (Tensor) - 压力的Tensor。数据类型为float。
            - **success** (bool) - 判断是否更新压力参数。默认值： ``True`` 。

        返回：
            bool。是否成功更新了压力参数。

    .. py:method:: update_temperature(temperature: Tensor, success: bool = True)

        更新温度参数。

        参数：
            - **temperature** (Tensor) - 温度的Tensor。数据类型为float。
            - **success** (bool) - 判断是否更新温度参数。默认值： ``True`` 。

        返回：
            bool。是否成功更新了温度参数。

    .. py:method:: update_velocity(velocity: Tensor, success: bool = True)

        更新速度参数。

        参数：
            - **velocity** (Tensor) - 原子速度的Tensor。数据类型为float。
            - **success** (bool) - 判断是否更新速度参数。默认值： ``True`` 。

        返回：
            bool，是否成功更新了速度参数。

    .. py:method:: update_virial(virial: Tensor, success: bool = True)

        更新维里参数。

        参数：
            - **virial** (Tensor) - 维里的Tensor。数据类型为float。
            - **success** (bool) - 判断是否更新维里参数。默认值： ``True`` 。

        返回：
            bool。是否成功更新了维里参数。