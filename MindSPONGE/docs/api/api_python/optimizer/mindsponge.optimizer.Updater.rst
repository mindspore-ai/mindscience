mindsponge.optimizer.Updater
============================

.. py:class:: mindsponge.optimizer.Updater(system, controller=None, time_step=1e-3, velocity=None, weight_decay=0.0, loss_scale=1.0)

    更新空间参数(坐标和周期性边界条件box)的优化器。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **controller** (Controller) - 控制器。默认值："None"。
        - **time_step** (float) - 单步时间。默认值：1e-3。
        - **velocity** (Tensor) - 速度，shape为(B, A, D)。默认值："None"。
        - **weight_decay** (float) - 权重衰减值。默认值：0.0。
        - **loss_scale** (float) - 误差比例。默认值：1.0。

    符号：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **D** - 模拟系统的维度。

    .. py:method:: get_dt()

        获取学习率。

        返回：
            float。当前step的学习率。

    .. py:method:: get_kinetics(velocity)

        获取动力学。

        参数：
            - **velocity** (Tensor) - 速度。

        返回：
            Tensor。动力学。

    .. py:method:: get_pressure(kinetics, virial, pbc_box)

        获得压力。

        参数：
            - **kinetics** (Tensor) - 动力学。
            - **virial** (Tensor) - 维里。
            - **pbc_box** (Tensor) - 周期性边界条件box。

        返回：
            Tensor。压力。

    .. py:method:: get_temperature(kinetics=None)

        获取温度。

        参数：
            - **kinetics** (Tensor) - 动力学。默认值："None"。

        返回：
            Tensor。温度。

    .. py:method:: get_velocity()

        获取速度。

        返回：
            Tensor。速度值。

    .. py:method:: get_virial(pbc_grad, pbc_box)

        获取维里。

        参数：
            - **pbc_grad** (Tensor) - 周期性边界条件box的梯度。
            - **pbc_box** (Tensor) - 周期性边界条件box。

        返回：
            Tensor。维里。

    .. py:method:: next_step(success=True)

        完成当前优化step并且进行到下一个step。

        参数：
            - **success** (bool) - 是否移动到下一步。

        返回：
            bool。

    .. py:method:: set_step(step=0)

        设置步数。

        参数：
            - **step** (int) - 步数。默认值：0。

    .. py:method:: update_coordinate(coordinate, success=True)

        更新坐标的参数。

        参数：
            - **coordinate** (Tensor) - 原子的位置坐标。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了坐标的参数。

    .. py:method:: update_kinetics(kinetics, success=True)

        更新动力学参数。

        参数：
            - **kinetics** (Tensor) - 动力学。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了动力学参数。

    .. py:method:: update_pbc_box(pbc_box, success=True)

        更新周期性边界条件box。

        参数：
            - **pbc_box** (Tensor) - 周期性边界条件box。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了周期性边界条件box。

    .. py:method:: update_pressure(pressure, success=True)

        更新压力参数。

        参数：
            - **pressure** (Tensor) - 压力。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了压力参数。

    .. py:method:: update_temperature(temperature, success=True)

        更新温度参数。

        参数：
            - **temperature** (Tensor) - 温度。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了温度参数。

    .. py:method:: update_velocity(velocity, success=True)

        更新速度参数。

        参数：
            - **velocity** (Tensor) - 速度。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了速度参数。

    .. py:method:: update_virial(virial, success=True)

        更新维里参数。

        参数：
            - **virial** (Tensor) - 维里。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了维里参数。