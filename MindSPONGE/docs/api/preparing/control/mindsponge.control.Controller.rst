mindsponge.control.Controller
=============================

.. py:class:: mindsponge.control.Controller(system, control_step=1)

    控制器用于控制仿真过程中的参数，包括积分器、恒温器、气压调节器、约束器等。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **control_step** (int) - 控制器执行的步骤间隔。默认值：1。

    .. py:method:: get_com(coordinate)

        计算质心坐标。

        参数：
            - **coordinate** (Tensor) - 坐标。

        返回：
            Tensor。质心坐标。

    .. py:method:: get_com_velocity(velocity)

        计算质心速度。

        参数：
            - **velocity** (Tensor) - 速度。

        返回：
            Tensor。质心速度。

    .. py:method:: get_kinetics(velocity)

        根据速度计算动力学。

        参数：
            - **velocity** (Tensor) - 速度。

        返回：
            Tensor。根据速度获得的动力学。

    .. py:method:: get_pressure(kinetics, virial, pbc_box)

        根据动力学，维里，周期性边界条件box计算压力。

        参数：
            - **kinetics** (Tensor) - 动力学。
            - **virial** (Tensor) - 维里。
            - **pbc_box** (Tensor) - 周期性边界条件box。

        返回：
            Tensor。根据动力学，维里，周期性边界条件box计算压力。

    .. py:method:: get_temperature(kinetics)

        根据速度计算温度。

        参数：
            - **kinetics** (Tensor) - 动力学。

        返回：
            Tensor。温度。

    .. py:method:: get_virial(pbc_grad, pbc_box)

        根据周期性边界条件和梯度计算维里。

        参数：
            - **pbc_grad** (Tensor) - 周期性边界条件box的梯度。
            - **pbc_box** (Tensor) - 周期性边界条件box

        返回：
            Tensor。维里。

    .. py:method:: get_volume(pbc_box)

        根据周期性边界条件box计算容积。

        参数：
            - **pbc_box** (Tensor) - 用于计算容积的周期性边界条件。

        返回：
            Tensor。容积。

    .. py:method:: set_time_step(dt)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步时间所需时间。

    .. py:method:: set_degrees_of_freedom(dofs)

        设置自由度。

        参数：
            - **dofs** (int) - 自由度。

    .. py:method:: update_coordinate(coordinate, success=True)

        更新坐标的参数。

        参数：
            - **coordinate** (Tensor) - 原子的位置坐标。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值：True。

        返回：
            bool。是否更新了坐标的参数。

    .. py:method:: update_pbc_box(pbc_box, success)

        更新周期性边界条件box。

        参数：
            - **pbc_box** (Tensor) - 周期性边界条件box。
            - **success** (bool, 可选) - 判断是否成功的参数。

        返回：
            bool。是否更新了周期性边界条件box。