sponge.control.Controller
=============================

.. py:class:: sponge.control.Controller(system: Molecule, control_step: int = 1, **kwargs)

    MindSPONEG的控制器模块中的基类。
    在 `Updater` 中使用 `Controller` 用于控制仿真过程中的七个变量，包括坐标、速度、力、能量、动力学、维里和PBC box。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **control_step** (int) - 控制器执行的步骤间隔。默认值：1。
        - **kwargs** (dict) - 其他参数，用于扩展。

    输入：
        - **coordinate** (Tensor) - shape为 `(B, A, D)` 的Tensor。数据类型是float。
        - **velocity** (Tensor) - shape为 `(B, A, D)` 的Tensor。数据类型是float。
        - **force** (Tensor) - shape为 `(B, A, D)` 的Tensor。数据类型是float。
        - **energy** (Tensor) - shape为 `(B, 1)` 的Tensor。数据类型是float。
        - **kinetics** (Tensor) - shape为 `(B, D)` 的Tensor。数据类型是float。
        - **virial** (Tensor) - shape为 `(B, D)` 的Tensor。数据类型是float。
        - **pbc_box** (Tensor) - shape为 `(B, D)` 的Tensor。数据类型是float。
        - **step** (int) - 模拟步数。默认值：0

    输出：
        - 坐标，shape为 `(B, A, D)` 的Tensor。数据类型是float。
        - 速度，shape为 `(B, A, D)` 的Tensor。数据类型是float。
        - 力，shape为 `(B, A, D)` 的Tensor。数据类型是float。
        - 能量，shape为 `(B, 1)` 的Tensor。数据类型是float。
        - 动力学，shape为 `(B, D)` 的Tensor。数据类型是float。
        - 维里，shape为 `(B, D)` 的Tensor。数据类型是float。
        - 周期性边界条件PBC box，shape为 `(B, D)` 的Tensor。数据类型是float。

    符号：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **D** - 仿真系统的空间维度。通常是3。

    .. py:method:: boltzmann()

        获取当前单元中的玻尔兹曼常数。

        返回：
            float。当前单元中的玻尔兹曼常数。

    .. py:method:: set_time_step(dt: float)

        设置模拟单步时间。

        参数：
            - **dt** (float) - 单步时长。

    .. py:method:: set_degrees_of_freedom(dofs: int)

        设置自由度(DOFs)。

        参数：
            - **dofs** (int) - 自由度。

    .. py:method:: update_coordinate(coordinate: Tensor) -> Tensor

        更新模拟系统的坐标。

        参数：
            - **coordinate** (Tensor) - 原子坐标的Tensor。shape为 `(B, A, D)` 。数据类型为float。

        返回：
            Tensor。更新后的坐标的Tensor，shape和数据类型与原来一致。

    .. py:method:: update_pbc_box(pbc_box: Tensor) -> Tensor

        更新周期性边界条件box的参数。

        参数：
            - **pbc_box** (Tensor) - 周期性边界条件box的Tensor。shape为 `(B, D)` 。数据类型为float。

        返回：
            Tensor。更新后的PBC box的Tensor，shape和数据类型与原来的 `pbc_box` 一致。

    .. py:method:: get_kinetics(velocity: Tensor) -> Tensor

        根据速度计算动力学。

        参数：
            - **velocity** (Tensor) - 原子速度的Tensor。shape为 `(B, A, D)` 。数据类型为float。

        返回：
            Tensor，动力学。shape为 `(B, A, D)` 。数据类型为float。

    .. py:method:: get_temperature(kinetics: Tensor = None) -> Tensor

        根据速度计算温度。

        参数：
            - **kinetics** (Tensor) - 动力学的Tensor。shape为 `(B, D)` 。数据类型为float。默认值："None"。

        返回：
            Tensor，温度。shape为 `(B)` 。数据类型为float。

    .. py:method:: get_volume(pbc_box: Tensor) -> Tensor:

        根据周期性边界条件box计算容积。

        参数：
            - **pbc_box** (Tensor) - 用于计算容积的周期性边界条件。shape为 `(B, D)` 。数据类型为float。

        返回：
            Tensor，容积。shape为 `(B)` 。数据类型为float。

    .. py:method:: get_pressure(kinetics: Tensor, virial: Tensor, pbc_box: Tensor) -> Tensor

        根据动力学，维里和周期性边界条件计算压力。

        参数：
            - **kinetics** (Tensor) - 动力学的Tensor。shape为 `(B, D)` 。数据类型为float。
            - **virial** (Tensor) - 维里的Tensor。shape为 `(B, D)` 。数据类型为float。
            - **pbc_box** (Tensor) - 周期性边界条件box的Tensor。shape为 `(B, D)` 。数据类型为float。

        返回：
            Tensor。根据动力学，维里，周期性边界条件box计算压力。shape为 `(B, D)` 。数据类型为float。

    .. py:method:: get_com(coordinate: Tensor, keepdims: bool = True) -> Tensor

        计算质心坐标。

        参数：
            - **coordinate** (Tensor) - 原子坐标的Tensor。shape为 `(B, A, D)` 。数据类型为float。
            - **keepdims** (bool) - 如果为True，在结果中保持第二根轴对应的维度且长度为1。默认值： ``True`` 。

        返回：
            Tensor。质心坐标。shape为 `(B, A, D)` 或 `(B, D)` 。数据类型为float。

    .. py:method:: get_com_velocity(velocity: Tensor, keepdims: bool = True) -> Tensor

        计算质心速度。

        参数：
            - **velocity** (Tensor) - 速度的Tensor。shape为 `(B, A, D)` 。数据类型为float。
            - **keepdims** (bool) - 如果为True，在结果中保持第二根轴对应的维度且长度为1。默认值： ``True`` 。

        返回：
            Tensor。质心速度。shape为 `(B, A, D)` 或 `(B, D)` 。数据类型为float。