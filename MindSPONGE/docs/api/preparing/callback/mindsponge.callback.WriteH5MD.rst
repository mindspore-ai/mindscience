mindsponge.callback.WriteH5MD
=============================

.. py:class:: mindsponge.callback.WriteH5MD(system: Molecule, filename: str, save_freq: int = 1, directory: str = None, write_velocity: bool = False, write_force: bool = False, write_image: bool = True, length_unit: str = None, energy_unit: str = None)

    回调写HDF5分子数据(H5MD)文件。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **filename** (str) - 输出H5MD文件的名字。
        - **save_freq** (int) - 保存频率。默认值：1。
        - **directory** (str) - 输出文件的目录。
        - **write_velocity** (bool) - 是否把系统的速度写入H5MD文件中。默认值： ``False`` 。
        - **write_force** (bool) - 是否把系统的力写入H5MD文件中。默认值： ``False`` 。
        - **write_image** (bool) - 是否把系统的位置图片写入H5MD文件中。默认值： ``True`` 。
        - **length_unit** (str) - 坐标的长度单位。
        - **energy_unit** (str) - 能量单位。

    .. py:method:: begin(run_context: RunContext)

        在执行网络之前调用一次。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息。

    .. py:method:: end(run_context: RunContext)

        在网络训练之后调用一次。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息。

    .. py:method:: epoch_begin(run_context: RunContext)

        在每个epoch开始之前调用。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息。

    .. py:method:: epoch_end(run_context: RunContext)

        在每个epoch结束之后调用。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息。

    .. py:method:: step_begin(run_context: RunContext)

        在每个单步开始之前调用。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息。

    .. py:method:: step_end(run_context: RunContext)

        在每个单步结束之后调用。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息。