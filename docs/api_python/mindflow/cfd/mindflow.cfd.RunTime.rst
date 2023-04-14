mindflow.cfd.RunTime
=========================

.. py:class:: mindflow.cfd.RunTime(config, mesh_info, material)

    仿真运行时控制器。

    参数：
        - **config** (dict) - 参数字典。
        - **mesh_info** (MeshInfo) - 计算网格的信息。
        - **material** (Material) - 流体材料模型。

    .. py:method:: advance()

        根据时间步长向前仿真。

        异常：
            - **NotImplementedError** - 如果 `time step` 不合法。

    .. py:method:: compute_timestep(pri_var)

        计算物理的时间步长。

        参数：
            - **pri_var** (Tensor) - 原始变量。

    .. py:method:: time_loop(pri_var)

        是否继续仿真，当前时间达到终值或 ``NAN`` 值时返回False。

        参数：
            - **pri_var** (Tensor) - 原始变量。

        返回：
            bool，是否继续仿真。

        异常：
            - **ValueError** - 如果 `pri_var` 值为 ``NAN``。