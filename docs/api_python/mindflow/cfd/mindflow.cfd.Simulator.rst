mindflow.cfd.Simulator
=========================

.. py:class:: mindflow.cfd.Simulator(config, net_dict=None)

    CFD仿真器。MindFlow CFD中最上层的类。

    参数：
        - **config** (dict) - 参数字典。
        - **net_dict** (dict) - 网络字典, 默认为 ``None``。

    .. py:method:: integration_step(con_var, timestep)

        按时间步长积分。

        参数：
            - **con_var** (Tensor) - 守恒量。
            - **timestep** (float) - 积分的时间步长。

        返回：
            Tensor，一个时间步长之后的守恒量。