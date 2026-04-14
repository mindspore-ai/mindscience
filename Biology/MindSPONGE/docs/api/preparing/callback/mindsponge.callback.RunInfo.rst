mindsponge.callback.RunInfo
===========================

.. py:class:: mindsponge.callback.RunInfo(print_freq: int = 1)

    回调函数打印MD模拟的信息。

    参数：
        - **print_freq** (int) - 打印信息的频率。默认值：1。

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