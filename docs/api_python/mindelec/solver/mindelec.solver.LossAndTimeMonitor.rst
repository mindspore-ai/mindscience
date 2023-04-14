mindelec.solver.LossAndTimeMonitor
==================================

.. py:class:: mindelec.solver.LossAndTimeMonitor(data_size, per_print_times=1)

    监控训练中的loss。

    如果loss是NAN或INF，将终止训练。

    .. note::
        如果 `per_print_times` 为 ``0``，则不打印loss。

    参数：
        - **data_size** (int) - 每个epoch数据集的批次数。
        - **per_print_times** (int) - 表示每隔多少个step打印一次loss。默认值： ``1``。

    异常：
        - **ValueError** - 如果 `data_size` 不是整数或小于零。
        - **ValueError** - 如果 `per_print_times` 不是整数或小于零。

    .. py:method:: mindelec.solver.LossAndTimeMonitor.epoch_begin(run_context)

        在epoch开始时设置开始时间。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息，详情请参考 `mindspore.train.RunContext <https://mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.RunContext.html#mindspore.train.RunContext>`_。

    .. py:method:: mindelec.solver.LossAndTimeMonitor.epoch_end(run_context)

        在epoch结束时获得损失。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息，详情请参考 `mindspore.train.RunContext <https://mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.RunContext.html#mindspore.train.RunContext>`_。
