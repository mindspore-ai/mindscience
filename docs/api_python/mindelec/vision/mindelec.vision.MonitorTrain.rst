mindelec.vision.MonitorTrain
============================

.. py:class:: mindelec.vision.MonitorTrain(per_print_times=1, summary_dir='./summary_train')

    训练损失监测器。

    .. note::
        如果 `per_print_times` 为 ``0``，则不打印loss。

    参数：
        - **per_print_times** (int) - 打印loss间隔。默认值： ``1``。
        - **summary_dir** (str) - 摘要保存路径。默认值： ``'./summary_t'``。

    .. py:method:: mindelec.vision.MonitorTrain.step_end(run_context)

        在epoch结束时评估模型。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息，详情请参考 `mindspore.train.RunContext <https://mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.RunContext.html#mindspore.train.RunContext>`_。