mindelec.vision.MonitorEval
===========================

.. py:class:: mindelec.vision.MonitorEval(summary_dir='./summary_eval', model=None, eval_ds=None, eval_interval=10, draw_flag=True)

    用于评估的LossMonitor。

    参数：
        - **summary_dir** (str) - 摘要保存路径。默认值： ``'./summary_eval'``。
        - **model** (Solver) - 评估的模型对象。默认值： ``None``。
        - **eval_ds** (Dataset) - eval数据集。默认值： ``None``。
        - **eval_interval** (int) - eval间隔。默认值： ``10``。
        - **draw_flag** (bool) - 指定是否保存摘要记录。默认值： ``True``。

    .. py:method:: mindelec.vision.MonitorEval.epoch_end(run_context)

        在epoch结束时评估模型。

        参数：
            - **run_context** (RunContext) - 包含一些模型中的信息，详情请参考 `mindspore.train.RunContext <https://mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.RunContext.html#mindspore.train.RunContext>`_。

