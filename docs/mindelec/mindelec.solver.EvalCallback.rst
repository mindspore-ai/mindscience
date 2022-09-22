mindelec.solver.EvalCallback
============================

.. py:class:: mindelec.solver.EvalCallback(model, eval_ds, eval_interval)

    在训练期间评估模型。

    参数：
        - **model** (Model) - 测试网络。
        - **eval_ds** (Dataset) - 用于评估模型的数据集。
        - **eval_interval** (int) - 指定在计算之前要训练多少个epoch。

    .. py:method:: mindelec.solver.EvalCallback.epoch_end(run_context)

        在epoch结束时评估模型。

        参数：
            - **run_context** (RunContext) - 训练运行的上下文。

