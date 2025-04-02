mindflow.cell.DDIMPipeline
============================

.. py:class:: mindflow.cell.DDIMPipeline(model, scheduler, batch_size, seq_len, num_inference_steps=1000, compute_dtype=mstype.float32)

    DDIM采样过程控制实现。

    参数：
        - **model** (nn.Cell) - 训练模型。
        - **scheduler** (DDIMScheduler) - 噪声控制器，用于去噪。
        - **batch_size** (int) - batch大小。
        - **seq_len** (int) - 序列长度。
        - **num_inference_steps** (int) - 采样的步数。默认值： ``1000`` 。
        - **compute_dtype** (mindspore.dtype) - 数据类型。默认值： ``mstype.float32`` ，表示 ``mindspore.float32`` 。

    异常：
        - **TypeError** - 如果 `scheduler` 不是 `DDIMScheduler` 类型。
        - **ValueError** - 如果 `num_inference_steps` 大于 `scheduler.num_train_timesteps` 。
