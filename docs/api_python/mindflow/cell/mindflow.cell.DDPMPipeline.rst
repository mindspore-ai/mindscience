mindflow.cell.DDPMPipeline
============================

.. py:class:: mindflow.cell.DDPMPipeline(model, scheduler, batch_size, seq_len, num_inference_steps=1000, compute_dtype=mstype.float32)

    DDPM采样过程控制实现。

    参数：
        - **model** (nn.Cell) - 训练模型。
        - **scheduler** (DDPMScheduler) - 噪声控制器，用于去噪。
        - **batch_size** (int) - batch大小。
        - **seq_len** (int) - 序列长度。
        - **num_inference_steps** (int) - 采样的步数。默认值： ``1000`` 。
        - **compute_dtype** (mindspore.dtype) - 数据类型。默认值： ``mstype.float32`` ，表示 ``mindspore.float32`` 。

    异常：
        - **TypeError** - 如果 `scheduler` 不是 `DDPMScheduler` 类型。
        - **ValueError** - 如果 `num_inference_steps` 不等于 `scheduler.num_train_timesteps` 。
