mindscience.diffuser.DDIMPipeline
==================================

.. py:class:: mindscience.diffuser.DDIMPipeline(model, scheduler, batch_size, seq_len, num_inference_steps=1000, compute_dtype=mstype.float32)

    DDIM 生成流程管线。

    参数：
        - **model** (nn.Cell) - 扩散过程的主干网络模型。
        - **scheduler** (DDIMScheduler) - 与 `model` 配合使用的调度器，用于对样本进行去噪。
        - **batch_size** (int) - 生成样本的批次大小。
        - **seq_len** (int) - 输入序列的长度。
        - **num_inference_steps** (int, 可选) - 去噪推理步数。默认 ``1000``。
        - **compute_dtype** (mindspore.dtype, 可选) - 计算使用的数据类型。支持 ``mstype.float32`` 或 ``mstype.float16``。默认 ``mstype.float32``，表示 ``mindspore.float32``。

    异常：
        - **TypeError** - 当 `scheduler` 不是 `DDIMScheduler` 类型时抛出。
        - **ValueError** - 当 `num_inference_steps` 大于 `scheduler.num_train_timesteps` 时抛出。
