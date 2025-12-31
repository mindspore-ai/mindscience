mindscience.diffuser.DDPMPipeline
====================================

.. py:class:: mindscience.diffuser.DDPMPipeline(model, scheduler, batch_size, seq_len, num_inference_steps=1000, compute_dtype=mstype.float32)

    DDPM 生成流程管线。

    参数：
        - **model** (nn.Cell) - 扩散模型主干网络。
        - **scheduler** (DDPMScheduler) - 与 `model` 配合使用的调度器，用于对加噪样本进行去噪。
        - **batch_size** (int) - 生成样本的 batch 大小。
        - **seq_len** (int) - 输入序列长度。
        - **num_inference_steps** (int, 可选) - 去噪步数。默认 ``1000``。
        - **compute_dtype** (mindspore.dtype, 可选) - 计算使用的数据类型，支持 ``mstype.float32`` 或 ``mstype.float16``。默认 ``mstype.float32``，表示 ``mindspore.float32``。

    异常：
        - **TypeError** - 当 `scheduler` 不是 `DDPMScheduler` 类型时抛出。
        - **ValueError** - 当 `num_inference_steps` 不等于 `scheduler.num_train_timesteps` 时抛出。
