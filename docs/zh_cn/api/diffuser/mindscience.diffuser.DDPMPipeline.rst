.. py:class:: mindscience.diffuser.DDPMPipeline(model, scheduler, batch_size, seq_len, num_inference_steps=1000, compute_dtype=mstype.float32)

    用于 DDPM 生成的流水线，继承自 `DiffusionPipeline`。

    参数：
        - **model** (nn.Cell) - 扩散主干模型。
        - **scheduler** (DDPMScheduler) - 与 `model` 配合使用的调度器，用于对加噪样本进行去噪。
        - **batch_size** (int) - batch大小。
        - **seq_len** (int) - 输入序列长度。
        - **num_inference_steps** (int) - 去噪步数，默认 ``1000``。
        - **compute_dtype** (mindspore.dtype) - 计算数据类型，可为 ``mstype.float32`` 或 ``mstype.float16``；默认 ``mstype.float32``（即 ``mindspore.float32``）。

    异常：
        - **TypeError** - 当 `scheduler` 不是 `DDPMScheduler` 类型时抛出。
        - **ValueError** - 当 `num_inference_steps` 不等于 `scheduler.num_train_timesteps` 时抛出。
