mindscience.diffuser.DiffusionScheduler
==========================================

.. py:class:: mindscience.diffuser.DiffusionScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True, clip_sample_range=1.0, thresholding=False, sample_max_value=1.0, dynamic_thresholding_ratio=0.995, rescale_betas_zero_snr=False, timestep_spacing="leading", compute_dtype=mstype.float32)

    扩散调度器基类。

    参数：
        - **num_train_timesteps** (int, 可选) - 训练模型时使用的扩散步数。默认 ``1000``。
        - **beta_start** (float, 可选) - 推理阶段 `beta` 起始值。默认 ``0.0001``。
        - **beta_end** (float, 可选) - `beta` 终止值。默认 ``0.02``。
        - **beta_schedule** (str, 可选) - `beta` 调度方式，用于将 `beta` 取值范围映射为模型迭代所需的一组 `beta` 序列，支持以下类型： ``"linear"``、 ``"scaled_linear"`` 或 ``"squaredcos_cap_v2"``。默认 ``"squaredcos_cap_v2"``。
        - **prediction_type** (str, 可选) - 调度器函数的预测类型，支持以下类型： ``"epsilon"``（预测噪声）、 ``"sample"``（预测带噪样本）或 ``"v_prediction"``（参见 `Imagen Video <https://imagen.research.google/video/paper.pdf>`_ 论文 2.4 节）。默认 ``"epsilon"``。
        - **clip_sample** (bool, 可选) - 是否对预测样本进行裁剪以提升数值稳定性。默认 ``True``。
        - **clip_sample_range** (float, 可选) - 样本裁剪的最大幅值，仅在 `clip_sample=True` 时生效。默认 ``1.0``。
        - **thresholding** (bool, 可选) - 是否使用动态阈值方法。该方法不适用于潜空间扩散模型（如 Stable Diffusion）。默认 ``False``。
        - **sample_max_value** (float, 可选) - 动态阈值方法中的阈值上限，仅在 `thresholding=True` 时生效。默认 ``1.0``。
        - **dynamic_thresholding_ratio** (float, 可选) - 动态阈值方法中使用的比例参数，仅在 `thresholding=True` 时生效。默认 ``0.995``。
        - **rescale_betas_zero_snr** (bool, 可选) - 是否将 `beta` 重缩放，使终端信噪比（SNR）为零。启用后，模型可以生成极亮或极暗的样本，而不局限于中等亮度范围。该设置与 `offset_noise <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_ 相关。默认 ``False``。
        - **timestep_spacing** (str, 可选) - 推理过程中时间步的缩放方式。相关说明可参考论文 `Common Diffusion Noise Schedules and Sample Steps are Flawed <https://huggingface.co/papers/2305.08891>`_ 中的表 2。支持以下类型： ``"linspace"``、 ``"leading"`` 或 ``"trailing"``。默认 ``"leading"``。
        - **compute_dtype** (mindspore.dtype, 可选) - 计算过程中使用的数据类型，支持以下类型： ``mstype.float32`` 或 ``mstype.float16``。默认 ``mstype.float32``，表示 ``mindspore.float32``。

    .. py:method:: add_noise(original_samples, noise, timesteps)

        扩散加噪过程。

        参数：
            - **original_samples** (Tensor) - 当前样本。
            - **noise** (Tensor) - 要加入的随机噪声。
            - **timesteps** (Tensor) - 当前在扩散链中的离散时间步。

        返回：
            Tensor，下一步的噪声样本。

    .. py:method:: set_timesteps(num_inference_steps)

        设置推理阶段扩散链使用的离散时间步（需在推理前调用）。

        参数：
            - **num_inference_steps** (int) - 使用预训练模型生成样本时所采用的扩散步数。

        异常：
            - **ValueError** - 当 `num_inference_steps` 大于 `num_train_timesteps` 时抛出。

    .. py:method:: step(model_output, sample, timestep)

        扩散去噪步骤。

        参数：
            - **model_output** (Tensor) - 扩散模型的直接输出。
            - **sample** (Tensor) - 当前扩散过程中生成的样本实例。
            - **timestep** (Tensor) - 扩散链中的当前离散时间步。

        返回：
            Tensor，去噪后得到前一步的样本。

        异常：
            - **NotImplementedError** - 当 `num_inference_steps` 尚未设置时抛出，需在调用本方法前先调用 `set_timesteps`。
            - **NotImplementedError** - 当当前类未实现具体的 `step` 逻辑时抛出。该方法作为接口定义，需在子类中重写实现。