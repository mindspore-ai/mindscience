mindscience.diffuser.DDPMScheduler
====================================

.. py:class:: mindscience.diffuser.DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", variance_type="fixed_small_log", clip_sample=True, clip_sample_range=1.0, thresholding=False, sample_max_value=1.0, dynamic_thresholding_ratio=0.995, rescale_betas_zero_snr=False, timestep_spacing="leading", compute_dtype=mstype.float32)

    `DDPMScheduler` 实现了去噪扩散概率模型DDPM中介绍的去噪过程。更多信息可参考 `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_。

    参数：
        - **num_train_timesteps** (int) - 训练阶段扩散步数。默认 ``1000``。
        - **beta_start** (float) - 推理阶段 beta 起始值。默认 ``0.0001``。
        - **beta_end** (float) - beta 终止值。默认 ``0.02``。
        - **beta_schedule** (str) - ``beta`` 的调度方式，用于将 ``beta`` 区间映射为逐步更新的 ``beta`` 序列。支持以下类型： ``"linear"``、 ``"scaled_linear"`` 或 ``"squaredcos_cap_v2"``。默认 ``"squaredcos_cap_v2"``。
        - **prediction_type** (str) - 扩散调度器预测类型。支持以下类型： ``"epsilon"``（预测噪声）、 ``"sample"``（预测带噪样本）或 ``"v_prediction"``（参见 `Imagen Video <https://imagen.research.google/video/paper.pdf>`_ 论文 2.4 节）。默认 ``"epsilon"``。
        - **variance_type** (str) - 对去噪样本加噪时的方差处理策略，支持以下类型： ``"fixed_small"``、 ``"fixed_small_log"``、 ``"fixed_large"``、 ``"fixed_large_log"``、 ``"learned"`` 或 ``"learned_range"``。默认 ``"fixed_small_log"``。
        - **clip_sample** (bool) - 是否裁剪预测样本以提升数值稳定性。默认 ``True``。
        - **clip_sample_range** (float) - 样本裁剪最大幅值，仅当 ``clip_sample=True`` 生效。默认 ``1.0``。
        - **thresholding** (bool) - 是否使用动态阈值方法。该方法不适用于潜空间扩散模型如 Stable Diffusion。默认 ``False``。
        - **sample_max_value** (float) - 动态阈值上限，仅当 ``thresholding=True`` 生效。默认 ``1.0``。
        - **dynamic_thresholding_ratio** (float) - 动态阈值方法的比率，仅当 ``thresholding=True`` 生效。默认 ``0.995``。
        - **rescale_betas_zero_snr** (bool) - 是否重新缩放 ``betas`` 以使其终止信噪比为零。启用后，模型可生成更亮或更暗的样本，而非仅限于中等亮度样本。该选项与 `offset_noise <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_ 相关。默认 ``False``。
        - **timestep_spacing** (str) - 采样时间步缩放的计算方式。参考 `Common Diffusion Noise Schedules and Sample Steps are Flawed <https://huggingface.co/papers/2305.08891>`_ 中的表 2 了解更多信息。支持以下类型： ``"linspace"`` 、 ``"leading"`` 或 ``"trailing"`` 。默认 ``"leading"``。 
        - **compute_dtype** (mindspore.dtype) - 计算使用的数据类型，支持 ``mstype.float32`` 或 ``mstype.float16``。默认 ``mstype.float32``，表示 ``mindspore.float32``。

    .. py:method:: set_timesteps(num_inference_steps)

        设置 DDPM 推理阶段使用的时间步。

        参数：
            - **num_inference_steps** (int) - 推理阶段的去噪步数。

        异常：
            - **ValueError** - 当 ``num_inference_steps`` 不等于 ``num_train_timesteps`` 时抛出。

    .. py:method:: step(model_output, sample, timestep, predicted_variance=None)

        DDPM 去噪步骤。

        参数：
            - **model_output** (Tensor) - 学习得到的扩散模型的直接输出。
            - **sample** (Tensor) - 扩散过程中当前时间步对应的样本实例。
            - **timestep** (Tensor) - 扩散链中的当前离散时间步。
            - **predicted_variance** (Tensor) - 预测方差。默认 ``None``。

        返回：
            Tensor，前一步的去噪样本。
