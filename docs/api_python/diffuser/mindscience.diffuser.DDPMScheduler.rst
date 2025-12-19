mindscience.diffuser.DDPMScheduler
=================================

.. py:class:: mindscience.diffuser.DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", variance_type="fixed_small_log", clip_sample=True, clip_sample_range=1.0, thresholding=False, sample_max_value=1.0, dynamic_thresholding_ratio=0.995, timestep_spacing="leading", rescale_betas_zero_snr=False, compute_dtype=mstype.float32)

    `DDPMScheduler` 是 DDPM 中去噪过程的实现。
    更多信息可参考 `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_。

    参数：
        - **num_train_timesteps** (int) - 训练阶段扩散步数，默认 ``1000``。
        - **beta_start** (float) - 推理阶段 beta 起始值，默认 ``0.0001``。
        - **beta_end** (float) - beta 终止值，默认 ``0.02``。
        - **beta_schedule** (str) - beta 参数序列生成策略，可选 ``linear``、``scaled_linear``、``squaredcos_cap_v2``，默认 ``squaredcos_cap_v2``。
        - **prediction_type** (str) - 扩散调度器预测类型：``epsilon``（预测噪声）、``sample``（预测带噪样本）、``v_prediction``（参见 `Imagen Video <https://imagen.research.google/video/paper.pdf>`_ 论文 2.4 节），默认 ``epsilon``。
        - **variance_type** (str) - 对去噪样本加噪时的方差处理策略，可选 ``fixed_small``、``fixed_small_log``、``fixed_large``、``fixed_large_log``、``learned``、``learned_range``，默认 ``fixed_small_log``。
        - **clip_sample** (bool) - 是否裁剪预测样本以提升数值稳定性，默认 ``True``。
        - **clip_sample_range** (float) - 样本裁剪最大幅值，仅当 ``clip_sample=True`` 生效，默认 ``1.0``。
        - **thresholding** (bool) - 是否使用动态阈值方法（不适用于潜空间扩散模型如 Stable Diffusion），默认 ``False``。
        - **sample_max_value** (float) - 动态阈值上限，仅当 ``thresholding=True`` 生效，默认 ``1.0``。
        - **dynamic_thresholding_ratio** (float) - 动态阈值方法的比率，仅当 ``thresholding=True`` 生效，默认 ``0.995``。
        - **timestep_spacing** (str) - 采样时间步缩放的计算方式。参考 `通用的扩散噪声调度器和采样步骤有缺陷 <https://huggingface.co/papers/2305.08891>`_ 了解更多信息。支持以下类型： ``linspace`` , ``leading`` 和 ``trailing`` 。默认值： ``leading`` 。
        - **rescale_betas_zero_snr** (bool) - 是否重新缩放 betas 以使其终端 SNR 为零。这使模型能够生成非常明亮和黑暗的样本，而不是将其限制为中等亮度的样本。与 `offset_noise <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_ 松散相关。默认值： ``False`` 。
        - **compute_dtype** (mindspore.dtype) - 计算数据类型，可为 ``mstype.float32`` 或 ``mstype.float16``，默认 ``mstype.float32``（即 ``mindspore.float32``）。

    .. py:method:: add_noise(original_samples: Tensor, noise: Tensor, timesteps: Tensor)

        DDPM前向加噪步骤。

        参数：
            - **original_samples** (Tensor) - 样本。
            - **noise** (Tensor) - 随机噪声。
            - **timesteps** (Tensor) - 当前时间步。

        返回：
            Tensor - 加噪样本。

    .. py:method:: set_timesteps(num_inference_steps)

        设置 DDPM 推理阶段使用的时间步。

        参数：
            - **num_inference_steps** (int) - 去噪步数（The denoising step number）。

        异常：
            - **ValueError** - 当 `num_inference_steps` 不等于 `num_train_timesteps` 时抛出。

    .. py:method:: step(model_output, sample, timestep, predicted_variance=None)

        执行 DDPM 单步反向去噪。

        参数：
            - **model_output** (Tensor) - 扩散模型的直接输出。
            - **sample** (Tensor) - 扩散过程中的当前样本。
            - **timestep** (Tensor) - 当前离散时间步。
            - **predicted_variance** (Tensor) - 预测方差，默认 ``None``。

        返回：
            - Tensor - 上一步的样本（the sample of last step）。
