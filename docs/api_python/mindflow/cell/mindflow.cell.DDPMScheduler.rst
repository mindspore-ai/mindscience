mindflow.cell.DDPMScheduler
============================

.. py:class:: mindflow.cell.DDPMScheduler(num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, beta_schedule: str = "squaredcos_cap_v2", prediction_type: str = 'epsilon', variance_type: str = 'fixed_small_log', clip_sample: bool = True, clip_sample_range: float = 1.0, thresholding: bool = False, sample_max_value: float = 1.0, dynamic_thresholding_ratio: float=0.995, rescale_betas_zero_snr: bool = False, timestep_spacing: str = "leading", compute_dtype=mstype.float32)

    `DDPMScheduler` 实现了去噪扩散概率模型DDPM中介绍的去噪过程。具体细节见 `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_ 。

    参数：
        - **num_train_timesteps** (int) - DDPM扩散模型训练步数。默认值: ``1000`` 。
        - **beta_start** (float) - 噪声控制参数 `beta` 起始值。默认值： ``0.0001`` 。
        - **beta_end** (float) - 噪声控制参数 `beta` 终点值。默认值： ``0.02`` 。
        - **beta_schedule** (str) - 噪声控制参数计算方式。默认值： ``squaredcos_cap_v2`` 。支持以下类型： ``squaredcos_cap_v2`` , ``linear`` 和 ``scaled_linear`` 。默认值： ``squaredcos_cap_v2`` 。
        - **prediction_type** (str) - 扩散调度器预测类型。默认值： ``epsilon`` （预测噪声）。支持以下类型： ``sample`` (直接预测加噪样本) 和 ``v_prediction`` （参考 `Imagen Video <https://imagen.research.google/video/paper.pdf>`_ ）。
        - **variance_type** (str) - 在向去噪样本添加噪声时裁剪方差。支持以下类型： ``fixed_small`` , ``fixed_small_log`` , ``fixed_large`` , ``fixed_large_log`` , ``learned`` 和 ``learned_range`` 。默认值： ``fixed_small_log`` 。
        - **clip_sample** (bool) - 是否为了数值稳定性，裁剪预测的样本。默认值： ``True`` 。
        - **clip_sample_range** (float) - 样本裁剪最大幅度。默认值： ``1.0`` 。仅当 `clip_sample=True` 时有效。
        - **thresholding** (bool) - 是否采用动态阈值方法。默认值： ``False`` 。这不适用于潜在空间扩散模型，例如 Stable Diffusion。
        - **sample_max_value** (float) - 动态阈值方法的阈值。默认值： ``1.0`` 。仅当 `thresholding=True` 时有效。
        - **dynamic_thresholding_ratio** (float) - 动态阈值方法的比率。默认值： ``0.995`` 。
        - **timestep_spacing** (str) - 采样时间步缩放的计算方式。参考 `通用的扩散噪声调度器和采样步骤有缺陷 <https://huggingface.co/papers/2305.08891>`_ 了解更多信息。支持以下类型： ``linspace`` , ``leading`` 和 ``trailing`` 。默认值： ``leading`` 。
        - **rescale_betas_zero_snr** (bool) - 是否重新缩放 betas 以使其终端 SNR 为零。这使模型能够生成非常明亮和黑暗的样本，而不是将其限制为中等亮度的样本。与 `offset_noise <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_ 松散相关。默认值： ``False`` 。
        - **compute_dtype** (mindspore.dtype) - 数据类型。默认值： ``mstype.float32`` ，表示 ``mindspore.float32`` 。

    .. py:method:: add_noise(original_samples: Tensor, noise: Tensor, timesteps: Tensor)

        DDPM前向加噪步骤。

        参数：
            - **original_samples** (Tensor) - 样本。
            - **noise** (Tensor) - 随机噪声。
            - **timesteps** (Tensor) - 当前时间步。

        返回：
            Tensor - 加噪样本。

    .. py:method:: set_timesteps(num_inference_steps)

        设置采样步数。

        参数：
            - **num_inference_steps** (int) - 采样步数。

        异常：
            - **ValueError** - 如果 `num_inference_steps` 与 `num_train_timesteps` 不相等。

    .. py:method:: step(model_output, sample, timestep, predicted_variance=None)

        DDPM反向去噪步骤。

        参数：
            - **model_output** (Tensor) - 扩散模型预测的噪声。
            - **sample** (Tensor) - 当前样本。
            - **timestep** (Tensor) - 当前时间步。
            - **predicted_variance** (Tensor) - 预测的方差。默认值： ``None`` 。

        返回：
            Tensor - 去噪样本。
