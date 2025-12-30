mindscience.diffuser.DDIMScheduler
===================================

.. py:class:: mindscience.diffuser.DDIMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True, clip_sample_range=1.0, thresholding=False, sample_max_value=1.0, dynamic_thresholding_ratio=0.995, rescale_betas_zero_snr=False, timestep_spacing="leading", compute_dtype=mstype.float32)

    `DDIMScheduler` 实现了去噪扩散概率模型 DDPM 中介绍的去噪过程。具体细节见 `Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502>`_ 。

    参数：
        - **num_train_timesteps** (int) - 训练模型时使用的扩散时间步数。默认 ``1000``。
        - **beta_start** (float) - 噪声控制参数 `beta` 起始值。默认 ``0.0001``。
        - **beta_end** (float) - 噪声控制参数 `beta` 终点值。默认 ``0.02``。
        - **beta_schedule** (str) - beta 调度策略，用于将 beta 区间映射为模型逐步使用的 beta 序列。支持以下类型： ``"squaredcos_cap_v2"`` 、 ``"linear"`` 或 ``"scaled_linear"`` 。默认 ``"squaredcos_cap_v2"``。
        - **prediction_type** (str) - 扩散调度器预测类型。支持以下类型： ``"sample"`` (直接预测加噪样本) 或 ``"v_prediction"``（参考 `Imagen Video <https://imagen.research.google/video/paper.pdf>`_ ）。默认 ``"epsilon"``。
        - **clip_sample** (bool) - 是否为了数值稳定性，裁剪预测的样本。默认 ``True`` 。
        - **clip_sample_range** (float) - 样本裁剪最大幅度。仅当 ``clip_sample=True`` 时有效。默认 ``1.0``。
        - **thresholding** (bool) - 是否采用动态阈值方法。该方法不适用于潜在空间扩散模型，例如 Stable Diffusion。默认 ``False``。
        - **sample_max_value** (float) - 动态阈值方法中的阈值大小。仅当 ``thresholding=True`` 时有效。默认 ``1.0``。
        - **dynamic_thresholding_ratio** (float) - 动态阈值方法中使用的比例参数。仅当 ``thresholding=True`` 时有效。默认 ``0.995``。
        - **rescale_betas_zero_snr** (bool) - 是否对 beta 进行重新缩放以使终止信噪比（SNR）为零。启用后，模型可生成非常明亮或非常暗的样本，而不局限于中等亮度样本。该设置与 `offset_noise <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_ 在概念上存在一定关联。默认 ``False``。
        - **timestep_spacing** (str) - 采样时间步缩放的计算方式。相关说明可参考 `Common Diffusion Noise Schedules and Sample Steps are Flawed <https://huggingface.co/papers/2305.08891>`_ 的表2了解更多信息。支持以下类型： ``"linspace"`` 、 ``"leading"`` 或 ``"trailing"``。默认 ``"leading"``。
        - **compute_dtype** (mindspore.dtype) - 计算使用的数据类型。支持 ``mstype.float32`` 或 ``mstype.float16``。默认 ``mstype.float32``，表示 ``mindspore.float32``。

    .. py:method:: step(model_output, sample, timestep, eta=0.0, use_clipped_model_output=False)

        DDIM反向去噪步骤。

        参数：
            - **model_output** (Tensor) - 学习得到的扩散模型的直接输出。
            - **sample** (Tensor) - 扩散过程中当前时间步对应的样本实例。
            - **timestep** (Tensor) - 扩散链中的当前离散时间步。
            - **eta** (float) - 扩散步骤中附加噪声的权重。当 ``eta=0`` 时为 DDIM，当 ``eta=1`` 时退化为 DDPM。默认 ``0.0``。
            - **use_clipped_model_output** (bool) - 是否基于裁剪后的原始预测样本 ``x_0`` 重新计算噪声 ``epsilon``，以补偿 ``clip_sample`` 引入的偏差。该修正仅在采样阶段生效。若为 ``True``，则从裁剪后的 ``x_0`` 推导 ``epsilon``，并在去噪步骤中使用修正后的噪声，以提升数值稳定性；若为 ``False``，则直接使用原始 ``model_output``，保持模型未调整的预测结果。默认 ``False``。

        返回：
            Tensor，去噪后的输出。

        异常：
            - **ValueError** - 当 ``eta`` 不在区间 ``[0, 1]`` 内时抛出。
