mindscience.diffuser.DiffusionScheduler
======================================

.. py:class:: mindscience.diffuser.DiffusionScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True, clip_sample_range=1.0, thresholding=False, sample_max_value=1.0, dynamic_thresholding_ratio=0.995, rescale_betas_zero_snr=False, timestep_spacing="leading", compute_dtype=mstype.float32)

    扩散调度器基类。

    参数：
        - **num_train_timesteps** (int) - 训练阶段使用的扩散步数，默认 ``1000``。
        - **beta_start** (float) - 推理阶段 beta 起始值，默认 ``0.0001``。
        - **beta_end** (float) - beta 终止值，默认 ``0.02``。
        - **beta_schedule** (str) - beta 序列生成策略，可选 ``linear``、``scaled_linear``、``squaredcos_cap_v2``，默认 ``squaredcos_cap_v2``。
        - **prediction_type** (str) - 预测类型：``epsilon``（预测噪声）、``sample``（预测带噪样本）、``v_prediction``（参见 `Imagen Video <https://imagen.research.google/video/paper.pdf>`_ 论文 2.4 节），默认 ``epsilon``。
        - **clip_sample** (bool) - 是否裁剪预测样本以提升数值稳定性，默认 ``True``。
        - **clip_sample_range** (float) - 裁剪最大幅值，仅当 ``clip_sample=True`` 生效，默认 ``1.0``。
        - **thresholding** (bool) - 是否使用动态阈值方法（不适用于潜空间扩散模型如 Stable Diffusion），默认 ``False``。
        - **sample_max_value** (float) - 动态阈值上限，仅当 ``thresholding=True`` 生效，默认 ``1.0``。
        - **dynamic_thresholding_ratio** (float) - 动态阈值分位数比例，仅当 ``thresholding=True`` 生效，默认 ``0.995``。
        - **timestep_spacing** (str) - 推理时间步排列方式，可选 ``linspace``、``leading``、``trailing``，默认 ``leading``。
        - **rescale_betas_zero_snr** (bool) - 是否将 betas 重缩放为终端 SNR 为 0，默认 ``False``。
        - **compute_dtype** - 计算 dtype，可选 ``mstype.float32`` 或 ``mstype.float16``，默认 ``mstype.float32``。

    .. py:method:: set_timesteps(num_inference_steps)

        设置推理阶段扩散链使用的离散时间步（需在推理前调用）。

        参数：
            - **num_inference_steps** (int) - 推理/生成时使用的扩散步数。

        异常：
            - **ValueError** - 当 `num_inference_steps` 大于 `num_train_timesteps` 时抛出。
            - **ValueError** - 当 `timestep_spacing` 不在 ``'linspace'``、``'leading'``、``'trailing'`` 中时抛出。

    .. py:method:: add_noise(original_samples, noise, timesteps)

        扩散加噪过程。

        参数：
            - **original_samples** (Tensor) - 当前样本。
            - **noise** (Tensor) - 要加入的随机噪声。
            - **timesteps** (Tensor) - 当前离散时间步。

        返回：
            - Tensor - 加噪后的样本。

    .. py:method:: step(model_output, sample, timestep)

        扩散去噪一步。

        参数：
            - **model_output** (Tensor) - 扩散模型的直接输出。
            - **sample** (Tensor) - 扩散过程中的当前样本。
            - **timestep** (Tensor) - 当前离散时间步。

        返回：
            - Tensor - 去噪后的样本。

        异常：
            - **NotImplementedError** - 当未先调用 `set_timesteps` 设置 `num_inference_steps` 时抛出。 在调用本方法之前，必须先通过 ``set_timesteps`` 等初始化方法设置推理时间步数。
            - **NotImplementedError** - 当当前类未实现具体的 `step` 逻辑时抛出。该方法作为接口定义，需在子类中重写实现。