mindscience.diffuser.DiffusionTrainer
=======================================

.. py:class:: mindscience.diffuser.DiffusionTrainer(model, scheduler, objective='pred_noise', p2_loss_weight_gamma=0, p2_loss_weight_k=1, loss_type='l1')

    扩散训练器基类。

    参数：
        - **model** (nn.Cell) - 扩散模型的主干网络。
        - **scheduler** (DiffusionScheduler) - 扩散调度器，可为 DDPM 或 DDIM 调度器。
        - **objective** (str, 可选) - 调度器函数的预测目标类型，支持以下类型： ``"pred_noise"``（预测扩散过程中的噪声）、 ``"pred_x0"``（预测原始样本）或 ``"pred_v"``（参见 `Imagen Video <https://imagen.research.google/video/paper.pdf>`_ 论文第 2.4 节）。默认 ``"pred_noise"``。
        - **p2_loss_weight_gamma** (float, 可选) - p2 损失权重中的 `gamma` 参数，具体信息查看 `Perception Prioritized Training of Diffusion Models <https://arxiv.org/abs/2204.00227>`_ 。默认 ``0``。
        - **p2_loss_weight_k** (float, 可选) - p2 损失权重中的 k 参数，具体信息查看 `Perception Prioritized Training of Diffusion Models <https://arxiv.org/abs/2204.00227>`_ 。默认 ``1``。
        - **loss_type** (str, 可选) - 损失函数类型。支持以下类型： ``"l1"`` 或 ``"l2"``。默认 ``"l1"``。

    异常：
        - **TypeError** - 如果 `scheduler` 不是 `DiffusionScheduler` 类型时抛出。

    .. py:method:: get_loss(original_samples, noise, timesteps, condition=None)

        计算扩散过程的前向损失。

        参数：
            - **original_samples** (Tensor) - 学习得到的扩散模型的直接输出。
            - **noise** (Tensor) - 扩散过程中生成的当前噪声样本实例。
            - **timesteps** (Tensor) - 扩散链中的当前离散时间步。
            - **condition** (Tensor, 可选) - 用于控制输出结果的条件信息。默认 ``None``。

        返回：
            Tensor，模型的前向损失值。

 