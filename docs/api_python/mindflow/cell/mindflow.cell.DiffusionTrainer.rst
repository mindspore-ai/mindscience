mindflow.cell.DiffusionTrainer
================================

.. py:class:: mindflow.cell.DiffusionTrainer(model, scheduler, objective='pred_noise', p2_loss_weight_gamma=0., p2_loss_weight_k=1, loss_type='l1')

    扩散模型训练控制实现。

    参数：
        - **model** (nn.Cell) - 用于扩散模型训练的骨干网络。
        - **scheduler** (DiffusionScheduler) - 用于训练的噪声控制器。
        - **objective** (str) - 扩散模型预测结果的类型。默认值： ``pred_noise`` 。支持以下类型： ``pred_noise`` , ``pred_v`` 和 ``pred_x0`` 。
        - **p2_loss_weight_gamma** (float) - p2 loss权重 `gamma` ，具体信息查看 `Perception Prioritized Training of Diffusion Models <https://arxiv.org/abs/2204.00227>`_ 。默认值： ``0.0`` 。
        - **p2_loss_weight_k** (float) - p2 loss权重 `k` ，具体信息查看 `Perception Prioritized Training of Diffusion Models <https://arxiv.org/abs/2204.00227>`_ 。默认值： ``1`` 。
        - **loss_type** (str) - loss函数类型。默认值: ``l1`` 。支持以下类型： ``l1`` 和 ``l2`` 。

    .. py:method:: get_loss(original_samples, noise, timesteps, condition=None)

        计算扩散过程的前向loss。

        参数：
            - **original_samples** (Tensor) - 原始样本。
            - **noise** (Tensor) - 随机噪声。
            - **timesteps** (Tensor) - 时间步。
            - **condition** (Tensor) - 控制条件。默认值: ``None`` 。

        返回：
            - Tensor - 前向loss。
