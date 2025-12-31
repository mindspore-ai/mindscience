mindscience.common.SpectralNorm
================================

.. py:class:: mindscience.common.SpectralNorm(module, n_power_iterations=1, dim=0, eps=1e-12)

    对给定模块中的参数应用谱归一化。谱归一化通过使用谱范数对权重张量进行重标定，从而稳定生成对抗网络（GAN）中判别器（critic）的训练过程。

    参数：
        - **module** (nn.Cell) - 包含目标参数的模块。
        - **n_power_iterations** (int, 可选) - 用于计算谱范数的幂迭代次数。默认 ``1``。
        - **dim** (int, 可选) - 对应输出数量的维度。默认 ``0``。
        - **eps** (float, 可选) - 在计算范数时用于数值稳定性的 epsilon 值。默认 ``1e-12``。

    输入：
        - **input** - 包含位置参数的输入。
        - **kwargs** - 其它关键字参数。

    输出：
        模块的前向推理结果。