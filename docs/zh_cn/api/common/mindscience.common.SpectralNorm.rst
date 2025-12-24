mindscience.common.SpectralNorm
================================

.. py:class:: mindscience.common.SpectralNorm(module, n_power_iterations=1, dim=0, eps=1e-12)

    对模块中的参数应用谱归一化，通过权重张量的谱范数来稳定 GAN 判别器（或 critic）训练过程。

    参数：
        - **module** (nn.Cell) - 包含待归一化参数的模块。
        - **n_power_iterations** - 计算谱范数时的幂迭代次数。
        - **dim** - 对应输出数量的维度索引。
        - **eps** - 计算范数时的数值稳定项。

    输入：
        - **input** - 包含位置参数的输入。
        - **kwargs** - 其它关键字参数。

    输出：
        模块的前向推理结果。