mindearth.cell.DgmrDiscriminator
=================================

.. py:class:: mindearth.cell.DgmrDiscriminator(in_channels=1, num_spatial_frames=8, conv_type="standard")

    Dgmr判别器基于时间判别器和空间判别器，其中包含深度残差块。
    有关更多详细信息，请参考论文 `Skilful precipitation nowcasting using deep generative models of radar <https://www.nature.com/articles/s41586-021-03854-z>`_ 。

    参数：
        - **in_channels** (int) - 输入中的通道数。默认值： ``1``。
        - **num_spatial_frames** (int) - 待进行空间判别的时间步数。默认值： ``8``。
        - **conv_type** (str) - 卷积核类型。默认值： ``standard``。

    输入：
        - **x** (Tensor) - shape为 :math:`(2, frames\_size, channels, height\_size, width\_size)` 的Tensor。

    输出：
        Tensor，Dgmr Discriminator网络的输出。

        - **output** (Tensor) - shape为 :math:`(2, 2, 1)` 的Tensor。