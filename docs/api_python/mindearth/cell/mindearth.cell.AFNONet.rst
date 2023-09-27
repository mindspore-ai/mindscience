mindearth.cell.AFNONet
=========================

.. py:class:: mindearth.cell.AFNONet(image_size=(128, 256), in_channels=1,  out_channels=1, patch_size=8, encoder_depths=12, encoder_embed_dim=768, mlp_ratio=4, dropout_rate=1.0, compute_dtype=mindspore.float32)

    AFNO是一个基于傅立叶神经算子（FNO）并结合Vision Transformer结构的深度学习模型。
    有关更多详细信息，请参考论文 `Adaptive Fourier Neural Operators: Efficient Token Mixers For Transformers <https://arxiv.org/pdf/2111.13587.pdf>`_ 。

    参数：
        - **image_size** (tuple[int]) - 输入图像的尺寸。默认值： (128, 256)。
        - **in_channels** (int) - 输入中的通道数。默认值： ``1``。
        - **out_channels** (int) - 输出中的通道数。默认值： ``1``。
        - **patch_size** (int) - 图像的path尺寸。默认值： ``8``。
        - **encoder_depths** (int) - encoder层的层数。默认值： ``12``。
        - **encoder_embed_dim** (int) - encoder层的编码器维度。默认值： ``768``。
        - **mlp_ratio** (int) - 解码器层的通道数提升比率。默认值： ``4``。
        - **dropout_rate** (float) - dropout层的速率。默认值： ``1.0``。
        - **compute_dtype** (dtype) - encoder层、decoding_embedding层、decoder层和dense层的数据类型。默认值： ``mstype.float32``。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, feature\_size, image\_height, image\_width)` 的Tensor。

    输出：
        Tensor，AFNO网络的输出。

        - **output** (Tensor) - shape为 :math:`(batch\_size, patch\_size, embed\_dim)` 的Tensor。其中， :math:`patch\_size = (image\_height * image\_width) / (patch\_size * patch\_size)` 。