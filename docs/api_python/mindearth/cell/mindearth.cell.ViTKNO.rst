mindearth.cell.ViTKNO
=========================

.. py:class:: mindearth.cell.ViTKNO(image_size=(128, 256), patch_size=8, in_channels=1, out_channels=1, encoder_embed_dims=768, encoder_depths=16, mlp_ratio=4,dropout_rate=1., drop_path_rate=0., num_blocks=16, settings="MLP", high_freq=True, encoder_network=False, compute_dtype=mstype.float32)

    ViTKNO是一个基于Koopman理论并结合Vision Transformer结构的深度学习模型。该模型基于KNO神经算子，将原始非线性动力系统映射为线性动力系统，在线性空间进行时间推演。
    有关更多详细信息，请参考论文 `KoopmanLab: machine learning for solving complex physics equations <https://arxiv.org/pdf/2301.01104.pdf>`_ 。

    参数：
        - **image_size** (tuple[int], 可选) - 输入图像的尺寸。默认值： ``(128, 256)`` 。
        - **patch_size** (int, 可选) - 图像的path尺寸。默认值： ``8``。
        - **in_channels** (int, 可选) - 输入中的通道数。默认值： ``1``。
        - **out_channels** (int, 可选) - 输出中的通道数。默认值： ``1``。
        - **encoder_depths** (int, 可选) - encoder层的层数。默认值： ``12``。
        - **encoder_embed_dims** (int, 可选) - encoder层的编码器维度。默认值： ``768``。
        - **mlp_ratio** (int, 可选) - 解码器层的通道数提升比率。默认值： ``4``。
        - **dropout_rate** (float, 可选) - dropout层的速率。默认值： ``1.0``。
        - **drop_path_rate** (float, 可选) - drop path层的速率。默认值： ``0.0``。
        - **num_blocks** (int, 可选) - block层的层数。默认值： ``16``。
        - **settings** (str, 可选) - decoder层第一层结构类型。默认值： ``MLP``。
        - **high_freq** (bool, 可选) - 是否执行高分辨率数据处理。默认值： ``True``。
        - **encoder_network** (bool, 可选) - 是否执行encoder网络。默认值： ``False``。
        - **compute_dtype** (dtype, 可选) - encoder层、decoding_embedding层、decoder层和dense层的数据类型。默认值： ``mstype.float32``。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, feature\_size, image\_height, image\_width)` 的Tensor。

    输出：
        - **output** (Tensor) - shape为 :math:`(batch\_size, patch\_size, embed\_dim)` 的Tensor。其中， :math:`patch\_size = (image\_height * image\_width) / (patch\_size * patch\_size)` 。