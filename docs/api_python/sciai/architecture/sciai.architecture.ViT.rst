sciai.architecture.ViT
=========================

.. py:class:: sciai.architecture.ViT(image_size=(192, 384), in_channels=7, out_channels=3, patch_size=16, encoder_depths=12, encoder_embed_dim=768, encoder_num_heads=12, decoder_depths=8, decoder_embed_dim=512, decoder_num_heads=16, mlp_ratio=4, dropout_rate=1.0, dtype=ms.float16)

    该模块基于ViT，包括encoder层、decoding_embedding层、decoder层和dense层。

    参数：
        - **image_size** (tuple[int]) - 输入的图像尺寸。默认值： ``(192,384)``。
        - **in_channels** (int) - 输入的输入特征维度。默认值： ``7``。
        - **out_channels** (int) - 输出的输出特征维度。默认值： ``3``。
        - **patch_size** (int) - 图像的path尺寸。默认值： ``16``。
        - **encoder_depths** (int) - encoder层的层数。默认值： ``12``。
        - **encoder_embed_dim** (int) - encoder层的编码器维度。默认值： ``768``。
        - **encoder_num_heads** (int) - encoder层的head数。默认值： ``12``。
        - **decoder_depths** (int) - decoder层的解码器深度。默认值： ``8``。
        - **decoder_embed_dim** (int) - decoder层的解码器维度。默认值： ``512``。
        - **decoder_num_heads** (int) - decoder层的head数。默认值： ``16``。
        - **mlp_ratio** (int) - mlp层的比例。默认值： ``4``。
        - **dropout_rate** (float) - dropout层的速率。默认值： ``1.0``。
        - **dtype** (dtype.Number) - encoder层、decoding_embedding层、decoder层和dense层的数据类型。默认值： ``ms.float16``。

    输入：
        - **input** (Tensor) - shape为 :math:`(batch\_size, feature\_size, image\_height, image\_width)` 的Tensor。

    输出：
        - **output** (Tensor) - shape为 :math:`(batch\_size, patchify\_size, embed\_dim)` 的Tensor。其中，patchify_size = (image_height * image_width) / (patch_size * patch_size)
