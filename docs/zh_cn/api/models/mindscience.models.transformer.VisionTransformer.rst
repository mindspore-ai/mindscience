mindscience.models.transformer.VisionTransformer
===================================================

.. py:class:: mindscience.models.transformer.VisionTransformer(image_size=(192, 384), in_channels=7, out_channels=3, patch_size=16, encoder_depths=12, encoder_embed_dim=768, encoder_num_heads=12, decoder_depths=8, decoder_embed_dim=512, decoder_num_heads=16, dropout_rate=0.0, compute_dtype=mstype.float16)

    此模块基于 VisionTransformer 骨干，包含编码器、解码器嵌入、解码器和密集层。

    参数:
        - **image_size** (tuple[int], 可选) - 输入的图像大小。默认值： ``(192, 384)``。
        - **in_channels** (int, 可选) - 输入的特征大小。默认值： ``7``。
        - **out_channels** (int, 可选) - 输出的特征大小。默认值： ``3``。
        - **patch_size** (int, 可选) - 图像的补丁大小。默认值： ``16``。
        - **encoder_depths** (int, 可选) - 编码器层的深度。默认值： ``12``。
        - **encoder_embed_dim** (int, 可选) - 编码器层的嵌入维度。默认值： ``768``。
        - **encoder_num_heads** (int, 可选) - 编码器层的头数。默认值： ``12``。
        - **decoder_depths** (int, 可选) - 解码器层的深度。默认值： ``8``。
        - **decoder_embed_dim** (int, 可选) - 解码器层的嵌入维度。默认值： ``512``。
        - **decoder_num_heads** (int, 可选) - 解码器层的头数。默认值： ``16``。
        - **dropout_rate** (float, 可选) - dropout 层的速率。默认值： ``0.0``。
        - **compute_dtype** (mindspore.dtype, 可选) - 编码器、解码器嵌入、解码器和密集层的数据类型。默认值： ``mstype.float16``。

    输入:
        - **input** (Tensor) - 形状为 :math:`(batch\_size, feature\_size, image\_height, image\_width)`。

    输出:
        - **output** (Tensor) - 形状为 :math:`(batch\_size, patchify\_size, embed\_dim)`。其中 patchify_size = (image_height * image_width) / (patch_size * patch_size)。
