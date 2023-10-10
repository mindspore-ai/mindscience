mindearth.cell.DgmrGenerator
=================================

.. py:class:: mindearth.cell.DgmrGenerator(forecast_steps=18, in_channels=1, out_channels=256, conv_type="standard", latent_channels=768, context_channels=384, generation_steps=1)

    Dgmr 生成器基于 Conditional_Stack、Latent_Stack、Upsample_Stack 和 ConvGRU，其中包含深度残差块。
    有关更多详细信息，请参考论文 `Skilful precipitation nowcasting using deep generative models of radar <https://www.nature.com/articles/s41586-021-03854-z>`_ 。

    参数：
        - **forecast_steps** (int) - 待预测的步数。默认值： ``18``。
        - **in_channels** (int) - 输入张量的通道数。默认值： ``1``。
        - **out_channels** (int) - 输出张量的通道数。默认值： ``256``。
        - **conv_type** (str) - 卷积核类型。默认值： ``standard``。
        - **latent_channels** (int) - 网络隐变量的通道数。默认值： ``768``。
        - **context_channels** (int) - 上下文信息的通道数。默认值： ``384``。
        - **generation_steps** (int) - 前向生成的样本数目。默认值： ``1``。


    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, input\_frames, channels, height\_size, width\_size)` 的Tensor。

    输出：
        Tensor，Dgmr Generator网络的输出。

        - **output** (Tensor) - shape为 :math:`(batch\_size, output\_frames, channels, height\_size, width\_size)` 的Tensor。
