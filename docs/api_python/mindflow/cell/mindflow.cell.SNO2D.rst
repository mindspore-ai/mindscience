mindflow.cell.SNO2D
=========================

.. py:class:: mindflow.cell.SNO2D(in_channels, out_channels, hidden_channels=64, num_sno_layers=3, data_format="channels_first", transforms=None, kernel_size=5, num_usno_layers=0, num_unet_strides=1, activation="gelu", compute_dtype=mstype.float32)

    二维谱神经算子，包含一个提升层（编码器）、多个谱变换层（谱空间的线性变换）和一个投影层（解码器）。参见基类文档 :class:`mindflow.cell.SNO`。
