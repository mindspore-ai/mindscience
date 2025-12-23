mindscience.models.layers.MaskedLayerNorm
===================================================

.. py:class:: mindscience.models.layers.MaskedLayerNorm()

    掩码层归一化。对输入张量应用带掩码的层归一化。

    输入:
        - **act** (Tensor) - 形状为 :math:`(*, in\_channels)` 的张量。
        - **gamma** (Tensor) - 形状为 :math:`(in\_channels,)` 的缩放参数。
        - **beta** (Tensor) - 形状为 :math:`(in\_channels,)` 的偏移参数。
        - **mask** (Tensor, optional) - 形状为 :math:`(*, 1)` 的掩码张量。默认值：``None``。

    输出:
        形状为 :math:`(*, in\_channels)` 的张量。
