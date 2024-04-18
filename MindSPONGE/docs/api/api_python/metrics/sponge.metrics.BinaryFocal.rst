sponge.metrics.BinaryFocal
============================

.. py:class:: sponge.metrics.BinaryFocal(alpha=0.25, gamma=2.0, feed_in=False, not_focal=False)
    
    计算预测值和真实值之间的二元分类焦点误差。

    参考 `Lin, Tsung-Yi, et al. 'Focal loss for dense object detection' <https://arxiv.org/abs/1708.02002>`_ 。

    .. math::
        \mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^{\gamma}
        \log \left(p_{\mathrm{t}}\right)

    参数：
        - **alpha** (float，可选) - 交叉熵的权重，默认值为 ``0.25``。
        - **gamma** (float，可选) - 超参数，调节从难到易的损失，默认值为 ``2.0``。
        - **feed_in** (bool，可选) - 是否转换预测值，默认值为 ``False``。
        - **not_focal** (bool，可选) - 是否使用焦点损失，默认值为 ``False``。

    输入：
        - **prediction** (Tensor) - 预测值，shape为 :math:`(batch\_size, ndim)`。
        - **target** (Tensor) - 标签值，shape为 :math:`(batch\_size, ndim)`。

    输出：
        Tensor， shape为 :math:`(batch\_size,)`。
