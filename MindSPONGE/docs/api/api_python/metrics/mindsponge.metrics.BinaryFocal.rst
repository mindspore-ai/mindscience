mindsponge.metrics.BinaryFocal
========================================

.. py:class:: mindsponge.metrics.BinaryFocal(alpha=0.25, gamma=2., feed_in=False, not_focal=False)

    计算二分类中预测值和真实值之间的焦点损失，详细实现过程参考： `Lin, Tsung-Yi, et al. 'Focal loss for dense object detection' <https://arxiv.org/abs/1708.02002>`_ 。

    .. math::
        \mathrm{FL}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right)
    
    参数：
        - **alpha** (float) - 交叉熵误差使用的权重系数。默认值： ``0.25``。
        - **gamma** (float) - 超参数，调节误差难易程度。默认值： ``2.0``。
        - **feed_in** (bool) - 是否对输入进行转换。默认值： ``False``。
        - **not_focal** (bool) - 是否使用focal误差。默认值： ``False``。

    输入：
        - **prediction** (Tensor) - 模型预测值，shape为 :math:`(batch\_size, ndim)` 。
        - **target** (Tensor) - 标签值，shape为 :math:`(batch\_size, ndim)` 。

    输出：
        Tensor。shape为 :math:`(batch\_size)` 。