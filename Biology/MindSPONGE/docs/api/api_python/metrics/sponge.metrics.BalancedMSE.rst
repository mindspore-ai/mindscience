sponge.metrics.BalancedMSE
============================

.. py:class:: sponge.metrics.BalancedMSE(first_break, last_break, num_bins, beta=0.99, reducer_flag=False)
    
    计算预测值与真实值之间的平衡均方误差（Balanced MSE），以解决回归任务中的不平衡标签问题。

    参考 `Ren, Jiawei, et al. 'Balanced MSE for Imbalanced Visual Regression' <https://arxiv.org/abs/2203.16427>`_ 。

    .. math::
        L =-\log \mathcal{N}\left(\boldsymbol{y} ; \boldsymbol{y}_{\text {pred }},
        \sigma_{\text {noise }}^{2} \mathrm{I}\right)
        +\log \sum_{i=1}^{N} p_{\text {train }}\left(\boldsymbol{y}_{(i)}\right)
        \cdot \mathcal{N}\left(\boldsymbol{y}_{(i)} ; \boldsymbol{y}_{\text {pred }},
        \sigma_{\text {noise }}^{2} \mathrm{I}\right)

    参数：
        - **first_break** (float) - 箱线的起始值。
        - **last_break** (float) - 箱线的结束值。
        - **num_bins** (int) - 箱线数量。
        - **beta** (float，可选) - 移动平均系数，默认值为 ``0.99``。
        - **reducer_flag** (bool，可选) - 是否聚合多个设备的标签值，默认值为 ``False``。

    输入：
        - **prediction** (Tensor) - 预测值，shape为 :math:`(batch\_size, ndim)`。
        - **target** (Tensor) - 真实标签值，shape为 :math:`(batch\_size, ndim)`。

    输出：
        Tensor，shape为 :math:`(batch\_size, ndim)`。
