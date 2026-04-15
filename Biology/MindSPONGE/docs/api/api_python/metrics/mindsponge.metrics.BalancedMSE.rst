mindsponge.metrics.BalancedMSE
========================================

.. py:class:: mindsponge.metrics.BalancedMSE(first_break, last_break, num_bins, beta=0.99, reducer_flag=False)

    计算预测值和真实值之间的均衡平方误差，适用于回归任务中标签不平衡的场景。详细实现过程参考： `Ren, Jiawei, et al. 'Balanced MSE for Imbalanced Visual Regression' <https://arxiv.org/abs/2203.16427>`_ 。

    .. math::
        L =-\log \mathcal{N}(\boldsymbol{y} ; \boldsymbol{y}_{\text {pred }}, \sigma_{\text {noise }}^{2} \mathrm{I})+\log \sum_{i=1}^{N} p_{\text {train }}(\boldsymbol{y}_{(i)}) \cdot \mathcal{N}(\boldsymbol{y}_{(i)} ; \boldsymbol{y}_{\text {pred }}, \sigma_{\text {noise }}^{2} \mathrm{I})

    参数：
        - **first_break** (float) - bin划分的起始位置。
        - **last_break** (float) - bin划分的结束位置。
        - **num_bins** (int) - 划分bin的数目。
        - **beta** (float) - 滑动平均的系数。默认值： ``0.99``。
        - **reducer_flag** (bool) - 是否对多卡的标签值做聚合。默认值： ``False``。

    输入：
        - **prediction** (Tensor) - 模型预测值，shape为 :math:`(batch\_size, ndim)` 。
        - **target** (Tensor) - 标签值，shape为 :math:`(batch\_size, ndim)` 。

    输出：
        Tensor。shape为 :math:`(batch\_size, ndim)` 。