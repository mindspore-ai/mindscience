mindsponge.metrics.MultiClassFocal
========================================

.. py:class:: mindsponge.metrics.MultiClassFocal(num_class, beta=0.99, gamma=2., e=0.1, neighbors=2, not_focal=False, reducer_flag=False)

    计算多分类中预测值和真实值之间的焦点损失，详细实现过程参考： `Lin, Tsung-Yi, et al. 'Focal loss for dense object detection' <https://arxiv.org/abs/1708.02002>`_ 。
    
    参数：
        - **num_class** (int) - 分类类别数。
        - **beta** (float) - 滑动平均的系数。默认值： ``0.99``。
        - **gamma** (float) - 超参数。默认值： ``2.0``。
        - **e** (float) - 比例系数，focal误差占比。默认值： ``0.1``。
        - **neighbors** (int) - 标签中需要mask的邻居数。默认值： ``2``。
        - **not_focal** (bool) - 是否使用focal误差。默认值： ``False``。
        - **reducer_flag** (bool) - 是否对多卡的标签值做聚合。默认值： ``False``。

    输入：
        - **prediction** (Tensor) - 模型预测值，shape为 :math:`(batch\_size, ndim)` 。
        - **target** (Tensor) - 标签值，shape为 :math:`(batch\_size, ndim)` 。

    输出：
        Tensor。shape为 :math:`(batch\_size, )` 。