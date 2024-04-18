sponge.metrics.MultiClassFocal
===============================

.. py:class:: sponge.metrics.MultiClassFocal(num_class, beta=0.99, gamma=2.0, e=0.1, neighbors=2, not_focal=False, reducer_flag=False)
    
    计算预测值和真实值之间的多类别焦点误差。

    参考 `Lin, Tsung-Yi, et al. 'Focal loss for dense object detection' <https://arxiv.org/abs/1708.02002>`_ 。

    参数：
        - **num_class** (int) - 类别数量。
        - **beta** (float，可选) - 移动平均系数，默认值为 ``0.99``。
        - **gamma** (float，可选) - 超参数，默认值为 ``2.0``。
        - **e** (float，可选) - 焦点损失的比例，默认值为 ``0.1``。
        - **neighbors** (int，可选) - 目标中要屏蔽的邻居数，默认 ``2``。
        - **not_focal** (bool，可选) - 是否使用焦点损失，默认值为 ``False``。
        - **reducer_flag** (bool，可选) - 是否聚合多个设备的标签值，默认值为 ``False``。

    输入：
        - **prediction** (Tensor) - 预测值，shape为 :math:`(batch\_size, ndim)`。
        - **target** (Tensor) - 标签值，shape为 :math:`(batch\_size, ndim)`。

    输出：
        Tensor，shape为 :math:`(batch\_size,)`。
