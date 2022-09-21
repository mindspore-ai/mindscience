mindelec.common.L2
===================

.. py:class:: mindelec.common.L2

    计算L2度量。

    创建输入中每个元素 :math:`x` 和目标 :math:`y` 之间的L2度量准则 。

    .. math::
        \text{l2} = \sqrt {\sum_{i=1}^n \frac {(y_i - x_i)^2}{y_i^2}}

    这里 :math:`y_i` 是真值， :math:`x_i` 是预测值。

    .. note::
        `update` 方法必须使用 `update(y_pred, y)` 的形式调用。

    .. py:method:: mindelec.common.L2.clear()

        清理内部评估结果。

    .. py:method:: mindelec.common.L2.eval()

        计算L2度量。

        返回：
            Float，计算结果。

    .. py:method:: mindelec.common.L2.update(*inputs)

        更新内部评估结果 :math:`y_{pred}` 和 :math:`y`。

        参数：
            输入 `y_pred` 和 `y` 用于计算L2，其中 `y_pred` 和 `y` 一致。输入数据类型必须是tensor、list或numpy.array。

        异常：
            - **ValueError** - 如果输入的长度不是2。
            - **ValueError** - 如果y_pred和y的不相同。

