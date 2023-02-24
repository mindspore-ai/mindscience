mindflow.common.L2
==================

.. py:class:: mindflow.common.L2

    计算L2距离度量，即计算欧式距离。

    创建输入中每个元素 :math:`x` 和目标 :math:`y` 之间的L2度量准则。

    .. math::
        \text{l2} = \sqrt {\sum_{i=1}^n \frac {(y_i - x_i)^2}{y_i^2}}

    这里 :math:`y_i` 是真值， :math:`x_i` 是预测值。

    .. note::
        `update` 方法必须使用 `update(y_pred, y)` 的形式调用。

    .. py:method:: clear()

        清理内部评估结果。

    .. py:method:: eval()

        计算L2度量。

        返回：
            Float，计算结果。

    .. py:method:: update(*inputs)

        更新内部评估结果 :math:`\text{y_pred}` 和 :math:`y`。输入 `y_pred` 和 `y` 用于计算L2。

        参数：
            - **inputs** (Union[Tensor, list, numpy.array]) - `y_pred` 和 `y` 是输入 `input` 中位置为0和1的元素，用于计算L2的预测值和真实值。两者有相同的shape。

        异常：
            - **ValueError** - 如果输入的长度不是2。
            - **ValueError** - 如果y_pred和y的不相同。

