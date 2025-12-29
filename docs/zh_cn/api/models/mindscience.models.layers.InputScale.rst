mindscience.models.layers.InputScale
===================================================

.. py:class:: mindscience.models.layers.InputScale(input_scale, input_center=None)

    基于 :math:`(x_i - input\_center)*input\_scale` 将输入值缩放到指定区域。

    参数：
        - **input_scale** (list) - 输入的缩放系数。
        - **input_center** (Union[list, None]) - 坐标平移的位置偏移。默认值：``None``。

    输入：
        - **input** (Tensor) - 形状为 :math:`(*, channels)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(*, channels)` 的张量。

    异常：
        - **TypeError** - 如果 `input_scale` 不是列表。
        - **TypeError** - 如果 `input_center` 不是列表或 ``None``。
