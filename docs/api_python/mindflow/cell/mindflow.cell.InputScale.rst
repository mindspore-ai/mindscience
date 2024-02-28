mindflow.cell.InputScale
===========================

.. py:class:: mindflow.cell.InputScale(input_scale, input_center=None)

    将输入值缩放到指定的区域。按照 :math:`(x_i - input\_center)*input\_scale` 变换。

    参数：
        - **input_scale** (list) - 输入的比例因子。
        - **input_center** (Union[list, None]) - 坐标转换的中心位置，理解为偏移量。默认值： ``None``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, channels)` 的Tensor。

    输出：
        shape为 :math:`(*, channels)` 的Tensor。
    
    异常：
        - **TypeError** - 如果 `input_scale` 不是list类型。
        - **TypeError** - 如果 `input_center` 不是list类型或者 ``None``。
        