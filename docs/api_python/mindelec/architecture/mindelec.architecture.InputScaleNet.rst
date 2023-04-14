mindelec.architecture.InputScaleNet
===================================

.. py:class:: mindelec.architecture.InputScaleNet(input_scale, input_center=None)

    将输入值缩放到指定的区域。

    参数：
        - **input_scale** (list) - 输入的比例因子，其维度需要与输入维度相同。
        - **input_center** (Union[list, None]) - 坐标转换的中心位置。默认值： ``None``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, channels)` 的Tensor。

    输出：
        shape为 :math:`(*, channels)` 的Tensor。

    异常：
        - **TypeError** - 如果 `input_scale` 不是list。
        - **TypeError** - 如果 `input_center` 不是list或 ``None``。
