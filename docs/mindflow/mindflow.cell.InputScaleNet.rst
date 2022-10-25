.. py:class:: mindflow.cell.InputScaleNet(input_scale, input_center=None)

    将输入值缩放到指定的区域。

    参数：
        - **iinput_scale** (list) - 输入x/y/t的比例因子。
        - **iinput_center** (Union[list, None]) - 坐标转换的中心位置。默认值：None。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, channels)` 的Tensor。

    输出：
        shape为 :math:`(*, channels)` 的Tensor。
