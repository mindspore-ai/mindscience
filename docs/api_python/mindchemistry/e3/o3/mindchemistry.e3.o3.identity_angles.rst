mindchemistry.e3.o3.identity_angles
=========================================

.. py:function:: mindchemistry.e3.o3.identity_angles(*shape, dtype=float32)

    返回欧拉角的单位集合。

    参数：
        - **shape** (Tuple[int]): 附加维度的形状。
        - **dtype** (mindspore.dtype): 输入张量的类型。默认值：``mindspore.float32``。

    返回：
        - **alpha** (Tensor): alpha 欧拉角。
        - **beta** (Tensor): beta 欧拉角。
        - **gamma** (Tensor): gamma 欧拉角。

    异常：
        - **TypeError**: 如果 'shape' 不是元组类型。
        - **TypeError**: 如果 'shape' 中的元素不是整型。

