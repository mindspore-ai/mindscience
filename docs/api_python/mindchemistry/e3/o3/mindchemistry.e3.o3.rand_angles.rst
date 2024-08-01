mindchemistry.e3.o3.rand_angles
=====================================

.. py:function:: mindchemistry.e3.o3.rand_angles(*shape)

    给出一组随机的欧拉角。

    参数:
        - **shape** (Tuple[int]) - 附加尺寸的形状。

    返回:
        - **alpha** (Tensor) - alpha 欧拉角。
        - **beta** (Tensor) - beta 欧拉角。
        - **gamma** (Tensor) - gamma 欧拉角。

    异常:
        - **TypeError** - 如果 `shape` 的类型不是 tuple。
        - **TypeError** - 如果 `shape` 元素的类型不是 int。