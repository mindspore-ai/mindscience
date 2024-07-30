mindchemistry.e3.o3.angles_to_xyz
=========================================

.. py:function:: mindchemistry.e3.o3.angles_to_xyz(alpha, beta)

    将 :math:`(\alpha, \beta)` 转换为球体上的点 :math:`(x, y, z)`。

    参数:
        - **alpha** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - alpha 欧拉角。形状为:math:`(...)` 的张量。
        - **beta** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - beta 欧拉角。形状为:math:`(...)` 的张量。

    返回:
        - **output** (Tensor) - 点:math:`(x, y, z)`。形状为:math:`(..., 3)` 的张量。
