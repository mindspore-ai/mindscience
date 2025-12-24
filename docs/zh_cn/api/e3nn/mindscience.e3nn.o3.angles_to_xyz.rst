mindscience.e3nn.o3.angles_to_xyz
=========================================

.. py:function:: mindscience.e3nn.o3.angles_to_xyz(alpha, beta)

    将两个球面角 :math:`(\alpha, \beta)` 转换为单位球面上的笛卡尔坐标 :math:`(x, y, z)`。

    参数：
        - **alpha** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - alpha 欧拉角。形状为 :math:`(...)` 的张量。
        - **beta** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - beta 欧拉角。形状为 :math:`(...)` 的张量。

    返回：
        Tensor，点 :math:`(x, y, z)`。形状为 :math:`(..., 3)`。
