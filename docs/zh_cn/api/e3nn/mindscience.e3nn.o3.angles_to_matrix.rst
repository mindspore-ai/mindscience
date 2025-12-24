mindscience.e3nn.o3.angles_to_matrix
=========================================

.. py:function:: mindscience.e3nn.o3.angles_to_matrix(alpha, beta, gamma)

    将欧拉角 :math:`(\alpha, \beta, \gamma)` 转换为对应的 :math:`3 \times 3` 旋转矩阵。
    结果矩阵表示如下旋转：

    .. math::
        R = R_y(\alpha) \cdot R_x(\beta) \cdot R_y(\gamma)

    参数：
        - **alpha** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - alpha 欧拉角。形状为 :math:`(...)` 的张量。
        - **beta** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - beta 欧拉角。形状为 :math:`(...)` 的张量。
        - **gamma** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - gamma 欧拉角。形状为 :math:`(...)` 的张量。


    返回：
        Tensor，旋转矩阵。输出形状为 :math:`(..., 3, 3)`。
