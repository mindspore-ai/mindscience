mindchemistry.e3.o3.wigner_D
============================

.. py:function:: mindchemistry.e3.o3.wigner_D(l, alpha, beta, gamma)

    SO(3)的Wigner D矩阵表示。
    它满足以下特性:
    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`

    参数：
        - **l** (int) - z展示维度。
        - **alpha** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\alpha`，第三个作用。
        - **beta** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕X轴旋转 :math:`\beta`，第二个作用。
        - **gamma** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\gamma`，第一个作用。

    返回：
        - **output** (Tensor) - 张量，Wigner D矩阵 :math:`D^l(\alpha, \beta, \gamma)`。张量形状 :math:`(2l+1, 2l+1)`。
