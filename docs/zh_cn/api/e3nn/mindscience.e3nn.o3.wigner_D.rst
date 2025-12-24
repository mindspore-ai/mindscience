mindscience.e3nn.o3.wigner_D
============================

.. py:function:: mindscience.e3nn.o3.wigner_D(l, alpha, beta, gamma)

    SO(3) 的 Wigner D 矩阵表示。
    这些矩阵描述了具有角动量 `l` 的量子态在三次欧拉角（Z-Y-Z 约定）下如何旋转：

    1. 围绕原始 Z 轴旋转 :math:`\gamma`
    2. 围绕新的 Y 轴旋转 :math:`\beta`
    3. 围绕最新的 Z 轴旋转 :math:`\alpha`

    得到的 :math:`D^l(\alpha,\beta,\gamma)` 是一个 :math:`(2l+1) \times (2l+1)` 的酉矩阵，其元素是著名的 Wigner D 函数。

    它满足以下特性：

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`

    参数：
        - **l** (int) - 表示的阶数。
        - **alpha** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\alpha`，第三个作用。
        - **beta** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕X轴旋转 :math:`\beta`，第二个作用。
        - **gamma** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\gamma`，第一个作用。

    返回：
        Tensor，Wigner D 矩阵 :math:`D^l(\alpha, \beta, \gamma)`。张量形状 :math:`(2l+1, 2l+1)`。
