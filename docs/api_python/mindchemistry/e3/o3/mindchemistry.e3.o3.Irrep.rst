mindchemistry.e3.o3.Irrep
============================

.. py:class:: mindchemistry.e3.o3.Irrep(l, p)

    O(3) 的不可约表示。这个类不包含任何数据，它是描述表示的结构。
    通常用作库中其他类的参数，以定义函数的输入和输出表示。

    参数：
        - **l** (Union[int, str]) - 非负整数，表示的阶数，:math:`l = 0, 1, \dots`。或者用字符串表示阶数和奇偶性。
        - **p** (int) - ｛1，-1｝， 表示的奇偶性， 默认值: ``None``。

    异常：
        - **NotImplementedError** - 如果方法未实现。
        - **ValueError** - 如果 `l` 为负数或 `p` 不在 {1, -1} 中。
        - **ValueError** - 如果 `l` 不能转换为 `Irrep`。
        - **TypeError** - 如果 `l` 不是 int 或 str。

    .. py:method:: wigD_from_angles(alpha, beta, gamma, k)

        从欧拉角计算 O(3) 的 Wigner D 矩阵表示。

        参数：
            - **alpha** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转:math:`\alpha`，第三个作用。
            - **beta** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕X轴旋转:math:`\beta`，第二个作用。
            - **gamma** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转:math:`\gamma`，第一个作用。
            - **k** (Union[None, Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 应用奇偶校验的次数。默认值: ``None``。

        返回：
            张量，O(3) 的 Wigner D 矩阵表示。张量形状为 :math:`(..., 2l+1, 2l+1)` 。

    .. py:method:: wigD_from_matrix(R)

        从旋转矩阵中得到 O(3) 的 Wigner D 矩阵表示。

        参数：
            - **R** (Tensor) - 旋转矩阵。形状为:math:`(..., 3, 3)` 的张量。

        返回：
            O(3)的张量表示wigner D矩阵。形状为:math:`(..., 2l+1, 2l+1)` 的张量。

        异常：
            - **TypeError** - 如果 `R` 不是张量。