mindscience.e3nn.o3.Irrep
============================

.. py:class:: mindscience.e3nn.o3.Irrep(l, p=None)

    O(3) 的不可约表示。这个类不包含任何数据，它是描述表示的结构。
    通常用作库中其他类的参数，以定义函数的输入和输出表示。
    不可约表示由非负整数 `l` （阶数）和奇偶性 `p` （偶为 1，奇为 -1）标注。常用别名：`e` 表示偶性、`o` 表示奇性、`y` 表示按 \((-1)^l\) 的奇偶。

    参数：
        - **l** (Union[int, str]) - 非负整数，表示的阶数，:math:`l = 0, 1, \dots`。或者用字符串同时编码阶数与奇偶性（例如 ``"1o"``、``"2e"``，其中 `e`/`o`/`y` 各表示上述别名）。
        - **p** (int, 可选) - 表示的奇偶性，:math:`p \in \{1, -1\}`。当 ``l`` 为字符串时忽略此参数。默认值: ``None``。

    异常：
        - **NotImplementedError** - 如果方法未实现。
        - **ValueError** - 如果 `l` 为负数或 `p` 不在 {1, -1} 中。
        - **ValueError** - 如果 `l` 不能转换为 `Irrep`。
        - **TypeError** - 如果 `l` 不是 int 或 str。

    .. py:method:: wigD_from_angles(alpha, beta, gamma, k=None)

        根据三个描述旋转顺序的欧拉角 :math:`(\alpha, \beta, \gamma)` 计算 O(3) 的 Wigner-D 矩阵表示：

        1. 绕原始 Y 轴旋转 :math:`\gamma`。
        2. 绕新的 X 轴旋转 :math:`\beta`。
        3. 绕最新的 Y 轴旋转 :math:`\alpha`。

        参数：
            - **alpha** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\alpha`，第三个作用。
            - **beta** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕X轴旋转 :math:`\beta`，第二个作用。
            - **gamma** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\gamma`，第一个作用。
            - **k** (Union[None, Tensor[float32], list[float], tuple[float], ndarray[np.float32], float], 可选) - 应用奇偶校验的次数。默认值: ``None``。

        返回：
            Tensor，O(3) 的 Wigner D 矩阵表示。张量形状为 :math:`(..., 2l+1, 2l+1)` 。

    .. py:method:: wigD_from_matrix(R)

        从旋转矩阵中得到 O(3) 的 Wigner D 矩阵表示。

        参数：
            - **R** (Tensor) - 旋转矩阵。形状为 :math:`(..., 3, 3)` 的张量。

        返回：
            Tensor，O(3) 的 Wigner D 矩阵表示。张量形状为 :math:`(..., 2l+1, 2l+1)` 。

        异常：
            - **TypeError** - 如果 `R` 不是张量。