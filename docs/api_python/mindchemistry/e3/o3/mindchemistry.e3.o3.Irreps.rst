mindchemistry.e3.o3.Irreps
============================

.. py:class:: mindchemistry.e3.o3.Irreps(irreps)

    O(3)的不可约表示的直和。这个类不包含任何数据，它是一个描述表示的结构。它通常用作库的其他类的参数，以定义函数的输入和输出表示。

    参数：
        - **irreps** (Union[str, Irrep, Irreps, List[Tuple[int]]]) - 表示不可约表示的直和的字符串。

    异常：
        - **ValueError** - 如果 `irreps` 无法转换为 `Irreps`。
        - **ValueError** - 如果 `irreps` 的 mul 部分为负。
        - **TypeError** - 如果 `irreps` 的 mul 部分不是 int 类型。

    .. py:method:: count(ir)

        计算该"ir"的多重性。

        参数：
            - **ir** (Irrep) - 不可约表示。

        返回：
            int，该"ir"的多重性总数。

    .. py:method:: simplify()

        简化Irreps的表示。

        返回：
            `Irreps`

    .. py:method:: remove_zero_multiplicities()

        删除任何多重性为零的Irreps。

        返回：
            `Irreps`。

    .. py:method:: sort()

        按度数对表示进行递增排序。

        返回：
            - **irreps** (`Irreps`) - 排序后的 `Irreps`。
            - **p** (tuple[int]) - 置换顺序 `p[old_index] = new_index`。
            - **inv** (tuple[int]) - 反转排列顺序 `p[new_index] = old_index`。

    .. py:method:: filter(keep, drop)

        通过 `keep` 或 `drop` 过滤 `Irreps`。

        参数：
            - **keep** (Union[str, Irrep, Irreps, List[str, Irrep]]) - 要保留的 irrep 列表。默认值: ``None``。
            - **drop** (Union[str, Irrep, Irreps, List[str, Irrep]]) - 要删除的 irrep 列表。默认值: ``None``。

        返回：
            `Irreps`，过滤后的 irreps。

        异常：
            - **ValueError** - 如果 `keep` 和 `drop` 都不为 `None`。

    .. py:method:: decompose(v, batch=False)

        通过 `Irreps` 对向量进行分解。

        参数：
            - **v** (Tensor) - 要分解的向量。
            - **batch** (bool) - 是否重塑结果，使其至少有一个批次维度。默认值: ``False``。

        返回：
            Tensors 列表，通过 `Irreps` 分解后的向量。

        异常：
            - **TypeError** - 如果 `v` 不是 Tensor。
            - **ValueError** - 如果向量 `v` 的长度与 `Irreps` 的维度不匹配。

    .. py:method:: spherical_harmonics(lmax, p)

        球面谐波的表示。

        参数：
            - **lmax** (int) - `l` 的最大值。
            - **p** (int) - {1, -1}，表示的奇偶性。

        返回：
            `Irreps`，表示 :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`。

    .. py:method:: randn(*size, normalization)

        随机张量。

        参数：
            - ***size**  (List[int]) - 输出张量的大小，需要包含一个'-1'。
            - **normalization** (str) - ｛'component'，'norm'｝，规一化方法的类型。

        返回：
            张量，形状为"size"，其中"-1"被"self.dim"代替。

    .. py:method:: wigD_from_angles(alpha, beta, gamma, k)

        从欧拉角计算 O(3) 的 Wigner D 矩阵表示。

        参数：
            - **alpha** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\alpha`，作用于第三维。
            - **beta** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕X轴旋转 :math:`\beta`，作用于第二维。
            - **gamma** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\gamma`，作用于第一维。
            - **k** (Union[None, Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 应用奇偶校验的次数。默认值: ``None``。

        返回：
            O(3)的张量表示wigner D矩阵。形状为 :math:`(..., 2l+1, 2l+1)` 的张量。

    .. py:method:: wigD_from_matrix(R)

        从旋转矩阵中得到 O(3) 的 Wigner D 矩阵表示。

        参数：
            - **R** (Tensor) - 旋转矩阵。形状为 :math:`(..., 3, 3)` 的张量。

        返回：
            O(3)的张量表示wigner D矩阵。形状为 :math:`(..., 2l+1, 2l+1)` 的张量。

        异常：
            - **TypeError** - 如果 `R` 不是张量。