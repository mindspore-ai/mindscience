mindchemistry.e3.o3.Irreps
============================

.. py:class:: mindchemistry.e3.o3.Irreps(irreps)

    O（3）的不可约表示的直和。这个类不包含任何数据，它是一个描述表示的结构。它通常用作库的其他类的参数，以定义函数的输入和输出表示。

    参数:
        - **irreps** (Union[str, Irrep, Irreps, List[Tuple[int]]]) - 表示不可约表示的直和的字符串。

    .. py:method:: mindchemistry.e3.o3.Irreps.count(ir)

        计算该"ir"的多重性。

        参数:
            - **ir** (Irrep) - Irrep。

        返回:
            int，该"ir"的多重性总数。

    .. py:method:: mindchemistry.e3.o3.Irrep.simplify()

        简化Irreps的表达。

        参数:
            - 无。

        返回:
            `Irreps`

    .. py:method:: mindchemistry.e3.o3.Irrep.remove_zero_multiplicities()

        删除任何倍数为零的Irreps。

        参数:
            - 无。

        返回:
            `Irreps`。

    .. py:method:: mindchemistry.e3.o3.Irrep.sort()

        按递增的程度对表达进行排序。

        参数:
            - 无。

        返回:
            irreps（`irreps`）-排序`irreps`。
            p（tuple[int]）-置换顺序 `p[old_index] = new_index`。
            inv（tuple[int]）-反转排列顺序 `p[new_index] = old_index`。

    .. py:method:: mindchemistry.e3.o3.Irreps.filter(keep, drop)

        计算该"ir"的多重性。

        参数:
            - **keep** (Union[str, Irrep, Irreps, List[str, Irrep]]) - 要保留的不可恢复的列表。默认值:None。
            - **drop** (Union[str, Irrep, Irreps, List[str, Irrep]]) - 要删除的不可恢复的列表。默认值:None。

        返回:
            `Irreps`，过滤过的Irreps。

    .. py:method:: mindchemistry.e3.o3.Irreps.decompose(v, batch=False)

        计算该"ir"的多重性。

        参数:
            - **v** (Tensor) - 要分解的向量。
            - **batch** (bool) - 是否重新整形结果，使其至少有一个批维度。默认值:"False"。

        返回:
            张量列表，按"Irreps"分解的向量。

    .. py:method:: mindchemistry.e3.o3.Irreps.spherical_harmonics(lmax, p)

        计算该"ir"的多重性。

        参数:
            - **lmax** (int) - "l"的最大值。
            - **p** (int) - ｛1，-1｝，表示的奇偶性。

        返回:
            `Irreps`，表示:math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`。


    .. py:method:: mindchemistry.e3.o3.Irreps.randn(*size, normalization)

        随机张量。

        参数:
            - ***size**  (List[int]) - 输出张量的大小，需要包含一个'-1'。
            - **normalization** (str) - ｛'component'，'norm'｝，规一化方法的类型。

        返回:
            张量，形状为"size"的张量，其中"-1"被"self.dim"代替。

    .. py:method:: mindchemistry.e3.o3.Irreps.wigD_from_angles(alpha, beta, gamma, k)

        从欧拉角表示O（3）的wigner D矩阵。

        参数:
            - **alpha** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转:math:`\alpha`，作用于第三维。
            - **beta** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕X轴旋转:math:`\beta`，作用于第二维。
            - **gamma** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转:math:`\gamma`，作用于第一维。
            - **k** (Union[None, Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 应用奇偶校验的次数。默认值: ``None``。

        返回:
            O（3）的张量表示wigner D矩阵。形状为:math:`（...，2l+1,2l+1）` 的张量。

    .. py:method:: mindchemistry.e3.o3.Irreps.wigD_from_matrix(R)

        从旋转矩阵表示O（3）的wigner D矩阵。

        参数:
            - **R** (Tensor) - 旋转矩阵。形状为:math:`（...，3，3）` 的张量。

        返回:
            O（3）的张量表示wigner D矩阵。形状为:math:`（...，2l+1,2l+1）` 的张量。