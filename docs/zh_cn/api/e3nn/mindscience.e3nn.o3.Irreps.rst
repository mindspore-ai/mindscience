mindscience.e3nn.o3.Irreps
============================

.. py:class:: mindscience.e3nn.o3.Irreps(irreps=None)

    O(3) 的不可约表示的直和。这个类不包含任何数据，它是一个描述表示的结构。
    它通常用作库的其他类的参数，以定义函数的输入和输出表示。
    不可约表示以 `_MulIr` 元组的形式存储，每个元素包含一个多重数（multiplicity）与一个 `Irrep`，便于进行加法、乘法与过滤等操作。

    参数：
        - **irreps** (Union[str, Irrep, Irreps, list[tuple[int]]], 可选) - 表示不可约表示直和的输入，可为字符串、`Irrep`、`Irreps` 或二元组列表。默认值：``None``。

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

    .. py:method:: decompose(v, batch=False)

        根据当前的 `Irreps` 结构，将向量分解为不可约分量。

        该方法将输入张量 `v` 的最后一个轴重塑，使得每个切片对应 `self` 中列出的一个不可约表示。
        结果列表中的每个元素是一个张量，其形状为 `(..., multiplicity, irrep_dimension)`。

        参数：
            - **v** (Tensor) - 要分解的向量。
            - **batch** (bool, 可选) - 是否重塑结果，使其至少有一个批次维度。默认值: ``False``。

        返回：
            Tensor 列表，通过 `Irreps` 分解后的向量。

        异常：
            - **TypeError** - 如果 `v` 不是 Tensor。
            - **ValueError** - 如果向量 `v` 的长度与 `Irreps` 的维度不匹配。

    .. py:method:: filter(keep=None, drop=None)

        通过 `keep` 或 `drop` 过滤 `Irreps`。

        参数：
            - **keep** (Union[str, Irrep, Irreps, list[str, Irrep]], 可选) - 要保留的 irrep 列表。默认值: ``None``。
            - **drop** (Union[str, Irrep, Irreps, list[str, Irrep]], 可选) - 要删除的 irrep 列表。默认值: ``None``。

        返回：
            `Irreps`，过滤后的 irreps。

        异常：
            - **ValueError** - 如果 `keep` 和 `drop` 都不为 `None`。

    .. py:method:: randn(*size, normalization='component')

        生成一个随机张量，其最后一维与这些不可约表示的总维度相匹配。
        利用 irreps 结构将最后一轴拆分为各个不可约块，
        每个块可按“分量”或“不可约范数”两种方式之一进行归一化。

        参数：
            - **size**  (list[int]) - 输出张量的大小，需要包含一个'-1'。
            - **normalization** (str, 可选) - ｛'component'，'norm'｝，规一化方法的类型。默认值: ``'component'``。

        返回：
            Tensor，形状为 "size"，其中 "-1" 被 "self.dim" 代替。
        
        异常：
            - **ValueError** - 如果 `normalization` 不是 'component' 或 'norm'。

    .. py:method:: remove_zero_multiplicities()

        删除任何多重性为零的Irreps。

        返回：
            `Irreps`，删除多重性为零的 irreps。

    .. py:method:: simplify()

        简化Irreps的表示。

        返回：
            `Irreps`，简化后的 irreps。

    .. py:method:: sort()

        按度数对表示进行递增排序。

        返回：
            - **irreps** (`Irreps`) - 排序后的 `Irreps`。
            - **p** (tuple[int]) - 置换顺序 `p[old_index] = new_index`。
            - **inv** (tuple[int]) - 反转排列顺序 `p[new_index] = old_index`。

    .. py:method:: spherical_harmonics(lmax, p=-1)
        :staticmethod:

        球面谐波的表示。

        参数：
            - **lmax** (int) - `l` 的最大值。
            - **p** (int, 可选) - {1, -1}，表示的奇偶性。默认值: ``-1``。

        返回：
            `Irreps`，表示 :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`。

    .. py:method:: wigD_from_angles(alpha, beta, gamma, k=None)

        根据三个欧拉角 :math:`(\alpha, \beta, \gamma)` 计算 O(3) 的 Wigner-D 矩阵表示，旋转顺序如下：

        1. 绕原始 Y 轴旋转 :math:`\gamma`。
        2. 绕新的 X 轴旋转 :math:`\beta`。
        3. 绕最新的 Y 轴旋转 :math:`\alpha`。

        返回结果是该 `Irreps` 对象中每个不可约表示对应的 Wigner-D 矩阵的直和，按多重数重复。

        参数：
            - **alpha** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\alpha`，作用于第三维。
            - **beta** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕X轴旋转 :math:`\beta`，作用于第二维。
            - **gamma** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕Y轴旋转 :math:`\gamma`，作用于第一维。
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