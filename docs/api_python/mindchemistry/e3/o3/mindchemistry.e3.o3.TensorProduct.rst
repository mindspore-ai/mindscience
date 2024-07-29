mindchemistry.e3.o3.TensorProduct
=========================================

.. py:class:: mindchemistry.e3.o3.TensorProduct(irreps_in1, irreps_in2, filter_ir_out, ncon_dtype, **kwargs)

    两个输入 `Irreps` 和一个输出 `Irreps` 的多功能张量积运算符，它将两个张量转换为一个张量并保持几何张量属性。此类集成了不同的典型用法：`TensorSquare`、`FullTensorProduct`、`FullyConnectedTensorProduct`、`ElementwiseTensorProduct` 和 `Linear`。
    `TensorProduct` 类定义了一种具有等变性的代数结构。一旦 `TensorProduct` 对象被创建和初始化，算法将被确定。对于任何给定的两个合法输入张量，该对象将提供一个输出张量。如果对象没有可学习的权重，则输出张量是确定性的。当引入可学习的权重时，此运算符将对应于一个通用的双线性等变操作，作为标准张量积的推广。
    如果未指定 `irreps_in2`，它将被分配为 `irreps_in1`，对应于 `TensorSquare`。如果未指定 `irreps_out`，此运算符将考虑所有可能的输出 Irreps。如果同时未指定 `irreps_out` 和 `instructions`，则此运算符是没有任何可学习权重的标准张量积，对应于 ``FullTensorProduct``。
    每个输出 irrep 应满足：

    .. math::
        \| l_1 - l_2 \| \leq l_{out} \leq \| l_1 + l_2 \|
        p_1 p_2 = p_{out}

    参数：
        - **irreps_in1** (Union[str, Irrep, Irreps]): 第一个输入的 Irreps。
        - **irreps_in2** (Union[str, Irrep, Irreps, None]): 第二个输入的 Irreps。默认值: ``None``。如果 `irreps_in2` 为 None，`irreps_in2` 将被分配为 '0e' 在 'linear' 指令中，或者在其他情况下分配为 `irreps_in1`，对应于 `TensorSquare`。
        - **irreps_out** (Union[str, Irrep, Irreps, None]): 在 'connect' 和自定义指令中，输出的 Irreps，或在其他情况下过滤输出的 Irreps。如果 `irreps_out` 为 None，`irreps_out` 将是完整张量积的 Irreps（包括所有可能的路径）。默认值: ``None``。
        - **instructions** (Union[str, List[Tuple[int, int, int, str, bool, (float)]]]): 张量积路径指令列表。默认值: ``'full'``。对于 `str` 在 {'full', 'connect', 'element', 'linear', 'merge'} 中，指令将根据不同模式自动构造：

            - 'full': 每对输入 Irreps 的每个输出 Irrep 都独立创建并返回。输出不会互相混合。如果未指定 `irreps_out`，对应于标准张量积 `FullTensorProduct`。
            - 'connect': 每个输出是兼容路径的学习加权和。这允许运算符生成具有任意倍数的输出。对应于 `FullyConnectedTensorProduct`。
            - 'element': Irreps 一对一相乘。输入将被拆分，输出的多重性与输入的多重性匹配。对应于 `ElementwiseTensorProduct`。
            - 'linear': 对第一个 Irreps 进行线性操作等变，而第二个 Irreps 设置为 '0e'。这可以看作是几何张量版本的稠密层。对应于 `Linear`。
            - 'merge': 自动构建 'uvu' 模式的指令，具有可训练参数。`irreps_out` 在这里充当输出过滤器。

          对于 `List[Tuple[int, int, int, str, bool, (float)]]`，指令由手动构造。每个指令包含一个元组：(indice_one, indice_two, i_out, mode, has_weight, (optional: path_weight))。每个指令将 ``in1[indice_one]`` :math:`\otimes` ``in2[indice_two]`` 放入 ``out[i_out]``。

            - `indice_one`, `indice_two`, `i_out`: int，`irreps_in1`、`irreps_in2` 和 `irreps_out` 中的 Irrep 的索引。
            - `mode`: str，{'uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv'}，每个路径的多重性如何处理。'uvw' 是完全混合模式。
            - `has_weight`: bool，如果此路径应具有可学习权重，则为 `True`，否则为 `False`。
            - `path_weight`: float，应用于此路径输出的乘法权重。默认值: ``1.0``。

        - **irrep_norm** (str): {'component', 'norm'}，输入和输出表示的假定标准化。默认值: ``'component'``。
            - 'norm': :math:` \| x \| = \| y \| = 1 \Longrightarrow \| x \otimes y \| = 1`
        - **path_norm** (str): {'element', 'path'}，路径权重的标准化方法。默认值: ``'element'``。
            - 'element': 每个输出由元素的总数（独立于路径）进行标准化。
            - 'path': 每条路径由路径中的元素总数进行标准化，然后每个输出由路径数进行标准化。
        - **weight_init** (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}，权重的初始化方法。默认值: ``'normal'``。
        - **weight_mode** (str): {'inner', 'share', 'custom'} 确定权重的模式。默认值: ``'inner'``。
            - 'inner': 权重将在张量积内部初始化。
            - 'share': 权重应手动给出且没有批量维度。
            - 'custom': 权重应手动给出且有批量维度。
        - **dtype** (mindspore.dtype): 输入张量的类型。默认值: ``mindspore.float32``。
        - **ncon_dtype** (mindspore.dtype): `ncon` 计算模块的输入张量的类型。默认值: ``mindspore.float32``。

    输入：
        - **x** (Tensor) - 形状为 ``(..., irreps_in1.dim)`` 的张量。
        - **y** (Tensor) - 形状为 ``(..., irreps_in2.dim)`` 的张量。
        - **weight** (Tensor) - `Tensor` 或 `Tensor` 列表，可选。如果 ``internal_weights`` 为 ``False``，则需要。形状为 ``(self.weight_numel,)`` 的张量如果 ``shared_weights`` 为 ``True``；形状为 ``(..., self.weight_numel)`` 的张量如果 ``shared_weights`` 为 ``False``；或形状为 ``weight_shape`` / ``(...) + weight_shape`` 的张量列表。使用 ``self.instructions`` 来知道所使用的权重。形状为 ``(..., irreps_out.dim)`` 的张量。

    输出：
        - **outputs** (Tensor) - 形状为 ``(..., irreps_out.dim)`` 的张量。

    异常：
        - **ValueError**: 如果 `irreps_out` 不合法。
        - **ValueError**: 如果连接模式不在 ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv'] 中。
        - **ValueError**: 如果输入和输出的度数不匹配。
        - **ValueError**: 如果输入和输出的奇偶性不匹配。
        - **ValueError**: 如果输入和输出的多重性不匹配。
        - **ValueError**: 如果连接模式是 'uvw'，但 `has_weight` 为 `False`。
        - **ValueError**: 如果连接模式是 'uuw' 且 `has_weight` 为 `False`，但多重性不等于 1。
        - **ValueError**: 如果初始方法不受支持。
        - **ValueError**: 如果输入张量的数量与输入 Irreps 的数量不匹配。
