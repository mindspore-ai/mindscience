mindchemistry.e3.o3.TensorProduct
=========================================

.. py:class:: mindchemistry.e3.o3.TensorProduct(irreps_in1, irreps_in2=None, irreps_out=None, instructions='full', dtype=float32, irrep_norm='component', path_norm='element', weight_init='normal', weight_mode='inner', core_mode='ncon', ncon_dtype = float32, **kwargs)

    多功能张量乘积运算符，适用于两个输入 `Irreps` 和一个输出 `Irreps`，将两个张量发送到一个张量中并保持几何张量属性。
    该类集成了不同的典型用法：`TensorSquare`、`FullTensorProduct`、`FullyConnectedTensorProduct`、`ElementwiseTensorProduct` 和 `Linear`。

    `TensorProduct` 类定义了一个具有等变性的代数结构。
    一旦创建并初始化了 `TensorProduct` 对象，算法就确定了。对于任何给定的两个合法输入张量，该对象将提供一个输出张量。
    如果对象没有可学习的权重，输出张量是确定性的。
    当引入可学习权重时，该运算符将对应于一个一般的双线性等变操作，作为标准张量乘积的推广。

    如果未指定 `irreps_in2`，它将被分配为 `irreps_in1`，对应于 `TensorSquare`。
    如果未指定 `irreps_out`，该运算符将考虑所有可能的输出不变性。
    如果 `irreps_out` 和 `instructions` 都未指定，则该运算符是没有任何可学习权重的标准张量乘积，对应于 `FullTensorProduct`。

    每个输出 irrep 应满足：

    .. math::
        \| l_1 - l_2 \| \leq l_{out} \leq \| l_1 + l_2 \|
        p_1 p_2 = p_{out}

    参数：
        - **irreps_in1** (Union[str, Irrep, Irreps]) - 第一个输入的 Irreps。
        - **irreps_in2** (Union[str, Irrep, Irreps, None]) - 第二个输入的 Irreps。默认值：``None``。
          如果 `irreps_in2` 为 None，`irreps_in2` 将在 'linear' 指令中分配为 '0e'，否则将分配为 `irreps_in1`，对应于 `TensorSquare`。
        - **irreps_out** (Union[str, Irrep, Irreps, None]) - 在 'connect' 和自定义指令中的输出 Irreps，或其他情况下的输出 Irreps 过滤器。
          如果 `irreps_out` 为 None，`irreps_out` 将是完整的张量乘积 Irreps（包括所有可能的路径）。默认值：``None``。
        - **instructions** (Union[str, List[Tuple[int, int, int, str, bool, (float)]]]) - 张量乘积路径指令列表。默认值：``'full'``。
          对于 `str`，在 {'full', 'connect', 'element', 'linear', 'merge'} 中，根据不同模式自动构建指令：

          - 'full': 每对输入 Irreps 的每个输出 Irreps —— 独立创建并返回输出。输出不会相互混合。
            如果未指定 `irreps_out`，则对应于标准张量乘积 `FullTensorProduct`。
          - 'connect': 每个输出是兼容路径的加权和。允许运算符生成具有任意多重性的输出。
            对应于 `FullyConnectedTensorProduct`。
          - 'element': Irreps 逐个相乘。输入将被拆分，输出的多重性与输入的多重性匹配。
            对应于 `ElementwiseTensorProduct`。
          - 'linear': 在第一个 Irreps 上的线性运算，而第二个 Irreps 设置为 '0e'。这可以看作是几何张量版本的密集层。
            对应于 `Linear`。
          - 'merge': 使用可训练参数自动构建 'uvu' 模式指令。这里的 `irreps_out` 作为输出过滤器。

          对于 `List[Tuple[int, int, int, str, bool, (float)]]`，手动构建指令。

          每个指令包含一个元组：(indice_one, indice_two, i_out, mode, has_weight, (optional: path_weight))。
          每个指令将 ``in1[indice_one]`` :math:`\otimes` ``in2[indice_two]`` 放入 ``out[i_out]``。

          - `indice_one`, `indice_two`, `i_out`: int，对应于 `irreps_in1`、`irreps_in2` 和 `irreps_out` 中 irrep 的索引。
          - `mode`: str，在 {'uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv'} 中，每个路径的多重性的处理方式。'uvw' 是完全混合模式。
          - `has_weight`: bool，如果此路径应具有可学习权重，则为 `True`，否则为 `False`。
          - `path_weight`: float，应用于此路径输出的乘法权重。默认值：1.0。

        - **irrep_norm** (str) - {'component', 'norm'}，假定输入和输出表示的规范化方式。默认值：``'component'``。

          - 'norm': :math:`\| x \| = \| y \| = 1 \Longrightarrow \| x \otimes y \| = 1`

        - **path_norm** (str) - {'element', 'path'}，路径权重的规范化方法。默认值：``'element'``。

          - 'element': 每个输出按元素总数规范化（独立于其路径）。
          - 'path': 每个路径按路径中的元素总数规范化，然后每个输出按路径数目规范化。

        - **weight_init** (str) - {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}，权重的初始化方法。默认值：``'normal'``。
        - **weight_mode** (str) - {'inner', 'share', 'custom'} 确定权重的模式。默认值：``'inner'``。

          - 'inner': 权重将在张量乘积中内部初始化。
          - 'share': 权重应手动给定且无批次维度。
          - 'custom': 权重应手动给定且有批次维度。

        - **dtype** (mindspore.dtype) - 输入张量的类型。默认值：``mindspore.float32``。
        - **ncon_dtype** (mindspore.dtype) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32``。

    输入：
        - **x** (Tensor) - 形状为 ``(..., irreps_in1.dim)`` 的张量。
        - **y** (Tensor) - 形状为 ``(..., irreps_in2.dim)`` 的张量。
        - **weight** (Tensor) - `Tensor` 或 `Tensor` 列表，可选。
          当 ``internal_weights`` 为 ``False`` 时必需。
          当 ``shared_weights`` 为 ``True`` 时，形状为 ``(self.weight_numel,)`` 的张量。
          当 ``shared_weights`` 为 ``False`` 时，形状为 ``(..., self.weight_numel)`` 的张量，
          或形状为 ``weight_shape`` / ``(...) + weight_shape`` 的张量列表。
          使用 ``self.instructions`` 知道所使用的权重。
          形状为 ``(..., irreps_out.dim)`` 的张量。

    输出：
        - **outputs** (Tensor) - 形状为 ``(..., irreps_out.dim)`` 的张量。

    异常：
        - **ValueError** - 如果 `irreps_out` 不合法。
        - **ValueError** - 如果连接模式不在 ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv'] 中。
        - **ValueError** - 如果输入和输出的度数不匹配。
        - **ValueError** - 如果输入和输出的奇偶性不匹配。
        - **ValueError** - 如果输入和输出的多重性不匹配。
        - **ValueError** - 如果连接模式是 'uvw'，但 `has_weight` 为 `False`。
        - **ValueError** - 如果连接模式是 'uuw' 且 `has_weight` 为 `False`，但多重性不等于 1。
        - **ValueError** - 如果初始方法不受支持。
        - **ValueError** - 如果输入张量的数量与输入 Irreps 的数量不匹配。
