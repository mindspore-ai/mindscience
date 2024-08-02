mindchemistry.e3.o3.SphericalHarmonics
=========================================

.. py:class:: mindchemistry.e3.o3.SphericalHarmonics(irreps_out, normalize, normalization, irreps_in)

    返回球面谐波层。

    参数：
        - **irreps_out** (Union[str, `Irreps`]) - 球面谐波输出的不可约表示。
        - **normalize** (bool) - 是否将输入张量归一化为之前位于球体上的单位向量投影到球面谐波上。
        - **normalization** (str) - ｛‘integral’，‘component’，‘norm’｝，输出张量的归一化方法。默认值: ``"integral"``。
        - **irreps_in** (Union[str, `Irreps`, None]) - 球面谐波输入的不可约表示。默认值: ``None``。
        - **dtype** (mindspore.dtype) - 输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        张量，SphericalHarmonics网络的输入。

        - **x** (Tensor) - 构造球面谐波的张量。形状为 :math:`(..., 3)` 的张量。

    输出：
        张量，SphericalHarmonics网络的输出。

        - **output** (Tensor) - 张量,球面谐波 :math:`Y^l(x)`。形状为: ``(..., 2l+1)``。

    异常：
        - **ValueError** - 如果 `normalization` 不在 {'integral', 'component', 'norm'} 中。
        - **ValueError** - 如果球谐函数的 `irreps_in` 既不是向量 (`1x1o`) 也不是伪向量 (`1x1e`)。
        - **ValueError** - 如果球谐函数的 `irreps_out` 的 `l` 和 `p` 与 `irreps_in` 不一致。输出的奇偶性应为 p = {input_p**l}。
        - **NotImplementedError** - 如果 `l` 大于 11。