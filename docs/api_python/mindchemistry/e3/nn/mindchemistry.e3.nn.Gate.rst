mindchemistry.e3.nn.Gate
============================

.. py:class:: mindchemistry.e3.nn.Gate(irreps_scalars, acts, irreps_gates, act_gates, irreps_gated, dtype=float32, ncon_dtype=float32)

    门控激活函数。输入包含三部分：第一部分 `irreps_scalars` 是只受激活函数 `acts` 影响的标量；第二部分 `irreps_gates` 是受激活函数 `act_gates` 影响并与第三部分相乘的标量。

    .. math::
        \left(\bigoplus_i \phi_i(x_i) \right) \oplus \left(\bigoplus_j \phi_j(g_j) y_j \right)

    其中 :math:`x_i` 和 :math:`\phi_i` 来自 `irreps_scalars` 和 `acts`，而 :math:`g_j`、:math:`\phi_j` 和 :math:`y_j`
    来自 `irreps_gates`、`act_gates` 和 `irreps_gated`。

    参数：
        - **irreps_scalars** (Union[str, Irrep, Irreps]) - 将通过激活函数 `acts` 的输入标量不可约表示。
        - **acts** (List[Func]) - 对 `irreps_scalars` 的每部分应用的激活函数列表。`acts` 的长度将被剪切或填充为恒等函数，以匹配 `irreps_scalars` 的长度。
        - **irreps_gates** (Union[str, Irrep, Irreps]) - 将通过激活函数 `act_gates` 并与 `irreps_gated` 相乘的输入标量不可约表示。
        - **act_gates** (List[Func]) - 每个 `irreps_gates` 部分的激活函数列表。 `acts` 的长度将被剪切或填充为恒等函数，以匹配 `irreps_gates` 的长度。
        - **irreps_gated** (Union[str, Irrep, Irreps]) - 将被门控的输入不可约表示。
        - **dtype** (mindspore.dtype): 输入张量的类型。默认值：``mindspore.float32`` 。
        - **ncon_dtype** (mindspore.dtype) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **inputs** (Tensor) - 形状为 :math:`(..., irreps_in.dim)` 的张量。

    输出：
        - **outputs** (Tensor) - 形状为 :math:`(..., irreps_out.dim)` 的张量。

    异常：
        - **ValueError**: 如果 `irreps_scalars` 或 `irreps_gates` 包含非标量的不可约表示。
        - **ValueError**: 如果 `irreps_gates` 的总乘积不匹配 `irreps_gated` 的总乘积。