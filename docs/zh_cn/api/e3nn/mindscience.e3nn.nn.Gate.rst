mindscience.e3nn.nn.Gate
============================

.. py:class:: mindscience.e3nn.nn.Gate(irreps_scalars, acts, irreps_gates, act_gates, irreps_gated, dtype=mindspore.float32, ncon_dtype=mindspore.float32)

    门控激活函数。输入被概念性地分为三部分：
    
    1. `irreps_scalars`：仅由对应的激活函数 `acts` 逐元素变换的标量；
    2. `irreps_gates`：经激活函数 `act_gates` 逐元素变换后作为“门”使用的标量；
    3. `irreps_gated`：被第二部分的门标量逐通道相乘调制的不可约表示。
    
    数学上，门控激活函数的定义为：

    .. math::
        \left( \bigoplus_i \phi_i(x_i) \right) \oplus \left(\bigoplus_j \phi_j(g_j) y_j \right)

    其中：

    * :math:`x_i` 和 :math:`\phi_i` 来自 `irreps_scalars` 和 `acts`；
    * :math:`g_j`、:math:`\phi_j` 和 :math:`y_j` 分别对应 `irreps_gates`、`act_gates` 和 `irreps_gated`。

    输出的不可约表示为激活后的标量与门控后的不可约表示的拼接，保持整体的等变性性质。

    参数：
        - **irreps_scalars** (Union[str, Irrep, Irreps]) - 将通过激活函数 `acts` 的输入标量不可约表示。
        - **acts** (list[Func]) - 对 `irreps_scalars` 的每部分应用的激活函数列表。`acts` 的长度将被剪切或填充为恒等函数，以匹配 `irreps_scalars` 的长度。
        - **irreps_gates** (Union[str, Irrep, Irreps]) - 将通过激活函数 `act_gates` 并与 `irreps_gated` 相乘的输入标量不可约表示。
        - **act_gates** (list[Func]) - 每个 `irreps_gates` 部分的激活函数列表。 `acts` 的长度将被剪切或填充为恒等函数，以匹配 `irreps_gates` 的长度。
        - **irreps_gated** (Union[str, Irrep, Irreps]) - 将被门控的输入不可约表示。
        - **dtype** (mindspore.dtype，可选) - 输入张量的类型。默认值：``mindspore.float32`` 。
        - **ncon_dtype** (mindspore.dtype，可选) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **input** (Tensor) - 形状为 :math:`(..., irreps\_in.dim)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(..., irreps\_out.dim)` 的张量。

    异常：
        - **ValueError**: 如果 `irreps_scalars` 或 `irreps_gates` 包含非标量的不可约表示。
        - **ValueError**: 如果 `irreps_gates` 的总乘积不匹配 `irreps_gated` 的总乘积。