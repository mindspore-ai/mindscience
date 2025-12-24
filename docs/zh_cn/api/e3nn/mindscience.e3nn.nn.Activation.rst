mindscience.e3nn.nn.Activation
===============================

.. py:class:: mindscience.e3nn.nn.Activation(irreps_in, acts, dtype=mindspore.float32)

    标量不可约表示（l=0）的激活函数。每个不可约表示的奇偶性会随所用激活函数的奇偶性变化。
    偶性标量（ `0e` ）保持奇偶性不变；奇性标量（ `0o` ）在应用偶函数时会翻转为偶性（ `0e` ），
    只有在应用奇函数时才保持奇性（ `0o` ）。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的不可约表示。
        - **acts** (list[Func]) - 用于 `irreps_in` 每部分的激活函数列表。 `acts` 的长度将被剪切或填充为恒等函数，以匹配 `irreps_in` 的长度。
        - **dtype** (mindspore.dtype，可选) - 输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **inputs** (Tensor) - 形状为 :math:`(*, irreps\_in.dim)` 的张量。

    输出：
        - **outputs** (Tensor) - 形状为 :math:`(*, irreps\_in.dim)` 的张量。

    异常：
        - **ValueError**: 如果 `irreps_in` 包含非标量的不可约表示。
        - **ValueError**: 如果 `irreps_in` 中的一个不可约表示是奇性，但相应的激活函数既不是奇性也不是偶性。
