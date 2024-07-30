mindchemistry.e3.nn.Activation
===============================

.. py:class:: mindchemistry.e3.nn.Activation(irreps_in, acts, dtype=float32)

    标量张量的激活函数。根据每个激活函数的奇偶性可能改变不可约表示的奇偶性。
    奇数标量需要对应的激活函数是奇数或偶数。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的不可约表示。
        - **acts** (List[Func]) - 用于 `irreps_in` 每部分的激活函数列表。 `acts` 的长度将被剪切或填充为恒等函数，以匹配 `irreps_in` 的长度。
        - **dtype** (mindspore.dtype): 输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **inputs** (Tensor) - 形状为 :math:`(*, irreps_in.dim)` 的张量。

    输出：
        - **outputs** (Tensor) - 形状为 :math:`(*, irreps_in.dim)` 的张量。

    异常：
        - **ValueError**: 如果 `irreps_in` 包含非标量的不可约表示。
        - **ValueError**: 如果 `irreps_in` 中的一个不可约表示是奇性，但相应的激活函数既不是奇性也不是偶性。
