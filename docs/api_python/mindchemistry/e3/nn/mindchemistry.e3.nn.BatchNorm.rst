mindchemistry.e3.nn.BatchNorm
===============================

.. py:class:: mindchemistry.e3.nn.BatchNorm(irreps, eps=1e-5, momentum=0.1, affine=True, reduce='mean', instance=False, normalization='component', dtype=float32)

    正交表示的批归一化。
    通过表示的范数进行归一化。
    注意，范数仅对正交表示是不变的。
    不可约表示 `wigner_D` 是正交的。

    参数：
        - **irreps** (Union[str, Irrep, Irreps]) - 输入的不可约表示。
        - **eps** (float) - 归一化方差时避免除以零的值。默认值：``1e-5``。
        - **momentum** (float) - 滑动平均的动量。默认值：``0.1``。
        - **affine** (bool) - 是否包含权重和偏置参数。默认值：``True``。
        - **reduce** (str) - {'mean', 'max'}，用于归约的方法。默认值：``'mean'``。
        - **instance** (bool) - 应用实例归一化而不是批归一化。默认值：``False``。
        - **normalization** (str) - {'component', 'norm'}，归一化方法。默认值：``'component'``。
        - **dtype** (mindspore.dtype) - 输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **inputs** (Tensor) - 形状为 :math:`(batch, ..., irreps.dim)` 的张量。

    输出：
        - **outputs** (Tensor) - 形状为 :math:`(batch, ..., irreps.dim)` 的张量。

    异常：
        - **ValueError**: 如果 `reduce` 不是 'mean' 或 'max'。
        - **ValueError**: 如果 `normalization` 不是 'component' 或 'norm'。
