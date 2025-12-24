mindscience.e3nn.nn.BatchNorm
===============================

.. py:class:: mindscience.e3nn.nn.BatchNorm(irreps, eps=1e-5, momentum=0.1, affine=True, reduce='mean', instance=False, normalization='component', dtype=mindspore.float32)

    针对正交归一化群表示的批归一化。
    与常规 BatchNorm 不同，本层按不可约表示块的“不变范数”进行归一化，保证在诸如旋转等群作用下的等变性得以保持。统计量按每个重复（multiplicity）块独立计算，保持张量结构不变。
    范数仅对正交归一化表示是不变的。不可约表示 `wigner_D` （及其对应的实基）满足该条件，因此对标准 `e3nn` 表示是安全的。

    参数：
        - **irreps** (Union[str, Irrep, Irreps]) - 输入的不可约表示。
        - **eps** (float，可选) - 归一化方差时避免除以零的值。默认值：``1e-5``。
        - **momentum** (float，可选) - 滑动平均的动量。默认值：``0.1``。
        - **affine** (bool，可选) - 是否包含权重和偏置参数。默认值：``True``。
        - **reduce** (str，可选) - {'mean', 'max'}，用于归约的方法。默认值：``'mean'``。
        - **instance** (bool，可选) - 应用实例归一化而不是批归一化。默认值：``False``。
        - **normalization** (str，可选) - {'component', 'norm'}，归一化方法。默认值：``'component'``。
        - **dtype** (mindspore.dtype，可选) - 输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **input** (Tensor) - 形状为 :math:`(batch, ..., irreps.dim)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(batch, ..., irreps.dim)` 的张量。

    异常：
        - **ValueError**: 如果 `reduce` 不是 'mean' 或 'max'。
        - **ValueError**: 如果 `normalization` 不是 'component' 或 'norm'。
