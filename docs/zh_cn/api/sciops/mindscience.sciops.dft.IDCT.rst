mindscience.sciops.dft.IDCT
============================

.. py:class:: mindscience.sciops.dft.IDCT(shape, compute_dtype=mstype.float32)

    在最后一个轴上对实数进行 1D 离散余弦逆变换。结果应与 `scipy.fft.dct() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct>`_ 相同。
    参考： `A fast cosine transform in one and two dimensions <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163351>`_。

    参数：
        - **shape** (tuple) - 要变换的维度形状，其他维度无需包含。必须是长度为 1 的元组。
        - **compute_dtype** (mindspore.dtype) - 输入张量的类型。默认：``mstype.float32``。

    输入：
        - **a** (Tensor) - 要变换的实张量，尾随维度与 `shape` 对齐。

    输出：
        - **b** (Tensor) - 输出的实张量，尾随维度与 `shape` 对齐。
