mindscience.sciops.dft.IRDFTn
==============================

.. py:class:: mindscience.sciops.dft.IRDFTn(shape, dim=None, norm='backward', modes=None, compute_dtype=mstype.float32)

    1/2/3D 离散实数傅里叶逆变换。结果应与 `scipy.fft.irfftn() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.irfftn.html>`_ 相同。

    参数：
        - **shape** (tuple) - 要变换的维度形状，其他维度无需包含。
        - **dim** (tuple) - 要变换的维度。默认：``None``，将变换尾随维度。
        - **norm** (str) - 归一化模式，应为 'forward'、'backward'、'ortho' 之一。默认：``'backward'``，与 torch.fft.irfftn 相同
        - **modes** (Union[tuple, int, None]) - 输出变换轴的长度。`modes` 必须不大于输入 'x' 维度的一半。默认值：``None``。
        - **compute_dtype** (mindspore.dtype) - 输入张量的类型。默认：``mstype.float32``。

    输入：
        - **ar** (Tensor) - 要变换的张量的实部，尾随维度与 `shape` 对齐。
        - **ai** (Tensor) - 要变换的张量的虚部，尾随维度与 `shape` 对齐。

    输出：
        - **br** (Tensor) - 输出的实张量，尾随维度与 `shape` 对齐。
