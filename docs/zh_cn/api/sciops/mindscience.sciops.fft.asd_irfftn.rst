mindscience.sciops.fft.asd_irfftn
==================================

.. py:function:: mindscience.sciops.fft.asd_irfftn(xr, xi, n=None, ndim=1)

    使用Ascend NPU加速的N维复数到实数反向FFT变换。

    此函数为1D和2D复数到实数反向FFT变换提供统一接口，
    针对Ascend NPU硬件加速进行了优化。

    参数：
        - **xr** (Tensor) - 输入复数张量的实部，数据类型为float32。
        - **xi** (Tensor) - 输入复数张量的虚部，数据类型为float32。
        - **n** (int, 可选) - 输出张量的长度。默认值：``None``。
        - **ndim** (int, 可选) - 要变换的维度数。仅支持 ``1`` 和 ``2``。默认值：``1``。

    返回：
        yr (Tensor)。输出实数张量，数据类型为float32。
