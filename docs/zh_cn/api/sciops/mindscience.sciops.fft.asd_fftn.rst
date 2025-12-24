mindscience.sciops.fft.asd_fftn
================================

.. py:function:: mindscience.sciops.fft.asd_fftn(xr, xi, ndim=1)

    使用Ascend NPU加速的N维复数到复数前向FFT变换。

    此函数为1D和2D复数到复数FFT变换提供统一接口，
    针对Ascend NPU硬件加速进行了优化。

    参数：
        - **xr** (Tensor) - 输入复数张量的实部，数据类型为float32。
        - **xi** (Tensor) - 输入复数张量的虚部，数据类型为float32。
        - **ndim** (int, 可选) - 要变换的维度数。仅支持 ``1`` 和 ``2``。默认值：``1``。

    返回：
        Tuple[Tensor, Tensor]。包含以下内容的元组，

        - yr (Tensor)，输出复数张量的实部，数据类型为float32。
        - yi (Tensor)，输出复数张量的虚部，数据类型为float32。
