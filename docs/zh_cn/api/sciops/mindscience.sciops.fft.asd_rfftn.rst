mindscience.sciops.fft.asd_rfftn
=================================

.. py:function:: mindscience.sciops.fft.asd_rfftn(xr, ndim=1)

    使用Ascend NPU加速的N维实数到复数FFT变换。

    此函数为1D和2D实数到复数FFT变换提供统一接口，
    针对Ascend NPU硬件加速进行了优化。

    参数：
        - **xr** (Tensor) - 输入实数张量，数据类型为float32。
        - **ndim** (int, 可选) - 要变换的维度数。仅支持 ``1`` 和 ``2``。默认值：``1``。

    返回：
        Tuple[Tensor, Tensor]。包含以下内容的元组，

        - yr (Tensor)，输出复数张量的实部，数据类型为float32。
        - yi (Tensor)，输出复数张量的虚部，数据类型为float32。
