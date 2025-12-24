mindscience.sciops.fft.ASD_IRFFT
=================================

.. py:class:: mindscience.sciops.fft.ASD_IRFFT()

    使用Ascend NPU加速的1D复数到实数反向FFT变换。

    此算子对复数输入张量执行1D反向实数快速傅里叶变换，
    针对Ascend NPU硬件加速进行了优化。

    输入：
        - **xr** (Tensor) - 输入复数张量的实部，数据类型为float32。
        - **xi** (Tensor) - 输入复数张量的虚部，数据类型为float32。

    输出：
        - **yr** (Tensor) - 输出实数张量，数据类型为float32。

    异常：
        - **ValueError** - 如果输入张量数据类型不是float32。
