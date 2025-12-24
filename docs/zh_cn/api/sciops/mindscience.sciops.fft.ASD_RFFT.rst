mindscience.sciops.fft.ASD_RFFT
================================

.. py:class:: mindscience.sciops.fft.ASD_RFFT()

    使用Ascend NPU加速的1D实数到复数FFT变换。

    此算子对实数输入张量执行1D实数快速傅里叶变换，
    针对Ascend NPU硬件加速进行了优化。

    输入：
        - **xr** (Tensor) - 输入实数张量，数据类型为float32。

    输出：
        - **yr** (Tensor) - 输出复数张量的实部，数据类型为float32。
        - **yi** (Tensor) - 输出复数张量的虚部，数据类型为float32。

    异常：
        - **ValueError** - 如果输入张量数据类型不是float32。
