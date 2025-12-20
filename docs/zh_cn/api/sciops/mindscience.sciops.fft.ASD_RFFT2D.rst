mindscience.sciops.fft.ASD_RFFT2D
==================================

.. py:class:: mindscience.sciops.fft.ASD_RFFT2D()

    使用Ascend NPU加速的2D实数到复数FFT变换。

    此算子对实数输入张量执行2D实数快速傅里叶变换，
    针对Ascend NPU硬件加速进行了优化。

    输入：
        - **xr** (Tensor) - 输入实数张量，数据类型为float32，至少为2D。

    输出：
        - **yr** (Tensor) - 输出复数张量的实部，数据类型为float32。
        - **yi** (Tensor) - 输出复数张量的虚部，数据类型为float32。

    异常：
        - **ValueError** - 如果输入张量数据类型不是float32或张量维度少于2。
