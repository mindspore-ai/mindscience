mindscience.sciops.fft.asd_rfft2d
==================================

.. py:function:: mindscience.sciops.fft.asd_rfft2d(*args, **kwargs)

    使用Ascend NPU加速的2D实数到复数FFT变换。

    此函数对实数输入张量执行2D实数快速傅里叶变换，
    针对Ascend NPU硬件加速进行了优化。

    参数：
        - **\*args** - 可变长度参数列表。通常包括：

          - xr (Tensor): 输入实数张量，数据类型为float32，至少为2D。

        - **\*\*kwargs** - 任意关键字参数。

    返回：
        Tuple[Tensor, Tensor]。包含以下内容的元组，

        - yr (Tensor)，输出复数张量的实部，数据类型为float32。
        - yi (Tensor)，输出复数张量的虚部，数据类型为float32。

    异常：
        - **ValueError** - 如果输入张量数据类型不是float32或张量维度少于2。
