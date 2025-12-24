mindscience.sciops.fft.asd_fft2d
=================================

.. py:function:: mindscience.sciops.fft.asd_fft2d(*args, **kwargs)

    使用Ascend NPU加速的2D复数到复数前向FFT变换。

    此函数对复数输入张量执行2D快速傅里叶变换，
    针对Ascend NPU硬件加速进行了优化。

    参数：
        - **\*args** - 可变长度参数列表。通常包括：

          - xr (Tensor): 输入复数张量的实部，数据类型为float32，至少为2D。
          - xi (Tensor): 输入复数张量的虚部，数据类型为float32。

        - **\*\*kwargs** - 任意关键字参数。

    返回：
        Tuple[Tensor, Tensor]。包含以下内容的元组，

        - yr (Tensor)，输出复数张量的实部，数据类型为float32。
        - yi (Tensor)，输出复数张量的虚部，数据类型为float32。

    异常：
        - **ValueError** - 如果输入张量数据类型不是float32或张量维度少于2。
