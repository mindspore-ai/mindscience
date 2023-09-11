sciai.common.LeCunNormal
============================================

.. py:class:: sciai.common.LeCunNormal()

    生成一个服从Yan LeCun正态分布
    :math:`{N}(0, \text{sigma}^2)` 的随机数组用于初始化Tensor，其中：

    .. math::

        sigma = \sqrt{\frac{1}{fan\_in}}

    `fan_in` 是权重Tensor中输入单元的数量。

    更多关于Yan LeCun正态分布的细节请参考：
    `Neural Tangent Kernel: Convergence and Generalization in Neural Networks
    <https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html>`_。
