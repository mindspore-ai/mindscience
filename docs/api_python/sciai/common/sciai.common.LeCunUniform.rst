sciai.common.LeCunUniform
============================================

.. py:class:: sciai.common.LeCunUniform()

    生成一个服从Yan LeCun均匀分布
    :math:`{N}(-\text{boundary}, \text{boundary})` 的随机数组用于初始化Tensor，其中：

    .. math::

        boundary = \sqrt{\frac{3}{fan\_in}}

    `fan_in` 是权重Tensor中输入单元的数量。

    更多关于Yan LeCun均匀分布的细节请参考：
    `Neural Tangent Kernel: Convergence and Generalization in Neural Networks
    <https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html>`_。
