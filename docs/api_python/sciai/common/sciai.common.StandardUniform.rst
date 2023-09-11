sciai.common.StandardUniform
============================================

.. py:class:: sciai.common.StandardUniform()

    生成一个服从标准均匀分布
    :math:`{N}(0, \text{sigma}^2)` 的随机数组用于初始化Tensor，其中：

    .. math::

        boundary = \sqrt{\frac{1}{fan\_in}}

    `fan_in` 是权重Tensor中输入单元的数量。