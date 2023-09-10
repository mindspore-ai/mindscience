sciai.operators.JacobianWeights
======================================

.. py:class:: sciai.operators.JacobianWeights(model, out_shape, out_type=ms.float32)

    关于权重的雅可比矩阵。输入元组中的最后一个输入Tensor是权重Parameter，其余为网络的输入。

    参数：
        - **model** (Cell) - 用于计算关于权重的雅可比的网络。
        - **out_shape** (tuple) - 网络输出的形状。
        - **out_type** (type) - Mindspore 数据类型。 默认值：ms.float32。

    输入：
        - **\*x** (tuple[Tensor]) - 网络的输入，和用于求雅可比矩阵的权重。

    输出：
        Tensor，网络输出关于指定权重的雅可比矩阵。