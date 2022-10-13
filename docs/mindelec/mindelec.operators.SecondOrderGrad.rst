mindelec.operators.SecondOrderGrad
==================================

.. py:class:: mindelec.operators.SecondOrderGrad(model, input_idx1, input_idx2, output_idx)

    计算并返回指定输出列相对于指定输入列的二阶梯度。

    参数：
        - **model** (Cell) - 接受单个Tensor输入并返回单个Tensor的函数或网络。
        - **input_idx1** (int) - 指定输入的列索引，以生成一阶导数。取值范围为[0，网络输入维度 - 1]。
        - **input_idx2** (int) - 指定输入的列索引，以生成二阶导数。取值范围为[0，网络输入维度 - 1]。
        - **output_idx** (int) - 指定输出的列索引。取值范围为[0，网络输出维度 - 1]。

    输入：
        - **input** - 给定函数或网络 `model` 的输入。

    输出：
        Tensor。

    异常：
        - **TypeError** - 如果 `input_idx1` 、 `input_idx2` 或 `output_idx` 的类型不是int。
