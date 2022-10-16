mindsponge.function.vector_in_box
=================================

.. py:class:: mindsponge.function.vector_in_box(vector, pbc_box)

    在周期性边界条件下，使向量在-0.5box到0.5box的范围内。

    参数：
        - **vector** (Tensor) - 输入的向量，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，shape为(B, D)。

    输出：
        Tensor。diff_in_box，在-0.5box到0.5box的范围内的向量。shape为(B, ..., D)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。