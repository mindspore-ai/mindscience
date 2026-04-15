sponge.function.vector_in_pbc
=================================

.. py:function:: sponge.function.vector_in_pbc(vector: Tensor, pbc_box: Tensor, offset: float = -0.5)

    在周期性边界条件下，使向量在 :math:`-0.5 \times box` 到 :math:`0.5 \times box` 的范围内。

    参数：
        - **vector** (Tensor) - 输入的向量，数据类型为float，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，数据类型为float，shape为(B, D)。
        - **offset** (float) - 偏移率。默认值： ``-0.5`` 。

    返回：
        Tensor。diff_in_box，在 :math:`-0.5 \times box` 到 :math:`0.5 \times box` 的范围内的向量。数据类型为float，shape为(B, ..., D)。
