mindsponge.function.GetVector
=============================

.. py:class:: mindsponge.function.GetVector(use_pbc)

    获取有或者没有PBC box的向量。

    参数：
        - **use_pbc** (bool) - 计算向量时是否使用周期性边界条件。

    输出：
        Tensor。计算所得向量。shape为(B, ..., D)。

    .. py:method:: get_vector_default(position0, position1, pbc_box)

        获取向量。

        参数：
            - **position0** (Tensor) - 起始位置的坐标。
            - **position1** (Tensor) - 终止位置的坐标。
            - **pbc_box** (Any) - 虚假参数。

    .. py:method:: get_vector_with_pbc(position0, position1, pbc_box)

        在有周期性边界条件的情况下获取向量。

        参数：
            - **position0** (Tensor) - 起始位置的坐标。
            - **position1** (Tensor) - 终止位置的坐标。
            - **pbc_box** (Any) - 虚假参数。

    .. py:method:: get_vector_without_pbc(position0, position1, pbc_box)

        在没有周期性边界条件的情况下获取向量。

        参数：
            - **position0** (Tensor) - 起始位置的坐标。
            - **position1** (Tensor) - 终止位置的坐标。
            - **pbc_box** (Any) - 虚假参数。

    .. py:method:: set_pbc(use_pbc)

        设定是否使用周期性边界条件。

        参数：
            - **use_pbc** (bool) - 是否使用周期性边界条件。