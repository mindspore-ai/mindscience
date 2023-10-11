sponge.function.GetVector
=============================

.. py:class:: sponge.function.GetVector(use_pbc: bool = None)

    获取有或者没有PBC box的向量。

    参数：
        - **use_pbc** (bool) - 计算向量时是否使用周期性边界条件。默认值： ``None`` 。


    .. py:method:: calc_vector_default(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None)

        获取向量。

        参数：
            - **initial** (Tensor) - 起始位置的坐标。
            - **terminal** (Tensor) - 终止位置的坐标。
            - **pbc_box** (Tensor) - 默认值： ``None`` 。

    .. py:method:: calc_vector_nopbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None)

        在没有周期性边界条件的情况下获取向量。

        参数：
            - **initial** (Tensor) - 起始位置的坐标。
            - **terminal** (Tensor) - 终止位置的坐标。
            - **pbc_box** (Tensor) - 默认值： ``None`` 。

    .. py:method:: calc_vector_pbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None)

        在有周期性边界条件的情况下获取向量。

        参数：
            - **initial** (Tensor) - 起始位置的坐标。
            - **terminal** (Tensor) - 终止位置的坐标。
            - **pbc_box** (Tensor) - 默认值： ``None`` 。

    .. py:method:: set_pbc(use_pbc: bool)

        设定是否使用周期性边界条件。

        参数：
            - **use_pbc** (bool) - 是否使用周期性边界条件。默认值： ``None`` 。

    .. py:method:: use_pbc()

        是否使用周期性边界条件。

        返回：
            bool。是否使用周期性边界条件。