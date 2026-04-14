sponge.function.bonds_in
============================

.. py:function:: sponge.function.bonds_in(bonds, batch_bond)

    如果 bonds 中存在 batch_bond 则返回。

    参数：
        - **bonds** (Tensor) - 总bonds集。
        - **batch_bond** (Tensor) - 输入的bond集合。

    返回：
        如果 batch_bond 存在于 bonds 中，掩码将变为1，否则为0。
