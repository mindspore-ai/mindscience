sponge.function.GetDistance
===============================

.. py:class:: sponge.function.GetDistance(use_pbc=None)

    获取有或者没有PBC box的距离。

    参数：
        - **use_pbc** (bool) - 计算距离时是否使用周期性边界条件。默认值："None"。

    输出：
        Tensor。计算所得距离。shape为(B, ...)。

    符号：
        - **B** - Batchsize。

    .. py:method:: set_pbc(use_pbc)

        设定是否使用周期性边界条件。

        参数：
            - **use_pbc** (bool) - 是否使用周期性边界条件。