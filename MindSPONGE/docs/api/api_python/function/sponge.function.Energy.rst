sponge.function.Energy
==========================

.. py:class:: sponge.function.Energy(value: float, unit: str = 'kj/mol', **kwargs)

    能量。

    参数：
        - **value** (float) - 能量值。
        - **unit** (str) - 能量单位。默认值："kj/mol"。
        - **kwargs** - 其他参数。

    .. py:method:: abs_size()

        获取能量绝对值。

        返回：
            float。能量的绝对值。

    .. py:method:: change_unit(unit)

        改变单位。

        参数：
            - **unit** (Union[str, Units, float, int]) - 能量单位。

    .. py:method:: ref()

        获取能量参考值。

        返回：
            float。一个能量参考值。

    .. py:method:: unit()

        获取能量单位。

        返回：
            str。能量单位。

    .. py:method:: unit_name()

        获取能量单位的名称。

        返回：
            str。能量单位的名称。

    .. py:method:: value()

        获取能量值。

        返回：
            float。能量值。