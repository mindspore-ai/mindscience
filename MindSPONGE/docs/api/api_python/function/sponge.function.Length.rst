sponge.function.Length
==========================

.. py:class:: sponge.function.Length(value: float, unit: str = "nm", **kwargs)

    获取长度。

    参数：
        - **value** (float) - 长度值。
        - **unit** (str) - 长度单位。默认值："nm"。
        - **kwargs** (str) - 其他参数。

    .. py:method:: abs_size()

        获取长度绝对值。

        返回：
            float。长度的绝对值。

    .. py:method:: change_unit(unit)

        改变单位。

        参数：
            - **unit** (Union[str, Units, float, int]) - 长度单位。

    .. py:method:: ref()

        获取度参考值。

        返回：
            float。一个长度参考值。

    .. py:method:: unit()

        获取长度单位。

        返回：
            str。长度单位。

    .. py:method:: unit_name()

        获取长度单位的名称。

        返回：
            str。长度单位的名称。

    .. py:method:: value()

        获取长度值。

        返回：
            float。长度值。
