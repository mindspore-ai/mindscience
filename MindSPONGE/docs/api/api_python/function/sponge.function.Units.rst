sponge.function.Units
=========================

.. py:class:: sponge.function.Units(length_unit: str = None, energy_unit: str = None, **kwargs)

    记录和转换长度和能量单位。

    参数：
        - **length_unit** (str) - 长度单位。默认值： ``None`` 。
        - **energy_unit** (str) - 能量单位。默认值： ``None`` 。
        - **kwargs** - 其他参数。

    .. py:method:: acceleration_ref()
        :property:

        获取加速度的参考值。

        返回：
            float。加速度参考值。

    .. py:method:: avogadro()
        :property:

        获取阿佛加德罗数。

        返回：
            float。阿佛加德罗数。

    .. py:method:: boltzmann()
        :property:

        获取当前单位的玻尔兹曼常数。

        返回：
            float。当前单位的玻尔兹曼常数。

    .. py:method:: boltzmann_def()
        :property:

        获取kJ/mol单位下的玻尔兹曼常数。

        返回：
            float。kJ/mol单位下的玻尔兹曼常数。

    .. py:method:: convert_energy_from(unit)

        从指定单位转换能量。

        参数：
            - **unit** (Union[str, Units, Energy, float, int]) - 能量单位。

        返回：
            float。从指定单位转换来的能量。

    .. py:method:: convert_energy_to(unit)

        把能量转换到指定单位。

        参数：
            - **unit** (Union[str, Units, Energy, float, int]) - 能量单位。

        返回：
            float。转换到指定单位的能量。

    .. py:method:: convert_length_from(unit)

        从指定单位转换长度。

        参数：
            - **unit** (Union[str, Units, Length, float, int]) - 长度单位。

        返回：
            float。从指定单位转换来的长度。

    .. py:method:: convert_length_to(unit)

        把长度转换到指定单位。

        参数：
            - **unit** (Union[str, Units, Length, float, int]) - 长度单位。

        返回：
            float。转换到指定单位的长度。

    .. py:method:: coulomb
        :property:

        获取当前单位下的库伦常数。

        返回：
            float。当前单位下的库伦常数。

    .. py:method:: energy(value: float, unit = None)

        获取当前单位的能量值。

        参数：
            - **value** (float) - 能量值。
            - **unit** (Union[str, Units, Energy, float, int]) - 能力单位。

        返回：
            float。能量值。

    .. py:method:: energy_ref()
        :property:

        获取能量参考值。

        返回：
            float。能量参考值。

    .. py:method:: energy_unit()
        :property:

        获取能量单位。

        返回：
            str。能量单位。

    .. py:method:: energy_unit_name()
        :property:

        获取能量单位的名称。

        返回：
            str。能量单位的名称。

    .. py:method:: force_ref()
        :property:

        获取力的参考值。

        返回：
            float。力的参考值。

    .. py:method:: force_unit()
        :property:

        获取力的单位。

        返回：
            str。力的单位。

    .. py:method:: force_unit_name()
        :property:

        获取力的单位名称。

        返回：
            str。力的单位名称。

    .. py:method:: gas_constant()
        :property:

        获取气体常数。

        返回：
            float。气体常数。

    .. py:method:: get_boltzmann(energy_unit: str = None)

        获取指定单位下的玻尔兹曼常数。

        参数：
            - **energy_unit** (str) - 能量单位。默认值： ``None``。
    
    .. py:method:: get_coulomb(length_unit: str = None, energy_unit: str = None)

        获取指定单位下的库伦常数。

        参数：
            - **length_unit** (str) - 长度单位。默认值： ``None``。
            - **energy_unit** (str) - 能量单位。默认值： ``None``。

    .. py:method:: kinetic_ref()
        :property:

        获取动力学的参考值。

        返回：
            float。动力学的参考值。

    .. py:method:: length(value: float, unit = None)

        获取当前单位的长度值。

        参数：
            - **value** (float) - 长度值。
            - **unit** (Union[str, Units, Length, float, int]) - 长度单位。

        返回：
            float。长度值。

    .. py:method:: length_ref()
        :property:

        获取长度的参考值。

        返回：
            float。长度的参考值。

    .. py:method:: length_unit()
        :property:

        获取长度单位。

        返回：
            str。 长度单位。

    .. py:method:: length_unit_name()
        :property:

        获取长度单位的名称。

        返回：
            str。长度单位的名称。

    .. py:method:: pressure_ref()
        :property:

        获取压力的参考值。

        返回：
            float。压力的参考值。

    .. py:method:: set_energy_unit(unit: str = None)

        设置能量单位。

        参数：
            - **unit** (str) - 能量单位。

    .. py:method:: set_length_unit(unit: str = None)

        设置长度单位。

        参数：
            - **unit** (str) - 长度单位。

    .. py:method:: set_units(length_unit: str = None, energy_unit: str = None, units=None)

        设置长度单位。

        参数：
            - **length_unit** (str) - 长度单位。默认值： ``None``。
            - **energy_unit** (str) - 能量单位。默认值： ``None``。
            - **units** (Units) - 单位。默认值： ``None``。

    .. py:method:: velocity_unit()
        :property:

        获取速度单位。

        返回：
            str。速度单位。

    .. py:method:: velocity_unit_name()
        :property:

        获取速度单位的名称。

        返回：
            str。速度单位的名称。

    .. py:method:: volume_unit()
        :property:

        获取容积单位。

        返回：
            str。容积单位。

    .. py:method:: volume_unit_name()
        :property:

        获取容积单位的名称。

        返回：
            str。容积单位的名称。
