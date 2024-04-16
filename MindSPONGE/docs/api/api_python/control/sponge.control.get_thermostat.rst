sponge.control.get_thermostat
=================================

.. py:function:: sponge.control.get_thermostat(cls_name: Union[str, dict, :class:`sponge.control.Thermostat`], system: :class:`sponge.system.Molecule`, temperature: float = None, **kwargs)

    获取恒温器对象。

    参数：
        - **cls_name** (Union[str, dict, :class:`sponge.control.Thermostat`]) - 恒温器类名，恒温器的参数字典，或者恒温器对象。`
        - **system** (:class:`sponge.molecule.Molecule`) - 模拟系统。
        - **temperature** (float) - 温度耦合的参考温度。 默认值为 ``None``。如果为 ``None``，且`cls_name`的类型为`str`，则该函数返回 ``None``。
        - **kwargs** (dict) - 恒温器的其他关键字参数。
    
    返回：
        :class:`sponge.control.Thermostat`，恒温器对象。
