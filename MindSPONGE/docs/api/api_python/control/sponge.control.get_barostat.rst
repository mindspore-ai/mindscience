sponge.control.get_barostat
=================================

.. py:function:: sponge.control.get_barostat(cls_name: Union[str, dict, :class:`spogne.control.Barostat`], system: :class:`sponge.system.Molecule`, pressure: float = None,**kwargs)

    获取恒压器对象。

    参数：
        - **cls_name** (Union[str, dict, :class:`sponge.control.Barostat`]) - 恒压器的类名， 恒压器的参数字典，或者恒压器对象。
        - **system** (:class: `sponge.control.Molecule`) - 模拟系统。
        - **temperature** (float, 可选) - 压力耦合的参考压力。 默认值为 ``None``。如果为 ``None``，且`cls_name`的类型为`str`，则该函数返回 ``None``。
        - **kwargs** (dict) - 恒压器的其他关键字参数。

    返回：
        :class:`sponge.control.Barostat`。恒压器对象。
