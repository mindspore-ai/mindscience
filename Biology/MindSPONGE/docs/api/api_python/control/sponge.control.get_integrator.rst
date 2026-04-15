sponge.control.get_integrator
=================================

.. py:function:: sponge.control.get_integrator(cls_name: Union[str, dict, :class:`sponge.control.Integrator`], system: :class:`sponge.system.Molecule`, **kwargs)

    获取积分器对象。

    参数：
        - **cls_name** (Union[str, dict, :class:`sponge.control.Integrator`]) - 积分器类名, 积分器对象或者积分器类的参数字典。
        - **system** (:class:`sponge.molecule.Molecule`) - 模拟体系。
        - **kwargs** (dict) - 积分器的其他关键字参数。
    
    返回：
        Integrator。积分器对象。
