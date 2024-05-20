sponge.control.get_constraint
=================================

.. py:function:: sponge.control.get_constraint(constraint: Union[str, :class`sponge.control.Constraint`, List[:class`sponge.control.Constraint`]], system: :class`sponge.system.Molecule`)

    获取约束对象。

    参数：
        - **constraint** (Union[str, :class:`sponge.control.Constraint`, List[:class:`sponge.control.Constraint`]]) - 约束的类名，约束对象，或者约束对象列表。
        - **system** (:class:`sponge.system.Molecule`) - 模拟系统。
    
    返回：
        List[:class:`sponge.control.Constraint`]。约束对象列表。