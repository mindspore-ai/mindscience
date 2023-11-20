sponge.colvar.ConstantValue
=================================

.. py:class:: ConstantValue(value: Union[Tensor, ndarray, list, tuple], name: str = 'constant')

    常量值。

    参数：
        - **value** (Union[Tensor, ndarray, list, tuple]) - 常量值
        - **name** (str) - Colvar 的名称。默认值：'constant'

    支持的平台：
        ``Ascend`` ``GPU`` ``CPU``
    
    .. py:method:: construct(coordinate: Tensor, pbc_box: bool = None)

        返回常量值。

        参数：
            参数：
            - **coordinate** (Tensor) - 张量的shape (B, A, D) 。数据类型为float。
            - **pbc_box** (Tensor) - 张量的shape (B, D) 。数据类型为float。默认值：``None``。
        
        返回：
            常量值(Tensor): 张量的shape (B, ...) 或 (B, ..., 1) 。数据类型为浮点型。