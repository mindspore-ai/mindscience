sponge.function.get_ndarray
===============================

.. py:function:: sponge.function.get_ndarray(value: Union[Tensor, Parameter, ndarray, List[float], Tuple[float]], dtype: type = None)

    获取输入的ndarray类型。

    参数：
        - **value** (Union[Tensor, Parameter, ndarray, List[float], Tuple[float]]) - 输入的值。
        - **dtype** (type) - 数据类型。默认值： ``None`` 。

    返回：
        ndarray。把输入转换为ndarray类型并返回。