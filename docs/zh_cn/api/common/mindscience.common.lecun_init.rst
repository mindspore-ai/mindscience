mindscience.common.lecun_init
==============================

.. py:function:: mindscience.common.lecun_init(fan_in, initializer_name="linear")

    LeCun 初始化。

    参数：
        - **fan_in** (int) - 输入特征数。
        - **initializer_name** (str, 可选) - 初始化器名称，支持 ``"linear"`` 或 ``"relu"``。默认 ``"linear"``。

    返回：
        Initializer，返回生成的初始化器。