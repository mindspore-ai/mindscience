sciai.utils.lazy_func
=======================

.. py:function:: sciai.utils.lazy_func(func, *args, **kwargs)

    制造一个可以懒加载的函数，该函数可在之后直接被无参调用。

    参数：
        - **func** (Callable) - 配置字典。
        - **\*args** (any) - 配置字典。
        - **\*\*kwargs** (any) - 配置字典。

    返回：
        Function，所构造的懒加载无参函数。
