sponge.metrics.get_metrics
============================

.. py:function:: sponge.metrics.get_metrics(metrics: Union[dict, set])

    获取分析中使用的指标。

    参数：
        - **metrics** (Union[dict, set]) - 模型在分子动力学运行或分析过程中要评估的指标或变量的字典或集合。

    返回：
        dict，键是指标名称，值是指标方法的类实例。

    异常：
        - **TypeError** - 如果参数 `metrics` 的类型不是 ``None``、dict 或 set，则会抛出此异常。
    