mindsponge.data.get_hyper_parameter
===================================

.. py:function:: mindsponge.data.get_hyper_parameter(hyper_param, prefix)

    获取超参数。

    参数：
        - **hyper_param** (dict) - 超参数字典。
        - **prefix** (str) - 只有开头带有prefix的参数才会被加载。

    输出：
        Tensor。带有prefix的超参数。