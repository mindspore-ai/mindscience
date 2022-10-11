mindsponge.data.set_hyper_parameter
===================================

.. py:class:: mindsponge.data.set_hyper_parameter(hyper_param, prefix, param)

    把参数放入超参数中。

    参数：
        - **hyper_param** (dict) - 超参数字典。
        - **prefix** (str) - 只有开头带有prefix的参数才会被加载。
        - **param** (Union[str, Tensor]) - 需要被放入超参数字典中的参数。