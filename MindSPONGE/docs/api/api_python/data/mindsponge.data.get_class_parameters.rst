mindsponge.data.get_class_parameters
====================================

.. py:class:: mindsponge.data.get_class_parameters(hyper_param, prefix, num_class=1)

    从Cell类中获取超参数。

    参数：
        - **hyper_param** (dict) - 超参数字典。
        - **prefix** (str) - 只有开头带有prefix的参数才会被加载。
        - **num_class** (int) - 类的数量。

    输出：
        dict。超参数的字典。