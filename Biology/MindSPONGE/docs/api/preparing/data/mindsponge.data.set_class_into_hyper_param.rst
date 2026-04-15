mindsponge.data.set_class_into_hyper_param
==========================================

.. py:function:: mindsponge.data.set_class_into_hyper_param(hyper_param, types, cls, prefix='')

    从Cell类中取出超参数。

    参数：
        - **hyper_param** (dict) - 超参数字典。
        - **types** (dict) - 值的种类的字典。
        - **cls** (Cell) - 一个神经网络层。
        - **prefix** (str) - 只有开头带有prefix的参数才会被加载。默认值："''"。