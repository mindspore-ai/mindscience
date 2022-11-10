mindsponge.data.load_hyper_param_into_class
===========================================

.. py:function:: mindsponge.data.load_hyper_param_into_class(cls_dict, hyper_param, types, prefix='')

    把超参数加载到Cell类中。

    参数：
        - **cls_dict** (dict) - 一个神经网络层的字典。
        - **hyper_param** (dict) - 超参数字典。
        - **types** (dict) - 值的种类的字典。
        - **prefix** (str) - 只有开头带有prefix的参数才会被加载。默认值："''"。