mindsponge.data.set_class_parameters
====================================

.. py:class:: mindsponge.data.set_class_parameters(hyper_param, prefix, cell)

    把超参数放入Cell类中。

    参数：
        - **hyper_param** (dict) - 超参数字典。
        - **prefix** (str) - 只有开头带有prefix的参数才会被加载。
        - **cell** (Cell) - 一个神经网络层。