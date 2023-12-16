sponge.colvar.CosCV
========================

.. py:class:: sponge.colvar.CosCV(colvar: Colvar, name: str = 'cosine')

    集合变量(CVs)的余弦值 :math:`s(R)`。返回值与输入CVs有相同的shape。

    参数：
        - **colvar** (Colvar) - 集合变量(CVs) :math:`s(R)`。
        - **name** (str) - 集合变量的名称。默认值：'cosine'。