sponge.colvar.SinCV
========================

.. py:class:: sponge.colvar.SinCV(colvar: Colvar, name: str = 'sine')

    集合变量（CVs）的正弦值 :math:`s(R)`。返回值与输入CVs有相同的shape。

    .. math::

        s' = \sin{s(R)}

    参数：
        - **colvar** (Colvar) - 集合变量(CVs) :math:`s(R)`。
        - **name** (str) - 集合变量的名称。默认值：'cosine'。