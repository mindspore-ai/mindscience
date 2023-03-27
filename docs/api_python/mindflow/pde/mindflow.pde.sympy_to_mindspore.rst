mindflow.pde.sympy_to_mindspore
===============================

.. py:function:: mindflow.pde.sympy_to_mindspore(equations, in_vars, out_vars, params=None)

    将sympy定义的符号方程转换为MindSpore能够支持的语法。

    参数：
        - **equations** (dict) - 自定义偏微分方程组，每个方程的健由用户定义，其值为sympy表达式。
        - **in_vars** (list[sympy.core.Symbol]) - 求解偏微分方程网络模型的输入参数，sympy符号表示的自变量，和输入数据的维度一致。
        - **out_vars** (list[sympy.core.Function]) - 求解偏微分方程网络模型的输出参数，sympy符号表示的因变量，和输出数据的维度一致。
        - **params** (list[sympy.core.Function]) - 求解偏微分方程网络模型的额外的可训练参数，sympy符号表示的因变量。

    返回：
        list[FormulaNode]，转换后的表达式，能够被MindSpore识别。