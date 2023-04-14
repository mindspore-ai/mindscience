mindflow.pde.PDEWithLoss
========================

.. py:class:: mindflow.pde.PDEWithLoss(model, in_vars, out_vars, params=None, params_val=None)

    求解偏微分方程问题的基类。
    所有用户自定义问题应该从该类继承，用于在每个数据集上设置约束。它用于建立每个子数据集和使用的定义损失函数之间的映射。损失将根据每个子数据集的约束类型自动计算。为了获得目标标签输出，用户必须根据约束类型重载相应的成员函数。例如，对于dataset1，约束类型为“pde”，因此必须重载成员函数“pde”以告诉如何获得pde残差。用于求解残差的数据（例如输入）被传递到parse_node，便可自动计算每个方程的残差。

    参数：
        - **model** (mindspore.nn.Cell) - 用于训练的网络模型。
        - **in_vars** (List[sympy.core.Symbol]) - `model` 的输入参数，sympy符号表示的自变量。
        - **out_vars** (List[sympy.core.Function]) - `model` 的输出参数，sympy符号表示的因变量。
        - **params** (List[sympy.core.Function]) - 问题中非输入的可学习参数。
        - **params_val** (List[sympy.core.Function]) - 问题中非输入的可学习参数的值。

    .. note::
        - `pde` 方法必须重写，用于定义sympy符号微分方程。
        - `get_loss` 方法必须重写，用于计算符号微分方程的损失。

    .. py:method:: get_loss()

        计算所有定义的微分方程的损失。用户必须重写该方法。

    .. py:method:: parse_node(formula_nodes, inputs=None, norm=None)

        计算定义的微分方程的预测结果。

        参数：
            - **formula_nodes** (list[FormulaNode]) - 转义后的sympy表达式，可以被MindSpore识别。
            - **inputs** (Tensor) - 网络模型的输入数据。默认值： ``None``。
            - **norm** (Tensor) - 输入数据点的法向量。对于曲面上某点P处的法向量是垂直于该点的切平面的向量。默认值： ``None``。

        返回：
            list[Tensor]，偏微分方程的计算结果。
    
    .. py:method:: pde()

        抽象方法，基于sympy定义的控制方程。如果相关约束为控制方程，该方法必须被重写。