mindelec.solver.Problem
=======================

.. py:class:: mindelec.solver.Problem

    求解偏微分方程问题的基类，用户自定义的问题需要继承此基类以建立子数据集和损失函数之间的映射。
    映射将由Constraint API构造，损失将由每个子数据集的约束类型自动计算。
    根据约束类型，对应的成员函数必须由用户重载，以获取目标残差。例如，对于dataset1，约束类型设置为"Equation"，因此成员函数"governing_equation"必须重载，以告知如何获取方程残差。

    .. py:method:: mindelec.solver.Problem.boundary_condition(*output, **kwargs)

        边界条件，抽象方法。
        如果相应的约束类型为"BC"，必须重载此函数。
        如果边界条件可以表示为 `f(bc_points) = 0` ，则将返回残差f，bc_points为边界上的数据点。

        参数：
            - **output** (tuple) - 代理模型输出，如电场、磁场等。
            - **kwargs** (dict) - 代理模型输入，如时间、空间等。

    .. py:method:: mindelec.solver.Problem.constraint_function(*output, **kwargs)

        函数约束的普遍情况，抽象方法。
        如果相应的约束类型为"Label"或"Function"，则必须重载此函数。
        它是约束类型的更普遍情况，可以表示为 `f(inputs) = 0` ，inputs为通用函数的数据点。
        将返回残差f。

        参数：
            - **output** (tuple) - 代理模型输出，如电场、磁场等。
            - **kwargs** (dict) - 代理模型输入，如时间、空间等。

    .. py:method:: mindelec.solver.Problem.governing_equation(*output, **kwargs)

        控制方程，抽象方法。
        如果相应的约束类型为"Equation"，则必须重载此函数。
        如果方程为 `f(inputs) = 0` ，则将返回残差f，inputs为控制区间的数据点。

        参数：
            - **output** (tuple) - 代理模型输出，如电场、磁场等。
            - **kwargs** (dict) - 代理模型输入，如时间、空间等。

    .. py:method:: mindelec.solver.Problem.initial_condition(*output, **kwargs)

        初始条件，抽象方法。
        如果相应的约束类型为"IC"，则必须重载此函数。
        如果初始条件可以表示为 `f(ic_points) = 0` ，则将返回残差f，ic_points为初始时刻数据点。

        参数：
            - **output** (tuple) - 代理模型输出，如电场、磁场等。
            - **kwargs** (dict) - 代理模型输入，如时间、空间等。

