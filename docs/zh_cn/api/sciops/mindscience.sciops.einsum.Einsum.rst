mindscience.sciops.einsum.Einsum
=================================

.. py:class:: mindscience.sciops.einsum.Einsum(equation, use_opt=True)

    使用爱因斯坦求和约定（Einsum）进行张量计算的操作。

    此操作符使用爱因斯坦求和约定（Einsum）执行张量计算。支持对角化、约简、转置、矩阵乘法、乘积运算、内积等操作。

    参数：
        - **equation** (str) - 指定要执行的计算。仅接受：

          - 字母 ([a-z][A-Z])：表示输入张量的维度
          - ...：匿名维度
          - 逗号 (',')：分隔张量维度
          - 箭头 ('->')：左侧指定输入张量，右侧指定所需的输出维度

        - **use_opt** (bool, 可选) - 默认为 ``True``。设置为 ``False`` 时，不执行收缩路径优化。

    输入：
        - **operands** (List[Tensor]) - 可变数量的张量输入。

    输出：
        - **out_tensor** (Tensor) - Einsum 操作的结果。
