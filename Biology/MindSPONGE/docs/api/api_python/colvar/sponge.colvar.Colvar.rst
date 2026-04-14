sponge.colvar.Colvar
========================

.. py:class:: sponge.colvar.Colvar(shape: Tuple[int] = (), periodic: Union[bool, List[bool]] = False, use_pbc: bool = None, name: str = 'colvar', unit: str = None, dtype: type = ms.float32)

    广义集合变量(Collective Variables, CVs)基类 :math:`s(R)`。

    在数学中，CVs :math:`s(R)` 被定义为模拟系统的原子坐标 :math:`s(R)` 的低维函数，它应该是指描述感兴趣过程中慢动作的变量。

    在MindSPONGE中，Colvar Cell是 ``"generalized"`` CVs 的基类。狭义的CV通常是一个向量，即它的秩(ndim)是1。例如，shape (S) 。而 Colvar 单元格可以是更高的秩(ndim)，例如，shape (S_1, S_2, ..., S_n) 的Colvar。

    对于 Colvar，可以使用多组坐标计算多个值。因此，对于shape (S_1, S_2, ... , S_n) 的Colvar 单元，使用shape (B, A, D) 的张量表示的原子坐标集进行计算，生成shape (B, S_1, S_2, ... , S_n) 的张量。其中B是批量大小，即模拟中的步行者数量。A是系统中的原子数。D是仿真系统的维度。通常为3。{S_i}是集合变量的维度。

    参考:
        Yang, Y. I.; Shao, Q.; Zhang, J.; Yang, L.; Gao, Y. Q.
        Enhanced Sampling in Molecular Dynamics [J].
        The Journal of Chemical Physics, 2019, 151(7): 070902.

    参数：
        - **shape** (Tuple) - 集合变量的shape。默认值：()
        - **periodic** (bool) - 判断集合变量是否是周期性的。默认值： ``False``。
        - **use_pbc** (bool) - 是否使用周期边界条件。如果给出 `None`，它将根据是否提供 `pbc_box` 来确定是否使用周期性边界条件。默认值： ``None``。
        - **name** (str) - 集合变量的名称。默认值：'colvar'。
        - **unit** (str) - 集合变量的单位。注意：这不是包裹长度和能量的 `Units` 单元格。默认值： ``None``。
        - **dtype** (type) - 集合变量的数据类型。默认值：float32。

    .. py:method:: all_periodic()
        :property:

        判断所有维度是否为周期性的。

    .. py:method:: any_periodic()
        :property:

        判断任一维度是否为周期性的。

    .. py:method:: dtype()
        :property:

        集合变量的数据类型。

        返回：
            类型，Colvar的数据类型。

    .. py:method:: get_unit(units: Units = None)

        返回集合变量的单位。

    .. py:method:: name()
        :property:

        集合变量的名称。

        返回：
            str，集合变量的名称。

    .. py:method:: ndim()
        :property:

        集合变量的秩（维度数）。

        返回：
            整型，集合变量的秩。

    .. py:method:: periodic()
        :property:

        返回数据类型为 `bool` 的张量，以指示CV是否是周期性的。    

    .. py:method:: reshape(input_shape: tuple)

        重新排列shape。

    .. py:method:: set_name(name: str)

        设置集合变量的名称。

    .. py:method:: set_pbc(use_pbc: bool)

        设置是否使用周期边界条件。

    .. py:method:: shape()
        :property:

        集合变量的shape (S_1, S_2, ..., S_n) 

        返回：
            shape(tuple)：Colvar的shape。

    .. py:method:: use_pbc()
        :property:

        判断是否使用周期边界条件。

        返回：
            bool，判断是否使用周期边界条件。

    .. py:method:: vector_in_pbc(vector: Tensor, pbc_box: Tensor)
        :classmethod:

        在 -0.5box 到 0.5box 的范围内计算出向量的差异。