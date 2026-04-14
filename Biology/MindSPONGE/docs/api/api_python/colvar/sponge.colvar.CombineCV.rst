sponge.colvar.CombineCV
============================

.. py:class:: sponge.colvar.CombineCV(colvar: Union[List[Colvar], Tuple[Colvar]], weights: Union[float, List[float], Tuple[float], Tensor] = 1,offsets: Union[float, List[float], Tuple[float], Tensor] = 0, exponents: Union[float, List[float], Tuple[float], Tensor] = 1, normal: bool = False, periodic_min: Union[float, ndarray, Tensor] = None, periodic_max: Union[float, ndarray, Tensor] = None, periodic_mask: Union[Tensor, ndarray] = None, use_pbc: bool = None, name: str = 'combine')

    一组 Colvar :math:`{s_i}` 与shape为 (S_1， S_2， ...， S_n) 的多项式组合。{S_i}表示集合变量的维度。

    .. math::

        S = \sum_i^n{w_i (s_i - o_i)^{p_i}}

    参数：
        - **colvar** (Union[List[Colvar], Tuple[Colvar]]) - 要组合的 `Colvar` 数组 :math:`{s_i}`。
        - **weights** (Union[List[float], Tuple[Float], float, Tensor]) - Weights :math:`{w_i}` 对每组Colvar。如果给定列表或元组，则元素的数量应等于 CVs 的数量。如果给定float或Tensor，则该值将用于所有 Colvar。默认值：1。
        - **offsets** (Union[List[float], Tuple[Float], float, Tensor]) - Offsets :math:`{o_i}` 对每组Colvar。如果给定列表或元组，则元素的数量应等于 CVs 的数量。如果给定float或Tensor，则该值将用于所有 Colvar。默认值：0。
        - **exponents** (Union[List[float], Tuple[Float], float, Tensor]) - Exponents :math:`{p_i}` 对每组Colvar。如果给定列表或元组，则元素的数量应等于 CVs 的数量。如果给定float或Tensor，则该值将用于所有 Colvar。默认值：1。
        - **normal** (bool) - 是否将所有权重归一化为 1。默认值： ``False``。
        - **periodic_min** (Union[float, ndarray, Tensor]) - CVs 组合输出的周期性最小值。如果输出不是周期性的，则应为空。默认值： ``None``。
        - **periodic_max** (Union[float, ndarray, Tensor]) - CVs 组合输出的周期性最大值。如果输出不是周期性的，则应为空。默认值： ``None``。
        - **periodic_mask** (Union[Tensor, ndarray]) - 输出周期性的掩码。张量的shape应与输出相同，即 (S_1, S_2, ..., S_n) 。默认值： ``None``。
        - **use_pbc** (bool) - 是否使用周期边界条件。如果给出 ``None`` ，它将确定是否使用基于是否提供 `pbc_box` 的周期性边界条件。默认值： ``None``。
        - **name** (str) - 集合变量的名称。默认值：'combine'。

    .. py:method:: set_pbc(use_pbc: bool)

        设置是否使用周期边界条件。