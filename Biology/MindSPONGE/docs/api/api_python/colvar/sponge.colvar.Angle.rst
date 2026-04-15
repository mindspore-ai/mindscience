sponge.colvar.Angle
========================

.. py:class:: sponge.colvar.Angle(atoms: AtomsBase = None, atoms_a: AtomsBase = None, atoms_b: AtomsBase = None, atoms_c: AtomsBase = None, vector1: Vector = None, vector2: Vector = None, use_pbc: bool = None, batched: bool = False, keepdims: bool = None, axis: int = -2, name: str = 'angle')

    角度的集合变量。

    参数：
        - **atoms** (AtomsBase) - shape为 (..., 3, D) 的原子形成shape为 (...) 或 (..., 1) 的角度。不能与 `atoms_a` 或 `atoms_b` 一起使用。默认值： ``None``。其中，D表示仿真系统的维度。通常为3。        
        - **atoms_a** (AtomsBase) - shape为 (..., D) 的原子A形成shape为 (...) 或 (..., 1) 的角度。必须与 `atoms_b` 和 `atoms_c` 一起使用。不能与 `atoms` 一起使用。默认值： ``None``。
        - **atoms_b** (AtomsBase) - shape为 (..., D) 的原子B形成shape为 (...) 或 (..., 1) 的角度。必须与 `atoms_a` 和 `atoms_c` 一起使用。不能与 `atoms` 一起使用。默认值： ``None``。
        - **atoms_c** (AtomsBase) - shape为 (..., D) 的原子C形成shape为 (...) 或 (..., 1) 的角度。必须与 `atoms_a` 和 `atoms_b` 一起使用。不能与 `atoms` 一起使用。默认值： ``None``。
        - **vector1** (Vector) - shape为 (..., D) 的vector1变为shape为 (...) 或 (..., 1) 的角度形式。必须与 `vector2` 一起使用。不能与原子一起使用。默认值： ``None``。
        - **vector2** (Vector) - shape为 (..., D) 的vector2变为shape为 (...) 或 (..., 1) 的角度形式。必须与 `vector1` 一起使用。不能与原子一起使用。默认值： ``None``。
        - **use_pbc** (bool) - 是否在周期边界条件下计算距离。默认值： ``None``。
        - **batched** (bool) - 索引的第一个维度是否为批大小。默认值： ``False``。
        - **keepdims** (bool) - 是否保留向量最后一个维度。默认值： ``False``。
        - **axis** (int) - 从原子坐标中收集点的轴。默认值：-2。
        - **name** (str) - Colvar的名称。默认值：'angle'。