sponge.partition.IndexColvar
=============================

.. py:class:: sponge.partition.IndexColvar(use_pbc: bool = None)

    基于索引的集体变量 (Collective variables)。

    参数：
        - **use_pbc** (bool，可选) - 是否在周期边界条件（PBC）下计算CV。如果给定 ``None`` ，则将在运行时基于是否给定 `pbc_box` 来确定。默认值： ``None``。

    .. py:method:: set_pbc(use_pbc: bool)

        设置周期边界条件。
        
        参数：
            - **use_pbc** (bool) - 是否在周期边界条件（PBC）下计算CV。如果给定 ``None``，则将在运行时基于是否给定 `pbc_box` 来确定。

    .. py:method:: vector_in_pbc(vector: Tensor, pbc_box: Tensor)

        将向量的差异设置在从-0.5 box 到0.5 box 的范围内。
        
        参数：
            - **vector** (Tensor) - shape为 :math:`(B, A, D)` 的Tensor。数据类型为float。系统的坐标。
            - **pbc_box** (Tensor) - shape为 :math:`(B, D)` 的Tensor。数据类型为float。周期边界条件。
