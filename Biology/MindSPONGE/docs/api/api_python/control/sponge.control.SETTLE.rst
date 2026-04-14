sponge.control.SETTLE
=========================

.. py:class:: sponge.control.SETTLE(system: :class:`sponge.system.Molecule`, index: Union[Tensor, ndarray, List[int]] = None, distance: Union[Tensor, ndarray, List[float]] = None)

    SETTLE约束控制器。

    参考文献 Miyamoto, S., Kollman, P.A., 1992. Settle: An analytical version of the SHAKE and RATTLE algorithm for rigid water models. Journal of Computational Chemistry 13, 952–962.

    参数：
        - **system** (:class:`sponge.system.Molecule`) - 模拟系统。
        - **index** (Union[Tensor, ndarray, List[int]], 可选) - SETTLE索引。shape为 :math:`(C, 3)` 或 :math:`(B, C, 3)` 的Tensor。这里 :math:`B` 为分子模拟中walker的数目， :math:`C` 是约束数目。数据类型为int。如果取值为 ``None``，则使用`system`中的`settle_index`。默认值： ``None``。
        - **distance** (Union[Tensor, ndarray, List[float]], 可选) - SETTLE距离。shape为 :math:`(C, 3)` 或 :math:`(B, C, 2)` 的Tensor。数据类型为float。如果取值为 ``None``，则使用`system`中的`settle_dis`。默认值： ``None``。

    输入：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。这里 :math:`B` 是分子模拟中walker的数目， :math:`A` 是原子数目， :math:`D` 是模拟系统的空间维数，通常为3。
        - **velocity** (Tensor) - 速度。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **energy** (Tensor) - 能量。shape为 :math:`(B, 1)` 的Tensor。数据类型是float。
        - **kinetics** (Tensor) - 动能。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **virial** (Tensor) - 维里。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **pbc_box** (Tensor) - 周期性边界条件盒子。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **step** (int) - 模拟步数。默认值： ``0``。

    输出：
        - **coordinate** (Tensor) - 坐标。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **velocity** (Tensor) - 速度。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **force** (Tensor) - 原子力。shape为 :math:`(B, A, D)` 的Tensor。数据类型是float。
        - **energy** (Tensor) - 能量。shape为 :math:`(B, 1)` 的Tensor。数据类型是float。
        - **kinetics** (Tensor) - 动能。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **virial** (Tensor) - 维里。shape为 :math:`(B, D)` 的Tensor。数据类型是float。
        - **pbc_box** (Tensor) - 周期性边界条件盒子。shape为 :math:`(B, D)` 的Tensor。数据类型是float。

    .. py:method:: apply_transform(q: Tensor, vec: Tensor)

        应用变换四元数。

        参数：
            - **q** (Tensor) - 变换四元数。
            - **vec** (Tensor) - 向量。

        返回：
            - Tensor，变换后的向量。

    .. py:method:: get_mass_center(crd_)

        获取质心坐标。

        参数：
            - **crd_** (Tensor) - 坐标。

        返回：
            - Tensor，质心坐标。

    .. py:method:: get_inverse(quater: Tensor)

        获取四元数的逆。

        参数：
            - **quater** (Tensor) - 四元数。

        返回：
            - Tensor，四元数的逆。

    .. py:method:: get_transform(crd_)

        获取A0B0C0与a0b0c0之间的变换。

        参数：
            - **crd_** (Tensor) - 水分子的坐标。

        返回：
            - Tensor，水分子坐标SETTLE数轴的变换以及逆变换。

    .. py:method:: get_vector_transform(vec1: Tensor, vec2: Tensor)

        获取两个向量的变换四元数。

        参数：
            - **vec1** (Tensor) - 初始向量。
            - **vec2** (Tensor) - 目标向量。

        返回：
            - Tensor，变换四元数。
    
    .. py:method:: get_vel_force_update(crd0_: Tensor, vel0_: Tensor)
            
        获取速度和力的更新。

        参数：
            - **crd0_** (Tensor) - SETTLE之后在初始坐标轴上的坐标。
            - **vel0_** (Tensor) - 初始速度。

        返回：
            - Tensor，约束速度。
            - Tensor，约束力。

    .. py:method:: group_hamiltonian_product(q: Tensor, vec: Tensor)

        四元数与4维向量的哈密顿积。

        参数：
            - **q** (Tensor) - 四元数。
            - **vec** (Tensor) - 向量。

        返回：
            - Tensor，四元数与向量的哈密顿积 :math:`q v q^{-1}`。

    .. py:method:: hamiltonian_product(q: Tensor, v: Tensor)
            
        四元数与向量的哈密顿积。

        参数：
            - **q** (Tensor) - 四元数。
            - **v** (Tensor) - 向量。

        返回：
            - Tensor，四元数与向量的哈密顿积 :math:`q v q^{-1}`。

    .. py:method:: quaternion_multiply(tensor_1: Tensor, tensor_2: Tensor)

        四元数乘法。

        参数：
            - **tensor_1** (Tensor) - 第一个四元数。如果最后一个维度的大小为3，则会自动补齐为4。
            - **tensor_2** (Tensor) - 第二个四元数。如果最后一个维度的大小为3，则会自动补齐为4。

        返回：
            - Tensor，两个四元数乘积。
