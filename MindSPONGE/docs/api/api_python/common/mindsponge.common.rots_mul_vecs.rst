mindsponge.common.rots_mul_vecs
===============================

.. py:function:: mindsponge.common.rots_mul_vecs(m, v)

    利用旋转矩阵 :math:`\vec m = (m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8)` 对输入向量 :math:`\vec v = (v_0, v_1, v_2)` 进行旋转。

    .. math::
        
        out = m \cdot v^T = (m_0 \times v_0 + m_1 \times v_1 + m_2 \times v_2, m_3 \times v_0 + m_4 \times v_1 + m_5 \times v_2, m_6 \times v_0 + m_7 \times v_1 + m_8 \times v_2)

    参数：
        - **m** (tuple) - 旋转矩阵 :math:`\vec m` ，长度为9，数据类型为标量或者shape相同的Tensor。
        - **v** (tuple) - 向量 :math:`\vec v` ，长度为3，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple, 旋转后的向量，长度为3，数据类型为标量或者shape相同的Tensor。。