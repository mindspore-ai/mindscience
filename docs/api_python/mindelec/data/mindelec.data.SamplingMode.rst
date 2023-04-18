mindelec.data.SamplingMode
==========================

.. py:class:: mindelec.data.SamplingMode

    点采样方法，目前支持 ``UPPERBOUND(0)`` 和 ``DIMENSIONS(1)``。

    - ``'UPPERBOUND'``：限制整个采样空间内的采样点数上限，每个轴上的采样点数等其他空间参数可以根据空间大小比自动计算。
    - ``'DIMENSIONS'``：用户可以指定每个维度中的采样编号，轴顺序为x:y:z。
