mindscience.e3nn.nn.Scatter
===========================

.. py:class:: mindscience.e3nn.nn.Scatter(mode='add')

    易用的 scatter 操作封装：根据索引将源张量的值聚合到目标张量中。

    参数：
        - **mode** (str，可选) - {'add', 'sum', 'div', 'max', 'min', 'mul'}，scatter 模式。

          - 'add' 或 'sum'：逐元素相加；
          - 'div'：逐元素相除；
          - 'max'：逐元素取最大值；
          - 'min'：逐元素取最小值；
          - 'mul'：逐元素相乘。
          
          默认值：``'add'`` 。

    输入：
        - **src** (Tensor) - 源张量。
        - **index** (Tensor) - 待聚合元素的索引，必须为整型张量。
        - **out** (Tensor，可选) - 目标张量；提供时在该张量上就地执行 scatter。默认值：``None`` 。
        - **dim_size** (int，可选) - 当未提供 `out` 时，自动创建大小为 `dim_size` 的输出；若未提供，则返回最小尺寸的输出。默认值：``None`` 。

    输出：
        - **output** (Tensor) - scatter 操作结果的张量。
    
    异常：
        - **ValueError** - 如果 `mode` 非法。
