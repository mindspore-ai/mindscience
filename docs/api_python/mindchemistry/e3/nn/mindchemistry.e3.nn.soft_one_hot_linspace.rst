mindchemistry.e3.nn.soft_one_hot_linspace
==========================================

.. py:function:: mindchemistry.e3.nn.soft_one_hot_linspace(x, start, end, number, basis='smooth_finite', cutoff=True)

    投影到函数基上。返回一组 :math:`\{y_i(x)\}_{i=1}^N`，

    .. math::
        y_i(x) = \frac{1}{Z} f_i(x)

    其中 :math:`x` 是输入，:math:`f_i` 是第 i 个基函数。
    :math:`Z` 是一个常数（如果可能的话定义），使得：

    .. math::
        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    注意 `bessel` 基函数不能被归一化。

    参数：
        - **x** (Tensor) - 形状为 :math:`(...)` 的张量。
        - **start** (float) - 基函数区间最小值。
        - **end** (float) - 基函数区间最大值。
        - **number** (int) - 基函数的数量 :math:`N`。
        - **basis** (str) - {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}，基函数的种类。默认值：``'smooth_finite'`` 。
        - **cutoff** (bool) - 是否要求 :math:`y_i(x)` 在域 (`start`, `end`) 外取值为零。默认值：``True`` 。

    返回：
        shape为 :math:`(..., N)` 的Tensor。

    异常：
        - **ValueError** - 如果 `basis` 不是 {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'} 之一。




