sciai.common.Sampler
============================================

.. py:class:: sciai.common.Sampler(dim, coords, func, name=None)

    常用的数据采样器。

    参数：
        - **dim** (int) - 数据的维度。
        - **coords** (Union[array, list]) - 下界坐标和上界坐标，例如[[0.0, 0.0], [0.0, 1.0]]。
        - **func** (Callable) - 精确解函数。
        - **name** (str) - 采样器名称。默认值：None。

    .. py:method:: sciai.common.Sampler.fetch_minibatch(n, mu_x, sigma_x)

        从采样器采出一个minibatch的数据。

        参数：
            - **n** (int) - 一个minibatch的数据点个数。
            - **mu_x** (int) - 采样点的均值。
            - **sigma_x** (int) - 采样点的方差。

        返回：
            tuple[Tensor]，一个minibatch的正则化后的采样点。

    .. py:method:: sciai.common.Sampler.normalization_constants(n)

        归一化均值与标准差。

        参数：
            - **n** (int) - 用于计算均值与标准差的采样点个数。

        返回：
            tuple[Tensor]，采样点的均值与方差。

    .. py:method:: sciai.common.Sampler.sample(n)

        在指定区域中采样。

        参数：
            - **n** (int) - 采样点个数。

        返回：
            tuple[Tensor]，`n` 个采样点的x与y。