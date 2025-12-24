mindscience.sciops.fft.set_fft_cache_size
==========================================

.. py:function:: mindscience.sciops.fft.set_fft_cache_size(cache_size)

    设置ASD FFT算子的缓存数量以优化函数调用性能。
    如果没有缓存，每次ASD FFT函数调用都会从.so文件中动态dlopen函数符号，
    这会引入一些开销。
    使用缓存后，函数符号将在第一次ASD FFT函数调用时加载到缓存中，
    在后续调用中将不再重新加载。

    参数：
        - **cache_size** (int) - ASD FFT算子的缓存数量。
