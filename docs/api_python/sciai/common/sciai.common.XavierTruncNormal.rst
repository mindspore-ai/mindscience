sciai.common.XavierTruncNormal
============================================

.. py:class:: sciai.common.XavierTruncNormal(trunc_interval=(-2, 2))

    Xavier截断正态初始化，丢弃Xavier正态初始化平均值附近2倍标准差外的点。

    参数：
        - **trunc_interval** (Union[None, tuple[Number]]) - 正态分布截断区间。 若为 `(-2, 2)` ，则丢弃并重新采集任何
          与均值 0 相差超过2个标准差的样本点。默认值：`(-2, 2)`。
