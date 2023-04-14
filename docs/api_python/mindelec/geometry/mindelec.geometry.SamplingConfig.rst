mindelec.geometry.SamplingConfig
================================

.. py:class:: mindelec.geometry.SamplingConfig(part_sampling_dict)

    全局采样配置定义。

    参数：
        - **part_sampling_dict** (dict) - 采样配置，可配置键为 ``"domain"``， ``"BC"``， ``"IC"`` 或 ``"time"``。

          - ``"domain"``：问题的可行域。
          - ``"BC"``：问题的边界条件。
          - ``"IC"``：问题的初始条件。
          - ``"time"``：问题的时域。

    异常：
        - **ValueError** - 如果 `coord_min` 或 `coord_max` 既不是int也不是float。
        - **TypeError** - 如果 `part_sampling_dict` 不是dict。
        - **KeyError** - 如果 `geom_type` 不是 ``"domain"``、 ``"BC"``、 ``"IC"`` 或 ``"time"``。
        - **TypeError** - 如果'config'不是PartSamplingConfig对象。
        - **ValueError** - 如果 `self.domain.size` 既不是list也不是tuple。
        - **ValueError** - 如果 `self.ic.size` 既不是list也不是tuple。
        - **ValueError** - 如果 `self.time.size` 既不是list也不是tuple。
