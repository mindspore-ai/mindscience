mindscience.utils.log_timer
==========================

.. py:function:: mindscience.utils.log_timer(func)

    用于统计训练步骤端到端总耗时的装饰器。

    参数：
        - **func** (callable) - 被装饰的函数。

    返回：
        - callable - 包装后的函数，会在执行结束后打印端到端总耗时。