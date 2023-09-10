sciai.common.LbfgsOptimizer
============================================

.. py:class:: sciai.common.LbfgsOptimizer(closure, weights)

    L-BFGS 二阶优化器，目前仅在 `PYNATIVE_MODE` 中支持。

    参数：
        - **closure** (Callable) - 返回损失的函数。
        - **weights** (list[Parameter]) - 需要优化的参数。

    输入：
        - **options** (映射[str，任何]) - 参考 mindspore.scipy.minimize。

    输出：
        OptimizeResults，保存优化结果的对象。