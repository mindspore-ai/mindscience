sciai.utils.print_log
=======================

.. py:function:: sciai.utils.print_log(*msg, level=logging.INFO, enable_log=True)

    在标准输出流和日志文件中打印。

    参数：
        - **\*msg** (any) - 要打印和记录的消息。
        - **level** (int) - 日志级别。 默认值：`logging.INFO`。
        - **enable_log** (bool) - 是否记录消息。 在某些情况下，比如在记录配置之前，这个标志会被设置为False。默认值：True。