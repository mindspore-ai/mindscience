mindearth.utils.create_logger
==============================================

.. py:function:: mindearth.utils.create_logger(path="./log.log", level=logging.INFO)

    创建一个日志系统。

    参数：
        - **path** (str) - 存放日志的路径。默认值： ``"./log.log"``。
        - **level** (int) - 日志等级。默认值： ``logging.INFO``。

    返回：
        logging.RootLogger，记录信息的日志系统。
