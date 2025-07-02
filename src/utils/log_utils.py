"""log utils"""
import logging
import os


def log_config(log_dir='./logs', model_name="model", permission=0o644):
    """
    Log configuration.
    Args:
        log_dir (str): Directory to save log.
        model_name (str): Project name as prefix of log. Default: "model".
    """
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = os.path.join(log_dir, f"{model_name}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode='w')
    os.chmod(log_path, permission)


def print_log(*msg, level=logging.INFO, enable_log=True):
    """
    Print in the standard output stream as well as into the log file.
    Args:
        *msg (any): Message(s) to print and log.
        level (int): Log level. Default: logging.INFO.
        enable_log (bool): Whether to log the message. In some cases, like before logging configuration, this flag would
            be set as False. Default: ``True``.
    """

    def log_help_func(*messages):
        if not enable_log:
            return
        if len(messages) == 1:
            logging.log(level=level, msg=messages[0])
        else:
            logging.log(level=level, msg=", ".join([str(_) for _ in messages]))

    print_funcs = [print, log_help_func]
    for func in print_funcs:
        func(*msg)
