# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""log utils"""
import logging
import os
import mindspore as ms

from sciai.utils.time_utils import time_str


def set_log(args):
    """
    Set log configurations according to args namespace. The attributes `model_name`, `problem` and `amp_level` are used
        to construct the log name.

    Args:
        args (Namespace): argument namespace.
    """
    if hasattr(args, "log_path"):
        project_name_list = [args.model_name if hasattr(args, "model_name") else "model"]
        if hasattr(args, "problem"):
            project_name_list.append(args.problem)
        if hasattr(args, "amp_level"):
            project_name_list.append(args.amp_level)
        if hasattr(args, "model_name") and args.model_name == "ppinns":
            device_id = ms.get_context("device_id")
            project_name_list.append(str(device_id))
        project_name = "_".join(project_name_list)
        log_config(args.log_path, project_name)


def log_config(directory, log_name="model"):
    """
    Log configuration.

    Args:
        directory (str): Directory to save log.
        log_name (str): Log name prefix. Default: "model".
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(directory, f"{log_name}_{time_str()}.log"),
                        filemode='w')


def print_log(*msg, level=logging.INFO, enable_log=True):
    """
    Print in the standard output stream as well as into the log file.

    Args:
        *msg (any): Message(s) to print and log.
        level (int): Log level. Default: logging.INFO.
        enable_log (bool): Whether to log the message. In some cases, like before logging configuration, this flag would
            be set as False. Default: True.
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
