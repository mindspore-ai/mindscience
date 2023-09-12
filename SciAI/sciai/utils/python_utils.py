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
"""python utils"""
import argparse
import glob
import os
import shlex
import subprocess
import sys
import time
from functools import wraps

import yaml
from filelock import FileLock

from sciai.utils.check_utils import _Attr
from sciai.utils.file_utils import _is_folder_non_empty
from sciai.utils.log_utils import print_log


class _LazyProperty:
    """
    Lazy property decorator class.

    Args:
        func (Callable): Function to be decorated.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val


def lazy_property(func):
    """
    Lazy property decorator signature.

    Args:
        func (Callable): decorated arguments.

    Returns:
        _LazyProperty, Decorated object.
    """
    return _LazyProperty(func)


def lazy_func(func, *args, **kwargs):
    """
    Fabricate a lazy function which can be directly called later without arguments.

    Args:
        func (Callable): The function to be lazily loaded.
        *args (any): All non-keyword arguments for `func`.
        **kwargs(any): All keyword arguments for `func`.

    Returns:
        Function, the fabricateed lazy function without arguments.

    Examples:
        >>> from sciai.utils import lazy_func
        >>> def funct(a):
        >>>     print(a)
        >>> lazy_f = lazy_func(funct, "printing")
        >>> lazy_f()
        printing
    """

    def lz_func(*_, **__):
        return func(*args, **kwargs)

    return lz_func


class _ArgParserTst:
    """
    Arguments parser class for test cases.
    """


def parse_arg(config):
    """
    Parse arguments according to terminal/bash inputs and config dictionary.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        Union(Namespace, object), Representation of configurations in `Namespace` or `object`.
    """
    if _running_under_pytest():
        args = _ArgParserTst()
        setattr(args, "device_id", None)
        for key, value in config.items():
            setattr(args, key, value)
    else:
        parser = argparse.ArgumentParser(description=config.get("case"))
        parser.add_argument(f'--device_id', type=int, default=None)
        for key, value in config.items():
            value_type = type(value)
            if value_type in (str,):
                parser.add_argument(f'--{key}', type=str, default=value)
            elif value_type in (int,):
                parser.add_argument(f'--{key}', type=int, default=value)
            elif value_type in (float,):
                parser.add_argument(f'--{key}', type=float, default=value)
            elif value_type in (list, tuple) and value:
                elem_type = type(value[0])
                parser.add_argument(f'--{key}', nargs='+', type=elem_type, default=value)
            elif value_type in (bool,):
                parser.add_argument(f'--{key}', type=lambda x: (str(x).lower() == 'true'), default=value)
            elif value is None:
                parser.add_argument(f'--{key}', type=int, default=None)
            else:
                raise Exception(f"unrecognized data type in config '{key}':''{type(value)}")
        args = parser.parse_args()
    return args


def _running_under_pytest():
    """
    Judge whether it's currently on pytest.

    Returns:
        boolean, True if on pytest, otherwise False.
    """
    return 'pytest' in sys.modules


def download_resource(model_name: str, is_force=False):
    """
    Download the dataset and(or) checkpoint files for model named `model_name`.
    If the model config contains "data_status", then it will download data according to "remote_data_path"(if it has) or
    "model_path".

    Args:
        model_name (str): The name of target model.
        is_force (bool): Whether download the dataset by force.

    Raises:
        ValueError: If `model_name` is not a supported model name.
    """
    all_status = _load_model_configs()
    if model_name not in all_status:
        raise ValueError(f"model {model_name} is not included")
    model_status = all_status.get(model_name)
    if "remote_data_path" not in model_status:
        return
    data_path = model_status.get("remote_data_path")
    data_status_folder = _is_folder_non_empty("checkpoints")
    data_status_config = model_status.get("data_status")
    cmd_download = f'wget -r -np -nH -R *.html* ' \
                   f'https://download.mindspore.cn/{data_path}/ --no-check-certificate '
    cmd_download = shlex.split(cmd_download)
    if is_force or (not data_status_config and not data_status_folder):
        try:
            res = subprocess.Popen(cmd_download, stdout=subprocess.PIPE, shell=False)
            res.communicate(timeout=2000)
            cmd_copy = ['cp', '-rf'] + glob.glob(f'{data_path}/*') + ['./']
            subprocess.Popen(cmd_copy, stdout=subprocess.PIPE, shell=False).communicate(timeout=100)
            print_log(f"Data downloaded to current directory successfully.")
        except IOError as _:
            print_log("failed to download resources due to system error.")
        except subprocess.TimeoutExpired:
            print_log("Download resources time expired")
        finally:
            cmd_clear = ['rm', '-rf', 'mindscience']
            subprocess.Popen(cmd_clear, stdout=subprocess.PIPE, shell=False).communicate(timeout=100)
    else:
        print_log(f"Data is already downloaded.")


def _update_data_status(model_name, status):
    """update yaml file, only called after download data"""
    model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../model')
    lock_dir = os.path.abspath(os.path.join(model_dir, '.cache'))
    file_path = os.path.abspath(os.path.join(model_dir, 'model_status.yaml'))
    os.makedirs(lock_dir, exist_ok=True)
    lock = FileLock(os.path.join(lock_dir, 'model_status.yaml.lock'), timeout=5)
    with lock:
        with open(file_path, 'r') as f:
            all_status = yaml.safe_load(f)
        all_status[model_name]["data_status"] = status
        with open(file_path, 'w') as f:
            yaml.dump(all_status, f)


def _load_model_configs():
    """load model configs"""
    model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../model')
    lock_dir = os.path.abspath(os.path.join(model_dir, '.cache'))
    file_path = os.path.abspath(os.path.join(model_dir, 'model_status.yaml'))
    os.makedirs(lock_dir, exist_ok=True)
    lock = FileLock(os.path.join(lock_dir, 'model_status.yaml.lock'), timeout=5)
    with lock:
        with open(file_path, 'r') as f:
            all_status = yaml.safe_load(f)
    return all_status


def print_args(args):
    """
    Print and log args line by line.

    Args:
        args (Namespace): arguments namespace.
    """
    print_log("\n")
    for k, v in _Attr.all_items(args):
        print_log(f"{k} : {v}")
    print_log("\n")


def print_time(task):
    """
    Print end-to-end time elapsed for a function.

    Args:
        task (Callable): Decorated function task name.

    Returns:
        Wrapper function which would calculate and print time elapsed.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print_log(f"Total time running {task}: {elapsed_time} seconds.")
            return res

        return wrapper

    return decorate
