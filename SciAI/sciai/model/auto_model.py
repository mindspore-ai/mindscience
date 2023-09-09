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
# ============================================================================
"""AutoModel class"""
import difflib
import importlib
import inspect
import os
import sys
from argparse import Namespace

import mindspore as ms

from sciai.context import init_project
from sciai.utils import FunctionType, print_log, to_tuple
from sciai.utils.check_utils import _Attr
from sciai.utils.file_utils import _isfile, _get_automodel_module_name
from sciai.utils.python_utils import _load_model_configs, lazy_func

_SUITE_MODELS_DIR = "suite_models"


class AutoModel:
    """
    AutoModel is the API for all networks in SciAI
    Example:
    >>> from sciai.model import AutoModel
    >>> model = AutoModel.from_pretrained("cpinns")
    >>> model.update_config(load_data_path="your_data_path")
    >>> model.train()
    >>> model.evaluate()
    """

    def __init__(self):
        raise Exception("Calling __init__ is not allowed.")

    @staticmethod
    def from_pretrained(model_name, problem=None):
        """instantiate model with model name"""
        all_status = _load_model_configs()
        if model_name not in all_status:
            raise ValueError(f"model `{model_name}` is not supported")
        model_kit = all_status.get(model_name).get("kit")

        if model_kit in ["SciAI", "MindElec", "MindFlow"]:
            return AutoSciAI(model_name, problem)
        if model_kit in ["MindSPONGE"]:
            from sciai.model.auto_mindsponge import AutoSPONGE
            return AutoSPONGE(model_name)
        raise ValueError(f"model `{model_name}` is not supported")

    @staticmethod
    def _show_all_models(all_status, max_len=50):
        """
        Show all model names
        Args:
            all_status (dict): Model dict.
            max_len (int): maximum model name length.
        Return:
            str, the models formatted.
        """
        model_names = list(all_status.keys())
        num_models = len(model_names)
        midpoint = num_models // 2
        res = []
        for i in range(midpoint):
            res.append(f"{i + 1}.{model_names[i]:<{max_len} {i + midpoint}.{model_names[i + midpoint]}:<{max_len}}")
        if num_models % 2:
            res.append(f"{i + 1}.{model_names[-1]:<{max_len}}")
        return "\n".join(res)

    @staticmethod
    def _calc_similarity(all_status, model_name, threshold=0.6):
        """
        Calculate similarity between each model and given model name.
        """
        similar_models = []
        for filter_model_name in all_status.keys():
            similarity = difflib.SequenceMatcher(None, model_name.lower(), filter_model_name.lower()).ratio()
            if similarity >= threshold:
                similar_models.append((filter_model_name, similarity))
        similar_models.sort(key=lambda x: x[1], reverse=True)
        return " or ".join(list(map(lambda x: f"`{x[0]}`", similar_models)))


class AutoSciAI(_Attr):
    """High level API class for SciAI"""

    def __init__(self, model_name, problem=None):
        """initialization"""
        super(AutoSciAI, self).__init__()
        self.automodel_path = os.path.abspath(os.path.dirname(__file__))
        self.exec_path = os.getcwd()
        self.mode = ms.GRAPH_MODE
        self._dict_functions = {FunctionType.TRAINER: None, FunctionType.VAL: None, FunctionType.TRAIN_PREPARE: None}
        self.all_status = _load_model_configs()
        self.model_name = model_name
        self._register_model()
        if _is_normal_suite(self.all_status, self.model_name):
            self.model_full_path = os.path.join(os.path.join(self.automodel_path, "suite_models"), model_name)
        else:
            self.model_full_path = os.path.join(self.automodel_path, model_name)
        self.local_module_name = _get_automodel_module_name(model_name)

        self.func_train = self._dict_functions.get(FunctionType.TRAINER)
        self.func_prepare = self._dict_functions.get(FunctionType.TRAIN_PREPARE)
        self.func_val = self._dict_functions.get(FunctionType.VAL)
        self.func_args = self.__set_args(self.__get_func_args(problem), self._check_make_dir)

    def update_config(self, **kwargs):
        """update func args"""
        for key, value in kwargs.items():
            if key == "mode":
                self.mode = value
                continue
            if self.hasattr(self.func_args[0], key):
                self.setattr(self.func_args[0], key, value)
            else:
                raise ValueError(f"Unknown keyword: {key}. All keywords are: {self.all_attr(self.func_args[0])}")

    def train(self):
        """run train"""
        self._run_func(self.func_train)

    def finetune(self, load_ckpt_path=None):
        """finetune the model"""
        if not isinstance(load_ckpt_path, str):
            raise ValueError(f"Please give a checkpoint path.")
        if not load_ckpt_path.endswith(".ckpt"):
            raise ValueError(f"Invalid checkpoint file: `{load_ckpt_path}`")

        abs_ckpt_path = os.path.abspath(load_ckpt_path)
        self.setattr(self.func_args[0], "load_ckpt_path", abs_ckpt_path)
        self.setattr(self.func_args[0], "load_ckpt", True)
        self._run_func(self.func_train)

    def evaluate(self):
        """run eval"""
        self._run_func(self.func_val)

    def _run_func(self, func):
        """run a train/eval func with context setting and directory change."""
        try:
            os.makedirs(self.local_module_name, exist_ok=True)
            os.chdir(self.local_module_name)
            ms.set_context(compile_cache_path=self.exec_path)
            if _is_normal_suite(self.all_status, self.model_name):
                self.func_args = self.__set_args(self.func_args, self._set_third_party_path)
            init_project(mode=self.mode, args=self.func_args[0])
            func(*self.func_args)
        finally:
            os.chdir(self.exec_path)

    def __get_func_args(self, problem):
        """run prepare"""
        params_dict = inspect.signature(self.func_prepare).parameters
        if "problem" in params_dict:
            func_args = self.func_prepare(problem=problem)
        else:
            func_args = self.func_prepare()
        func_args = to_tuple(func_args)  # convert to a tuple, where func_args[0] is Namespace or dict.
        return func_args

    def __set_args(self, func_args, func):
        """setup all paths in args"""
        for arg in self.all_attr(func_args[0]):
            if arg.endswith("_path"):
                path_value = self.getattr(func_args[0], arg)
                if isinstance(path_value, (list, tuple)):
                    type_collection = type(path_value)
                    cwd_path_value = type_collection()
                    for path_value_single in path_value:
                        cwd_path_value_single = func(path_value_single)
                        cwd_path_value += type_collection([cwd_path_value_single])
                    self.setattr(func_args[0], arg, cwd_path_value)
                else:
                    self.setattr(func_args[0], arg, func(path_value))
        return func_args

    def _check_make_dir(self, path_value_single):
        """check a single path value and make directory if its corresponding directory doesn't exist."""
        cwd_path_value_single = os.path.join(self.exec_path, self.local_module_name, path_value_single)
        to_create_dir = cwd_path_value_single
        if _isfile(to_create_dir):
            to_create_dir = os.path.dirname(to_create_dir)
        os.makedirs(to_create_dir, exist_ok=True)
        return cwd_path_value_single

    def _set_third_party_path(self, path_value):
        """set path to be third party path"""
        if not os.path.exists(path_value):
            orig_path = path_value.split(self.exec_path)[1].split(self.local_module_name)[1]
            if orig_path:
                orig_path = orig_path[1:]
            return os.path.join(self.model_full_path, orig_path)
        return path_value

    def _register_model(self):
        """
        Register model to function dictionary.
        """
        dict_functions, model_name, all_status = self._dict_functions, self.model_name, self.all_status
        automodel_dir = os.path.abspath(os.path.dirname(__file__))
        if all_status.get(self.model_name).get("kit") in ["MindFlow", "MindElec"]:
            model_dir = os.path.join(automodel_dir, _SUITE_MODELS_DIR, model_name)
        else:
            model_dir = os.path.join(automodel_dir, model_name)
        sys.path.append(model_dir)
        internal_models = {model_name for model_name in all_status if all_status.get(model_name).get("kit") == "SciAI"}
        external_models = {k: v for k, v in all_status.items() if k not in internal_models}
        if model_name in internal_models:
            train_func_name, train_module = "main_train", importlib.import_module("__init__")
            eval_func_name, eval_module = "main_eval", importlib.import_module("__init__")
            train_args_func_name, train_args_module = "prepare", importlib.import_module("__init__")
            eval_args_func_name, eval_args_module = "prepare", importlib.import_module("__init__")
        elif model_name in external_models:
            os.environ["MS_JIT_MODULES"] = "src"
            model_funcs = external_models.get(model_name).get("funcs")
            if model_funcs is None:
                raise ValueError(f"model funcs of `{model_name}` is not subscribed.")
            train_err_func = lazy_func(_print_raise, f"`train` function is not supported for model `{model_name}`, "
                                                     f"Please try `evaluation` for this model", ValueError)
            eval_err_func = lazy_func(_print_raise, f"`validate` function is not supported for model `{model_name}`"
                                                    f"Please try `train` for this model", ValueError)
            train_module, train_func_name \
                = _calc_module_func(model_funcs, "train_module", "train_func", model_name, train_err_func)
            eval_module, eval_func_name \
                = _calc_module_func(model_funcs, "eval_module", "eval_func", model_name, eval_err_func)
            train_args_module, train_args_func_name \
                = _calc_module_func(model_funcs, "train_args_module", "train_args_func", model_name, train_err_func)
            eval_args_module, eval_args_func_name \
                = _calc_module_func(model_funcs, "eval_args_module", "eval_args_func", model_name, eval_err_func)
        else:
            raise ValueError(f"model `{model_name}` is not supported")

        dict_functions[FunctionType.TRAINER] = getattr(train_module, train_func_name)
        dict_functions[FunctionType.VAL] = getattr(eval_module, eval_func_name)
        dict_functions[FunctionType.TRAIN_PREPARE] = getattr(train_args_module, train_args_func_name)
        dict_functions[FunctionType.EVAL_PREPARE] = getattr(eval_args_module, eval_args_func_name)


def _print_raise(msg, exception_type=Exception):
    """
    Print and raise exception.
    Args:
        msg (str): Error message.
        exception_type (type): Exception type.
    """
    print_log(msg)
    raise exception_type(msg)


def _calc_module_func(model_funcs, target_module, target_func, model_name, err_func="func"):
    """
    Calculate module instance and function name.
    Args:
        model_funcs (dict)
    """
    script_name = model_funcs.get(target_module)
    if script_name is None:
        module = Namespace()
        setattr(module, target_func, err_func)
        func_name = target_func
    else:
        module = importlib.import_module('.'.join([f"sciai.model.{_SUITE_MODELS_DIR}", model_name, script_name]))
        func_name = model_funcs.get(target_func)
    return module, func_name


def _is_normal_suite(all_status, model_name):
    """
    Judge whether the model is in normal suite. Normal suite: non-sciai and code-exposed suites.
    Args:
        all_status (dict): Configuration file model_status dict.
        model_name (str): Model name string.
    """
    return all_status.get(model_name).get("kit") in ["MindFlow", "MindElec"]
