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
"""High level API for MindSPONGE"""
import os.path
import pickle

from mindsponge import PipeLine

from sciai.utils import print_log, log_config
from sciai.utils.python_utils import download_resource, _load_model_configs
from sciai.utils.file_utils import make_sciai_dirs


class AutoSPONGE:
    """ Wrapper class for MindSPONGE PipeLine"""

    # APIs as string in mindsponge.PipeLine
    _str_initialize = "initialize"
    _str_set_device_id = "set_device_id"
    _str_predict = "predict"
    _str_train = "train"
    _str_save_model = "save_model"

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.pipe_line = PipeLine(model_name)
        self.func_list = []

        self.model_status = _load_model_configs().get(model_name)
        self.trainable = "train" in self.model_status.get("modes")
        self.predictable = "predict" in self.model_status.get("modes")

        self.model_args = self.__init_model_args()
        self.func_args = self.__init_func_args()

        log_config("./", model_name)
        make_sciai_dirs()
        download_resource(self.model_name)

    def train(self, data_source=None, num_epochs=1, **kwargs):
        """Train MindSPONGE model"""
        if not self.trainable:
            print_log(f"Model `{self.model_name}` does not support training, skipped.")
            return
        if data_source is None:
            self.__validate_data_path("data_source", self._str_train)
            self.pipe_line.train(**self.func_args.get(self._str_train), **kwargs)
        else:
            self.pipe_line.train(data_source=data_source, num_epochs=num_epochs, **kwargs)

        if self.model_args.get("save_model"):
            self.pipe_line.save_model(**self.func_args.get(self._str_save_model))

    def finetune(self, data_source=None, num_epochs=1, load_ckpt_path=None, **kwargs):
        """Finetune MindSPONGE model"""
        if not self.trainable:
            print_log(f"Model `{self.model_name}` does not support training or finetune, skipped.")
            return
        if load_ckpt_path is None:
            self.pipe_line.model.from_pretrained(ckpt_path=self.model_args.get("load_ckpt_path"))
        else:
            if not isinstance(load_ckpt_path, str):
                raise ValueError("Invalid load checkpoint path given.")
            if not load_ckpt_path.endswith(".ckpt"):
                raise ValueError(f"Invalid load checkpoint path: `{load_ckpt_path}`")
            self.pipe_line.model.from_pretrained(load_ckpt_path)
        self.train(data_source, num_epochs, **kwargs)

    def evaluate(self, data=None, load_ckpt_path=None, **kwargs):
        """Evaluate MindSPONGE model"""
        if not self.predictable:
            print_log(f"Model `{self.model_name}` does not support prediction, skipped.")
            return None
        if load_ckpt_path is not None:
            if not isinstance(load_ckpt_path, str):
                raise ValueError("Invalid load checkpoint path given.")
            if not load_ckpt_path.endswith(".ckpt"):
                raise ValueError(f"Invalid load checkpoint path: `{load_ckpt_path}`")
            self.pipe_line.model.from_pretrained(load_ckpt_path)
        elif self.model_args.get("load_ckpt") and self.model_args.get("load_ckpt_path") is not None:
            self.pipe_line.model.from_pretrained(ckpt_path=self.model_args.get("load_ckpt_path"))
        elif self.model_args.get("load_ckpt"):
            self.pipe_line.model.from_pretrained(self.model_status.get("default_args").get("ckpt_path"))

        if data is None:
            self.__validate_data_path("data", self._str_predict)
            result = self.pipe_line.predict(**self.func_args.get(self._str_predict), **kwargs)
        else:
            result = self.pipe_line.predict(data, **kwargs)
        return result

    def update_config(self, **kwargs):
        """update func args"""
        for key, value in kwargs.items():
            if key in self.model_args:
                self.model_args[key] = value
            else:
                ret = self.__do_update(key, value)
                if not ret:
                    raise ValueError("Unknown keyword: {}. All keywords are: {}".format(key, dir(self.func_args)))

    def initialize(self, key=None, conf=None, config_path=None, **kwargs):
        """prepare the resource"""
        if key is not None:
            self.update_config(key=key)
        if conf is not None:
            self.update_config(conf=conf)
        if config_path is not None:
            self.update_config(config_path=config_path)

        self.pipe_line.set_device_id(**self.func_args.get(self._str_set_device_id))
        self.pipe_line.initialize(**self.func_args.get(self._str_initialize), **kwargs)

    def __do_update(self, key, value):
        """find the correct function and update"""
        for func, args in self.func_args.items():
            if key in args:
                self.func_args[func][key] = value
                self.model_args["need_init"] = (func == "initialize") or self.model_args.get("need_init")
                return True
        return False

    def __init_func_args(self):
        """init the function arguments"""
        default_args = self.model_status.get("default_args")
        default_data_path = default_args.get("data_path")
        func_args = {
            self._str_initialize: {"key": None, "conf": None, "config_path": None},
            self._str_set_device_id: {"device_id": 0},
            self._str_predict: {"data": default_data_path},
            self._str_train: {"data_source": default_data_path, "num_epochs": 1},
            self._str_save_model: {"ckpt_path": f"./checkpoint/{self.model_name}.ckpt"},
        }
        return func_args

    @staticmethod
    def __init_model_args():
        """init the model arguments"""
        model_args = {
            "save_model": True,
            "load_ckpt_path": None,
            "load_ckpt": False
        }
        return model_args

    def __validate_data_path(self, arg_name, func_name):
        """validate the data path and pre-process if necessary"""
        data_path = self.func_args.get(func_name).get(arg_name)
        if data_path is None:
            raise ValueError(f"Please provide the data input for model `{self.model_name}`, "
                             f"with `update_config({arg_name}=your_data)`")
        if isinstance(data_path, str):
            if data_path.endswith(".pkl"):
                if not os.path.exists(data_path):
                    raise ValueError(f"The data source `{data_path}` does not exist, "
                                     f"please update the `{arg_name}` with `update_config({arg_name}=your_data_path)`")
                with open(data_path, "rb") as file:
                    raw_feature = pickle.load(file)
                    self.func_args[func_name][arg_name] = raw_feature
