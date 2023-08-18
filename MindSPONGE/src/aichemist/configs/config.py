# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
config
"""

import re
import os
from datetime import time, timedelta, date
from collections import defaultdict
from typing import Callable, Union

from numpy import ndarray
from mindspore import Tensor, Parameter
from ..utils import read_yaml


class Tree(defaultdict):
    """
    core.Tree
    """

    def __init__(self):
        super().__init__(Tree)

    def flatten(self, prefix: str = None, result: dict = None):
        """
        Transform the Tree structure to dict. The key is all of the names joined by `.`

        Args:
            prefix (str, optional): The prefix of flattened keys. Defaults to None.
            result (dict, optional): The dict that contains reserved registered classes. Defaults to None.

        Returns:
            result: updated output that contained all of  hierarchical registered keys,
                    which has the same structure of input result.
        """
        if prefix is None:
            prefix = ""
        else:
            prefix = prefix + "."
        if result is None:
            result = {}
        for k, v in self.items():
            if isinstance(v, Tree):
                v.flatten(prefix + k, result)
            else:
                result[prefix + k] = v
        return result


class Registry:
    """
    Registry class for managing all call-by-name access to objects.
    Typical scenarios:
    1. Create a model according to a string.
    >>> gcn = R.search("GCN")(128, [128])
    2. Register a customize hook to the package.
    >>> @R.register("features.atom.my_feature")
    >>> def my_featurizer(atom):
    >>>     ...
    >>>
    >>> data.Molecule.from_smiles("C1=CC=CC=C1", atom_feature="my_feature")
    """

    table = Tree()
    order = []

    @classmethod
    def register(cls, name):
        """Register an object with a canonical name. Hierarchical names are separated by ``.``.

            Args:
                name (str): The name of registered class

            Raises:
                KeyError: The name should be unique in the Tree structure.

            Returns:
                wrapper: The callable wrapper function for class registration.
        """
        def wrapper(obj):
            entry = cls.table
            keys = name.split(".")
            for key in keys[:-1]:
                entry = entry[key]
            if keys[-1] in entry:
                raise KeyError(f"`{name}` has already been registered by {entry[keys[-1]]}")
            index = len(cls.order)
            cls.order.append(obj)
            entry[keys[-1]] = index
            obj.cls_name = name

            return obj

        return wrapper

    @classmethod
    def get(cls, category: str, cls_name: str = None):
        """
        Get an object with a canonical name. Hierarchical names are separated by `.`
        the name for indexing is the string join by categroy and class_name with `.`
        If cls_name is not given, category will be used as key for indexing.

        Args:
            category (str): category of registered name
            cls_name (str, optional): class name. Defaults to None.

        Raises:
            KeyError: The given name does not existed in the given category of registered Tree.

        Returns:
            obj: The class indexed with given name in category of registered Tree.
        """
        entry = cls.table
        keys = category.split('.') + cls_name.split(".") if cls_name else category.split('.')
        for i, key in enumerate(keys):
            if key not in entry:
                value = ".".join(keys[:i])
                raise KeyError(f"Can't find `{key}` in `{value}`")
            entry = entry[key]
        obj = cls.order[entry]
        return obj

    @classmethod
    def get_id(cls, cls_name):
        """
        Get the id of the object with a canonical name. Hierarchical names are separated by ``.``.

        Args:
            cls_name (str, optional): registered name of class. Defaults to None.

        Raises:
            KeyError: The given name does not existed in the registered Tree.

        Returns:
            obj: The class indexed with given name in category of registered Tree.
        """
        entry = cls.table
        keys = cls_name.split(".")
        for i, key in enumerate(keys):
            if key not in entry:
                value = ".".join(keys[:i])
                raise KeyError(f"Can't find `{key}` in `{value}`")
            entry = entry[key]
        return entry

    @classmethod
    def put(cls, obj: Callable, cls_name: str):
        """

        Put an class object with a canonical name to the registered Tree. Hierarchical names are separated by ``.``.

        Args:
            obj (Callable): The type of class need to be registered.
            cls_name (str): The registered name of the input class.

        Raises:
            KeyError: the class name should not exist in the registered Tree ever.
        """

        entry = cls.table
        keys = cls_name.split(".")
        for key in keys[:-1]:
            entry = entry[key]
        if keys[-1] in entry:
            value = entry[keys[-1]]
            raise KeyError(f"`{cls_name}` has already been registered by {value}")
        index = len(cls.order)
        cls.order.append(obj)
        entry[keys[-1]] = index
        obj.cls_name = cls_name

    @classmethod
    def search(cls, path, cls_name=None):
        """
        Get an object with a canonical name. Hierarchical names are separated by `.`
        the name for indexing is the string join by categroy and class_name with `.`
        If cls_name is not given, category will be used as key for indexing.

        Note:
            The name doesn't need to be canonical.
            For example, we can search ``GCN`` and get the object of ``models.GCN``.

        Args:
            category (str): category of registered name
            cls_name (str, optional): class name. Defaults to None.

        Raises:
            KeyError: The given name does not existed in the given category of registered Tree.

        Returns:
            value (Callable): The class indexed with given name in category of registered Tree.
        """

        keys = []
        if cls_name is not None:
            path = path + '.' + cls_name
        pattern = re.compile(rf"\b{path}\b")
        for k, v in cls.table.flatten().items():
            if pattern.search(k):
                keys.append(k)
                value = cls.order[v]
        length = len(keys)
        if length == 0:
            raise KeyError(f"Can't find any registered key containing `{cls_name}`")
        if length > 1:
            keys = ', '.join([f"`{key}`" % key for key in keys])
            raise KeyError(f"Ambiguous key `{cls_name}`. Found {keys}")
        return value

    @classmethod
    def build(cls, path: str, cls_name: str = None, **kwargs):
        """Initialize an object by the queried class name. Hierarchical names are separated by ``.``.

        Args:
            path (str): The categorical name
            name (str, optional): name. Defaults to None.

        Returns:
            obj: The object with the type of searched results.
        """
        if isinstance(cls_name, Callable):
            return cls_name
        if isinstance(cls_name, dict):
            name = cls_name.pop('cls_name')
            kwargs.update(cls_name)
        else:
            name = cls_name
        fn = cls.search(path, cls_name=name)
        obj = fn(**kwargs)
        return obj


class Config(type):
    """
    Class for load/save configuration.
    It will automatically record every argument passed to the ``__init__`` function.
    This class is inspired by :meth:`state_dict()` in PyTorch, but designed for hyperparameters.
    Inherit this class to construct a configurable class.
    >>> class MyClass(nn.Cell, core.Configurable):
    Note :class:`Configurable` only applies to the current class rather than any derived class.
    For example, the following definition only records the arguments of ``MyClass``.
    >>> class DerivedClass(MyClass):
    In order to record the arguments of ``DerivedClass``, explicitly specify the inheritance.
    >>> class DerivedClass(MyClass, core.Configurable):
    To get the configuration of an instance, use :meth:`config_dict()`,
    which returns a dict of argument names and values.
    If an argument is also an instance of :class:`Configurable`, it will be recursively expanded in the dict.
    The configuration dict can be passed to :meth:`load_config()` to create a copy of the instance.
    For classes already registered in :class:`Registry`,
    they can be directly created from the :class:`Configurable` class.
    This is convenient for building models from configuration files.
    >>> config = models.GCN(128, [128]).config_dict()
    >>> gcn = Config.load_config(config)
    """
    @classmethod
    def load_config(mcs, config):
        """
        Configurable.load_config
        """
        name = config["cls_name"]
        if mcs == Config:
            cls_name = Registry.search(name)
            load_func = getattr(cls_name, 'load_config', None)
            if load_func:
                return cls_name.load_func(config)
            mcs_ = cls_name
        elif getattr(mcs, "cls_name", mcs.__name__) != name:
            raise ValueError(f"Expect config cls_name to be `{mcs.__name__}`, but found `{name}`")

        kwargs = {}
        for k, v in config.items():
            if isinstance(v, dict) and "cls_name" in v:
                v = Config.load_config(v)
            elif isinstance(v, list):
                v = [Config.load_config(v_)
                     if isinstance(v_, dict) and "cls_name" in v_ else v_
                     for v_ in v]
            if k != "cls_name":
                kwargs[k] = v
        return mcs_(**kwargs)

    def config_dict(cls):
        """
        config_dict
        """
        def unroll_config_dict(obj):
            if isinstance(type(obj), Config):
                obj = obj.config_dict()
            elif isinstance(obj, (str, bytes)):
                return obj
            elif isinstance(obj, dict):
                return type(obj)({k: unroll_config_dict(v) for k, v in obj.items()})
            elif isinstance(obj, (list, tuple)):
                return type(obj)(unroll_config_dict(x) for x in obj)
            return obj

        cls_ = getattr(cls, "cls_name", cls.__class__.__name__)
        config = {"cls_name": cls_}
        for k, v in cls_.configs.items():
            config[k] = unroll_config_dict(v)
        return config

    @staticmethod
    def get_configure(configure: Union[str, dict], key: str = None) -> dict:
        """ Get template for molecule or residue.

        Args:
            configure (Union[dict, str):

        Returns:
            template (dict):  Template for molecule or residue

        """
        if configure is None:
            return configure

        if isinstance(configure, str):
            if os.path.exists(configure):
                filename = configure
            else:
                directory, _ = os.path.split(os.path.realpath(__file__))
                filename = os.path.join(directory, configure)
                if not os.path.exists(filename):
                    raise ValueError(f'Cannot find configure file: {configure}')
            configure: dict = read_yaml(filename)

        if not isinstance(configure, dict):
            raise TypeError(f'The type of configure must be str or dict but got: {type(configure)}')

        if key is not None:
            if key in configure.keys():
                return configure.get(key)
            raise KeyError(f'Cannot find key "{key}" in configure.')

        return configure

    @staticmethod
    def get_arguments(locals_: dict, kwargs: dict = None) -> dict:
        r"""get arguments of a class

        Args:
            locals_ (dict): Dictionary of the arguments from `locals()`.
            kwargs (dict): Dictionary of keyword arguments (kwargs) of the class.

        Returns:
            args (dict): Dictionary of arguments

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        """

        if '__class__' in locals_.keys():
            locals_.pop('__class__')

        arguments = {}
        if 'self' in locals_.keys():
            cls = locals_.pop('self')
            if hasattr(cls, 'reg_key'):
                cls_name = cls.reg_key.split('.')[-1]
            else:
                cls_name = cls.__class__.__name__
            arguments['cls_name'] = cls_name

        def _set_arguments(args_: dict):
            def _convert(value):
                if value is None or isinstance(value, (int, float, bool, str,
                                                       time, timedelta, date)):
                    return value
                if isinstance(value, ndarray):
                    return value.tolist()
                if isinstance(value, (Tensor, Parameter)):
                    return value.asnumpy().tolist()
                if isinstance(value, (list, tuple)):
                    return [_convert(v) for v in value]
                if isinstance(value, dict):
                    if 'cls_name' in value.keys():
                        return value
                    dict_ = value.copy()
                    for k, v in value.items():
                        dict_[k] = _convert(v)
                    return dict_

                cls_name = value.__class__.__name__
                if hasattr(value, '_kwargs'):
                    value = value.__dict__['_kwargs']
                elif hasattr(value, 'init_args'):
                    value = value.__dict__['init_args']
                else:
                    value = value.__class__.__name__

                if isinstance(value, dict) and 'cls_name' not in value.keys():
                    dict_ = {'cls_name': cls_name}
                    dict_.update(_set_arguments(value))
                    value = dict_

                return value

            for k, v in args_.items():
                args_[k] = _convert(v)
            return args_

        kwargs_ = {}
        if 'kwargs' in locals_.keys():
            kwargs_: dict = locals_.pop('kwargs')

        if kwargs is None:
            kwargs = kwargs_

        if 'cls_name' in kwargs.keys():
            kwargs.pop('cls_name')

        arguments.update(_set_arguments(locals_))
        arguments.update(_set_arguments(kwargs))

        return arguments
