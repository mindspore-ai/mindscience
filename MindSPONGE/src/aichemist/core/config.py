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
from collections import defaultdict
from contextlib import contextmanager


class MetaContainer:
    """
    Meta container that maintains meta types about members.
    The meta type of each member is tracked when a member is assigned.
    We use a context manager to define the meta types for a bunch of assignment.
    The meta types are stored as a dict in ``instance.meta_dict``,
    where keys are member names and values are meta types.
    >>> class MyClass(_MetaContainer):
    >>>     ...
    >>> instance = MyClass()
    >>> with instance.context("important"):
    >>>     instance.value = 1
    >>> assert instance.meta_dict["value"] == "important"
    Members assigned with :meth:`context(None) <context>` or without a context won't be tracked.
    >>> instance.random = 0
    >>> assert "random" not in instance.meta_dict
    You can also restrict available meta types by defining a set :attr:`_meta_types` in the derived class.
    .. note::
        Meta container also supports auto inference of meta types.
        This can be enabled by setting :attr:`enable_auto_context` to ``True`` in the derived class.
        Once auto inference is on, any member without an explicit context will be recognized through their name prefix.
        For example, ``instance.node_value`` will be recognized as ``node`` if ``node`` is defined in ``meta_types``.
        This may make code hard to maintain. Use with caution.
    """

    _meta_types = set()
    enable_auto_context = False

    def __init__(self, meta_dict=None, **kwargs):
        if meta_dict is None:
            meta_dict = {}
        else:
            meta_dict = meta_dict.copy()

        self._setattr("_meta_contexts", set())
        self._setattr("meta_dict", meta_dict)
        for k, v in kwargs.items():
            self._setattr(k, v)

    def __setattr__(self, key, value):
        if hasattr(self, "meta_dict"):
            types = self._meta_contexts
            if not types and self.enable_auto_context:
                for meta in self._meta_types:
                    if key.startswith(meta):
                        types.append(meta)
                if len(types) > 1:
                    raise ValueError(f"Auto context found multiple contexts for key `{key}`. "
                                     "If this is desired, set `enable_auto_context` to False "
                                     "and manually specify the context. ")
            if types:
                self.meta_dict[key] = types.copy()
        self._setattr(key, value)

    def __delattr__(self, key):
        if hasattr(self, "meta_dict") and key in self.meta_dict:
            del self.meta_dict[key]
            del self.data_dict[key]
        super().__delattr__(self, key)

    @property
    def data_dict(self):
        """A dict that maps tracked names to members."""
        return {k: getattr(self, k) for k in self.meta_dict}

    @staticmethod
    def _setattr(key, value):
        return super().__setattr__(key, value)

    @staticmethod
    def _standarize_type(types):
        if types is None:
            types = set()
        elif isinstance(types, str):
            types = {types}
        else:
            types = set(types)
        return types

    def data_by_meta(self, include=None, exclude=None):
        """
        Return members based on the specific meta types.
        Parameters:
            include (list of string, optional): meta types to include
            exclude (list of string, optional): meta types to exclude
        Returns:
            (dict, dict): data member dict and meta type dict
        """
        if include is None and exclude is None:
            return self.data_dict, self.meta_dict

        include = self._standarize_type(include)
        exclude = self._standarize_type(exclude)
        types = include or set().union(*self.meta_dict.values())
        types = types - exclude
        data_dict = {}
        meta_dict = {}
        for k, v in self.meta_dict.items():
            if v.issubset(types):
                data_dict[k] = getattr(self, k)
                meta_dict[k] = v
        return data_dict, meta_dict

    @contextmanager
    def context(self, meta_type):
        """
        Context manager for assigning members with a specific meta type.
        """
        if meta_type is not None and self._meta_types and meta_type not in self._meta_types:
            raise ValueError(f"Expect context type in {self._meta_types}, but got `{meta_type}`")
        self._meta_contexts.add(meta_type)
        yield
        self._meta_contexts.remove(meta_type)


class Tree(defaultdict):
    """
    core.Tree
    """
    def __init__(self):
        super().__init__(Tree)

    def flatten(self, prefix=None, result=None):
        """
        Tree.flatten
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
        """
        Register an object with a canonical name. Hierarchical names are separated by ``.``.
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
            # pylint: disable=protected-access
            obj.registry_key_ = name

            return obj

        return wrapper

    @classmethod
    def get(cls, name):
        """
        Get an object with a canonical name. Hierarchical names are separated by ``.``.
        """
        entry = cls.table
        keys = name.split(".")
        for i, key in enumerate(keys):
            if key not in entry:
                value = ".".join(keys[:i])
                raise KeyError(f"Can't find `{key}` in `{value}`")
            entry = entry[key]
        return cls.order[entry]

    @classmethod
    def get_id(cls, name):
        """
        Get an object with a canonical name. Hierarchical names are separated by ``.``.
        """
        entry = cls.table
        keys = name.split(".")
        for i, key in enumerate(keys):
            if key not in entry:
                value = ".".join(keys[:i])
                raise KeyError(f"Can't find `{key}` in `{value}`")
            entry = entry[key]
        return entry

    @classmethod
    def put(cls, obj, name):
        """
        Get an object with a canonical name. Hierarchical names are separated by ``.``.
        """
        entry = cls.table
        keys = name.split(".")
        for key in keys[:-1]:
            entry = entry[key]
        if keys[-1] in entry:
            value = entry[keys[-1]]
            raise KeyError(f"`{name}` has already been registered by {value}")
        index = len(cls.order)
        cls.order.append(obj)
        entry[keys[-1]] = index
        obj.registry_key_ = name

    @classmethod
    def search(cls, name):
        """
        Search an object with the given name. The name doesn't need to be canonical.
        For example, we can search ``GCN`` and get the object of ``models.GCN``.
        """
        keys = []
        pattern = re.compile(rf"\b{name}\b")
        for k, v in cls.table.flatten().items():
            if pattern.search(k):
                keys.append(k)
                value = cls.order[v]
        length = len(keys)
        if length == 0:
            raise KeyError(f"Can't find any registered key containing `{name}`")
        if length > 1:
            keys = ', '.join([f"`{key}`" % key for key in keys])
            raise KeyError(f"Ambiguous key `{name}`. Found {keys}")
        return value


class Configurable(type):
    """
    Class for load/save configuration.
    It will automatically record every argument passed to the ``__init__`` function.
    This class is inspired by :meth:`state_dict()` in PyTorch, but designed for hyperparameters.
    Inherit this class to construct a configurable class.
    >>> class MyClass(nn.Module, core.Configurable):
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
    >>> gcn = Configurable.load_config(config)
    """
    @classmethod
    def load_config(mcs, config):
        """
        Configurable.load_config
        """
        name = config["__class__"]
        if mcs == Configurable:
            cls_name = Registry.search(name)
            load_func = getattr(cls_name, 'load_config', None)
            if load_func:
                return cls_name.load_func(config)
            mcs_ = cls_name
        elif getattr(mcs, "registry_key_", mcs.__name__) != name:
            raise ValueError(f"Expect config __class__ to be `{mcs.__name__}`, but found `{name}`")

        kwargs = {}
        for k, v in config.items():
            if isinstance(v, dict) and "__class__" in v:
                v = Configurable.load_config(v)
            elif isinstance(v, list):
                v = [Configurable.load_config(v_)
                     if isinstance(v_, dict) and "__class__" in v_ else v_
                     for v_ in v]
            if k != "__class__":
                kwargs[k] = v
        return mcs_(**kwargs)

    def config_dict(cls):
        """
        config_dict
        """
        def unroll_config_dict(obj):
            if isinstance(type(obj), Configurable):
                obj = obj.config_dict()
            elif isinstance(obj, (str, bytes)):
                return obj
            elif isinstance(obj, dict):
                return type(obj)({k: unroll_config_dict(v) for k, v in obj.items()})
            elif isinstance(obj, (list, tuple)):
                return type(obj)(unroll_config_dict(x) for x in obj)
            return obj

        cls_ = getattr(cls, "registry_key_", cls.__class__.__name__)
        config = {"__class__": cls_}
        for k, v in cls_.configs.items():
            config[k] = unroll_config_dict(v)
        return config


def make_configurable(cls, module=None, ignore_args=()):
    """
    Make a configurable class out of an existing class.
    The configurable class will automatically record every argument passed to its ``__init__`` function.
    Parameters:
        cls (type): input class
        module (str, optional): bind the output class to this module.
            By default, bind to the original module of the input class.
        ignore_args (set of str, optional): arguments to ignore in the ``__init__`` function
    """
    ignore_args_ = set(ignore_args)
    module = module or cls.__module__
    meta_class = type(cls)
    if issubclass(meta_class, Configurable):  # already a configurable class
        return cls
    if meta_class != type:  # already have a meta class
        meta_class = type(Configurable.__name__, (meta_class, Configurable), {})
    else:
        meta_class = Configurable
    return meta_class(cls.__name__, (cls,), {"_ignore_args": ignore_args_, "__module__": module})
