"""
This **module** is used to provide help functions and classes about namespace
"""
import sys
from inspect import currentframe
from importlib import import_module, reload
from types import MethodType, FunctionType
from functools import partial


def source(module, into_global=True, reload_module=False):
    """
    This **function** import the module and merge all the global variables into the caller module globals().
    In fact, this is similar to the python "import", but it is more convenient to relatively import and reload.

    usage example::

        #import numpy as np
        np = source("numpy", False)

        # from .. import forcefield import amber as ff
        # from ..forcefield.amber import *
        ff = source("..forcefield.amber")

    :param module: the module name to import
    :param into_global: whether to merge the global variables
    :param reload_module: whether to reload the module
    :return: the module to import
    """
    global_ = currentframe().f_back.f_globals
    module_ = import_module(module, package=global_["__name__"])
    if reload_module:
        reload(module_)
    if into_global:
        for key, value in module_.__dict__.items():
            if not key.startswith("_"):
                global_[key] = value
    return module_


# for the special alternative name
SPECIAL_STRINGS = {"Pdb": "PDB", "Sponge": "SPONGE", "Nb14": "NB14", "Lj": "LJ", "Gb": "GB", "Mindsponge": "MindSponge",
                   "Residuetype": "ResidueType", "Pubchem": "PubChem", "Resp": "RESP", "Name2atom": "Name2Atom"}


def set_real_global_variable(name, value):
    """
    This **function** is used to set the variable to real global variable

    usage::

        set_real_global_variable(hello, lambda: print(”hello“))
        hello()

    :param name: the name to the use as a global variable
    :param value: the value corresponding to the variable
    :return: None
    """
    if name[0].isdigit():
        name = "_" + name
    sys.modules.get("__main__").__dict__[name] = value


def remove_real_global_variable(name):
    """
    This **function** is used to remove the variable from real global variable

    usage::

        x = 1
        remove_real_global_variable("x")
        x
        # NameError: name 'x' is not defined

    :param name: the global variable to remove
    :return: None
    """
    sys.modules.get("__main__").__dict__.pop(name)


def set_alternative_name(obj, func, set_method):
    """
    This **function** is used to set the alternative names for a function to an object.
    This is the basic function, and some other partial functions with specific ``set_method`` is more useful.

    :param obj: the object
    :param func: the function
    :param set_method: the method to set
    :return: None
    """
    name = func.__name__
    set_method(obj, name, func)
    new_name = "_".join([i.capitalize() for i in name.split("_")])
    second_new_name = "".join([i.capitalize() for i in name.split("_")])
    third_new_name = second_new_name[0].lower() + second_new_name[1:]

    set_method(obj, new_name, func)
    set_method(obj, second_new_name, func)
    set_method(obj, third_new_name, func)
    new_new_name = new_name
    for t, newt in SPECIAL_STRINGS.items():
        new_new_name = new_new_name.replace(t, newt)
        second_new_name = second_new_name.replace(t, newt)
        third_new_name = third_new_name.replace(t, newt)
    if new_new_name != new_name:
        set_method(obj, new_new_name, func)
        set_method(obj, second_new_name, func)
        set_method(obj, third_new_name, func)


set_attribute_alternative_name = partial(set_alternative_name, set_method=setattr)


def set_classmethod_alternative_names(cls):
    """
    This **function** is used to set the attribute/method alternative names for a class
    usage example::

        class A:
            #@staticmethod
            @classmethod
            def hello_world(cls):
                print("hello")
        set_classmethod_alternative_names(A)
        A.hello_world()
        A.helloWorld()
        A.Hello_World()
        A.HelloWorld()

    :param cls: the class to set alternative names
    :return: None
    """
    types = list(cls.__bases__)
    types.append(cls)
    for sometype in types:
        old_dict = {}
        old_dict.update(sometype.__dict__)
        for i in old_dict:
            self_func = getattr(cls, i, None)
            if self_func and isinstance(self_func, (MethodType, FunctionType)) and not self_func.__name__.startswith(
                    "_"):
                set_attribute_alternative_name(cls, self_func)


def set_attribute_alternative_names(instance):
    """
    This **function** is used to set the attribute/method alternative names for an instance
    usage example::

        class A:
            def __init__(self):
                set_attribute_alternative_names(self)
            def hello_world():
                print("hello")
        a = A()
        a.hello_world()
        a.helloWorld()
        a.Hello_World()
        a.HelloWorld()

    :param instance: the instance to set names
    :return: None
    """
    types = list(type(instance).__bases__)
    types.append(type(instance))
    for sometype in types:
        for i in sometype.__dict__:
            self_func = getattr(instance, i, None)
            if self_func and isinstance(self_func, (MethodType, FunctionType)) and not self_func.__name__.startswith(
                    "_"):
                set_attribute_alternative_name(instance, self_func)


def _dict_set_method(obj, name, func):
    """

    :param obj:
    :param name:
    :param func:
    :return:
    """
    obj[name] = func


set_dict_value_alternative_name = partial(set_alternative_name, set_method=_dict_set_method)


def set_global_alternative_names(real_global=False):
    """
    This **function** is used to set the alternative names of the functions in the module to be global
    usage example::

        def hello_world():
            print("hello")
        set_global_alternative_names()
        hello_world()
        helloWorld()
        Hello_World()
        HelloWorld()

    :param real_global: make the variable to real global, which can be used without the module name
    :return: None
    """
    dic = currentframe().f_back.f_globals
    new_dict = {}
    for key, value in dic.items():
        if not isinstance(value, (FunctionType, type)) or value.__name__.startswith("_") or not key.islower():
            continue

        if real_global:
            if isinstance(value, FunctionType):
                def _global_set_method(obj, name, func):
                    obj[name] = func
                    set_real_global_variable(name, func)

                set_alternative_name(new_dict, value, _global_set_method)
            else:
                set_real_global_variable(value.__name__, value)
        else:
            set_dict_value_alternative_name(new_dict, value)
    dic.update(new_dict)

set_global_alternative_names()
