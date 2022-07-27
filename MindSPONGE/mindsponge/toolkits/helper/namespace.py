"""
This **module** is used to provide help functions and classes about namespace
"""
import sys
from inspect import currentframe
from importlib import import_module
from types import MethodType, FunctionType
from functools import partial


def source(module):
    """
    This **function** import the module and merge all the global variables into the caller module globals()
    :param module:
    :return:
    """
    global_ = currentframe().f_back.f_globals
    module_ = import_module(module, package=global_["__name__"])
    for key, value in module_.__dict__.items():
        if not key.startswith("_"):
            global_[key] = value
    return module_


# for the special alternative name
SPECIAL_STRINGS = {"Pdb": "PDB", "Sponge": "SPONGE", "Nb14": "NB14", "Lj": "LJ",
                   "Residuetype": "ResidueType", "Pubchem": "PubChem", "Resp": "RESP", "Name2atom": "Name2Atom"}


def set_real_global_variable(name, value):
    """
    This **function** is used to set the variable to real global variable
    :param name:
    :param value:
    :return:
    """
    sys.modules.get("__main__").__dict__[name] = value


def remove_real_global_variable(name):
    """
    This **function** is used to remove the variable from real global variable
    :param name:
    :param value:
    :return:
    """
    sys.modules.get("__main__").__dict__.pop(name)


def set_alternative_name(obj, func, set_method):
    """
    This **function** is used to set the alternative name for a function and an object
    :param obj:
    :param func:
    :param set_method:
    :return:
    """
    name = func.__name__
    set_method(obj, name, func)
    new_name = "_".join([i.capitalize() for i in name.split("_")])
    second_new_name = "".join([i.capitalize() for i in name.split("_")])
    third_new_name = second_new_name[0].lower() + second_new_name[1:]

    set_method(obj, new_name, func)
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
    :param cls:
    :return:
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
    :param instance:
    :return:
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


def set_global_alternative_names(dic, real_global=False):
    """
    This **function** is used to set the variables to be global
    :param dic:
    :param real_global:
    :return:
    """
    new_dict = {}
    for value in dic.values():
        if not isinstance(value, FunctionType) or value.__name__.startswith("_"):
            continue

        if real_global:
            def _global_set_method(obj, name, func):
                obj[name] = func
                set_real_global_variable(name, func)

            set_alternative_name(new_dict, value, _global_set_method)
        else:
            set_dict_value_alternative_name(new_dict, value)
    dic.update(new_dict)
