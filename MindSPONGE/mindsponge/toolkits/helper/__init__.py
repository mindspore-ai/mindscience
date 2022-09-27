"""
This **module** is used to provide help functions and classes

.. autoclass:: _GlobalSetting
    :members:

"""
import os
import time
import stat
import copy
from types import MethodType, FunctionType
from functools import partial, partialmethod, wraps
from collections import OrderedDict
from collections.abc import Iterable
from itertools import product
from abc import ABC

import numpy as np

from .namespace import set_real_global_variable, remove_real_global_variable, set_alternative_name, \
    set_attribute_alternative_name, set_classmethod_alternative_names, set_attribute_alternative_names, \
    set_dict_value_alternative_name, set_global_alternative_names, source

from .math import get_rotate_matrix, get_fibonacci_grid, guess_element_from_mass, kabsch


class Xdict(dict):
    """
    This **class** is used to be a dict which can give not_found_message

    :param not_found_method: the method (function) which will accept the key and return the value \
when the key is not found. **New From 1.2.6.7**
    :param not_found_message: the string to print when the key is not found
    """
    def __init__(self, *args, **kwargs):
        if "not_found_method" in kwargs:
            self.not_found_method = kwargs.pop("not_found_method")
        else:
            self.not_found_method = None
        if "not_found_message" in kwargs:
            self.not_found_message = kwargs.pop("not_found_message")
        else:
            self.not_found_message = None
        super().__init__(*args, **kwargs)
        self.id = hash(id(self))

    def __getitem__(self, key):
        toget = self.get(key, self.id)
        if toget != self.id:
            return toget
        if self.not_found_method:
            value = self.not_found_method(key)
            self[key] = value
            return self[key]
        if self.not_found_message:
            raise KeyError(self.not_found_message.format(key))
        raise KeyError(f"{key}")


def xopen(filename, flag, mode=None):
    """
    This **function** is used to open a file with the proper permissions mode

    :param filename: the file to open
    :param flag: the open flag, one of "w", "wb", "r", "rb"
    :param mode: the permission mode. 0o0777 for default
    :return: the opened file
    """
    if mode is None:
        mode = stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU
    if flag in ("w", "wb"):
        real_flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
    elif flag in ("r", "rb"):
        real_flags = os.O_RDONLY
    else:
        raise NotImplementedError
    fd = os.open(filename, real_flags, mode)
    fo = os.fdopen(fd, flag)
    return fo


def xprint(*args, **kwargs):
    """
    This **function** is used to print information according to the verbose level in GlobalSetting

    :param args, kwargs except verbose: the arguments to print
    :param verbose: only print when the verbose level is not less than this value
    :return: None
    """
    if "verbose" in kwargs:
        verbose = kwargs.pop("verbose")
    else:
        verbose = 0
    if GlobalSetting.verbose >= verbose:
        print(*args, **kwargs)


class _GlobalSetting():
    """
    This **class** is used to set the global settings.

    There is an initialized instance with the name GlobalSetting after import the module.
    Do not initialize an instance yourself.
    """
    def __init__(self):
        set_attribute_alternative_names(self)
        # 打印信息的详细程度
        self.verbose = 0
        """the global verbose level"""
        # 是否将分子移到中心
        self.nocenter = False
        """move the molecule to the center of the box when building"""
        # 分子与盒子之间的距离
        self.boxspace = 3
        """the distance between the molecule and the box border to set the box length when no box is given"""
        # 最远的成键距离，用于拓扑分析时最远分析多远
        self.farthest_bonded_force = 0
        setattr(self, "HISMap", {"DeltaH": "", "EpsilonH": "", "HIS": Xdict()})
        # 所有的成键类型力的Type
        setattr(self, "BondedForces", [])
        setattr(self, "BondedForcesMap", Xdict(not_found_message="bonded force {} not found"))
        # 所有虚拟原子的Type和对应的依赖的其他原子的数量
        setattr(self, "VirtualAtomTypes", Xdict(not_found_message="virtual atom type {} not found"))
        # 单位换算
        setattr(self, "UnitMapping", {"distance": {"nm": 1e-9, "A": 1e-10},
                                      "energy": {"kcal/mol": 4.184, "eV": 96.4853, "kJ/mol": 1},
                                      "charge": {"e": 1, "SPONGE": 1.0 / 18.2223},
                                      "angle": {"degree": np.pi, "rad": 180}
                                      })
        setattr(self, "PDBResidueNameMap", {"head": Xdict(), "tail": Xdict(), "save": Xdict()})

    @staticmethod
    def set_unit_transfer_function(sometype):
        """
        This **function** is used to return a function to add a static method  ``_unit_transfer`` for a class.
        It is recommended used as a **decorator**. The origin ``_unit_transfer``  method will not be kept.

        ``_unit_transfer`` should receive the ForceType as an input, and modify the parameters in it. See the example
        in ``forcefield.base.lj_base``.

        :param sometype: the class
        :return: the **decorator**
        """

        def wrapper(func):
            setattr(sometype, "_unit_transfer", func)
            return func

        return wrapper

    @staticmethod
    def add_unit_transfer_function(sometype):
        """
        This **function** is used to return a function to add a static method  ``_unit_transfer`` for a class.
        It is recommended used as a **decorator**. The origin ``_unit_transfer``  method will be kept.

        ``_unit_transfer`` should receive the ForceType as an input, and modify the parameters in it. See the example
        in ``forcefield.base.lj_base``.

        :param sometype: the class
        :return: the **decorator**
        """
        func0 = getattr(sometype, "_unit_transfer")

        def wrapper(func):
            @wraps(func0)
            def temp(self):
                func0(self)
                func(self)

            setattr(sometype, "_unit_transfer", temp)
            return func

        return wrapper

    def add_pdb_residue_name_mapping(self, place, pdb_name, real_name):
        """
        This **function** is used to add the residue name mapping to the property ``PDBResidueNameMap``

        :param place: head or tail
        :param pdb_name: the residue name in pdb
        :param real_name: the residue name in Python
        :return: None
        """
        assert place in ("head", "tail")
        self.PDBResidueNameMap[place][pdb_name] = real_name
        self.PDBResidueNameMap["save"][real_name] = pdb_name

    def set_invisible_bonded_forces(self, types):
        """
        This **function** is used to disables the types of bonded forces when building.

        :param types: the types to set
        :return: None
        """
        for typename in types:
            self.BondedForces.remove(self.BondedForcesMap[typename])

    def set_visible_bonded_forces(self, types):
        """
        This **function** is used to disables the types of bonded forces except named here when building.

        :param types: the types to set
        :return: None
        """
        self.BondedForces.clear()
        for typename in types:
            self.BondedForces.append(self.BondedForcesMap[typename])


globals()["GlobalSetting"] = _GlobalSetting()


class Type(ABC):
    """
    This **class** is the abstract class of the types (atom types, bonded force types and so on).

    :param name: the name of the type
    :param kwargs: parameters of the type
    """
    _name = None
    _parameters = {"name": str}
    _types = Xdict(not_found_message="Type {} not found. Did you import the proper force field?")
    _types_different_name = Xdict(not_found_message="Type {} not found. Did you import the proper force field?")

    def __init__(self, **kwargs):

        prop_fmt = Xdict(type(self)._parameters)

        self.contents = Xdict().fromkeys(prop_fmt.keys())
        self.name = kwargs.pop("name")
        assert self.name not in type(self)._types.keys(), "The name '%s' has already existed in '%sType'" % (
            self.name, self.get_class_name())
        type(self)._types[self.name] = self
        type(self)._types_different_name[self.name] = self
        for key, value in kwargs.items():
            assert key in self.contents.keys(), "The parameter '%s' is not one of the parameters of '%sType'" % (
                key, self.get_class_name())
            self.contents[key] = prop_fmt[key](value)
        type(self)._unit_transfer(self)

    def __repr__(self):
        return "Type of " + self.get_class_name() + ": " + self.name

    def __hash__(self):
        return hash(repr(self))

    def __getattribute__(self, attr):
        if attr != "contents" and attr in self.contents.keys():
            return self.contents[attr]
        return super().__getattribute__(attr)

    def __setattr__(self, attr, value):
        if attr != "contents" and attr in self.contents.keys():
            self.contents[attr] = value
        else:
            super().__setattr__(attr, value)

    @classmethod
    def add_property(cls, parm_fmt, parm_default=None):
        """
        This **function** is used to add a property to the class

        :param parm_fmt: a dict mapping the name and the format of the property
        :param parm_default: a dict mapping the name and the default value of the property
        :return: None
        """
        cls._parameters.update(parm_fmt)
        if parm_default is None:
            parm_default = Xdict()
        for type_ in cls._types.values():
            type_.contents.update({key: parm_default.get(key, None) for key in parm_fmt.keys()})

    @classmethod
    def set_property_unit(cls, prop, unit_type, base_unit):
        """
        This **function** is used to set the unit of the property of the class

        In ``forcefield.base.lj_base``, here is an usage example::

                    LJType = Generate_New_Pairwise_Force_Type("LJ",
                                          {"epsilon": float, "rmin": float, "sigma": float, "A": float, "B": float})
                    LJType.Set_Property_Unit("rmin", "distance", "A")
                    LJType.Set_Property_Unit("sigma", "distance", "A")
                    LJType.Set_Property_Unit("epsilon", "energy", "kcal/mol")
                    LJType.Set_Property_Unit("A", "energy·distance^6", "kcal/mol·A^6")
                    LJType.Set_Property_Unit("B", "energy·distance^12", "kcal/mol·A^12")

        :param prop: the name of the property
        :param unit_type: the type of the unit
        :param base_unit: the basic unit used in Python
        :return: None
        """
        assert prop in cls._parameters.keys(), "Unknown property '%s' for type '%s'" % (prop, cls.name)
        temp_unit_lists = unit_type.split("·")
        temp_unit_power = []

        for i, temp_unit in enumerate(temp_unit_lists):
            unit = temp_unit.split("^")
            if len(unit) == 2:
                temp_unit_lists[i] = unit[0]
                temp_unit_power.append(int(unit[1]))
            else:
                temp_unit_power.append(1)
            assert unit[0] in GlobalSetting.UnitMapping.keys(), "Unknown unit type '%s'" % unit_type

        temp_unit_lists_units = [list(GlobalSetting.UnitMapping[unit].keys()) for unit in temp_unit_lists]
        alls = Xdict()
        for unit_combination in product(*temp_unit_lists_units):
            unit = []
            value = 1
            for i, current_unit in enumerate(unit_combination):
                power = temp_unit_power[i]
                value *= GlobalSetting.UnitMapping[temp_unit_lists[i]][current_unit] ** power
                if power == 1:
                    unit.append(current_unit)
                else:
                    unit.append(current_unit + "^" + str(power))
            alls["·".join(unit)] = value
        base_unit_rate = alls[base_unit]

        def temp_func(current_rate, base_unit_rate):
            return lambda x: float(x) * current_rate / base_unit_rate

        prop_alls = Xdict()
        for unit, current_rate in alls.items():
            temp_prop = prop + '[' + unit + ']'
            prop_alls[temp_prop] = temp_func(current_rate, base_unit_rate)
        cls.Add_Property(prop_alls)

    @classmethod
    def new_from_string(cls, string, skip_lines=0):
        """
        This **function** is used to update the types of the class

        Usage example::

            AtomType.New_From_String('''
            name property1[unit1] property2[unit2]
            XXX  XXX1  XXX2
            YYY  YYY1  YYY2
            ''')

        :param string: the string to update
        :param skip_lines: skip the first ``skip_lines`` line(s)
        :return: None
        """
        count = -1
        kwargs = OrderedDict()
        for line in string.split("\n"):
            if not line.strip() or line.strip()[0] in ("!", "#", ";", "/"):
                continue

            count += 1
            if count < skip_lines:
                continue
            if count == skip_lines:
                kwargs = kwargs.fromkeys(line.split())
            else:
                words = line.split()
                i = 0
                tempkw = Xdict().fromkeys(kwargs.keys())
                for key in tempkw.keys():
                    tempkw[key] = words[i]
                    i += 1
                type_already_have = False

                if tempkw["name"] in cls._types_different_name.keys():
                    tempkw["name"] = cls._types_different_name[tempkw["name"]].name
                    type_already_have = True
                if not type_already_have:
                    if "reset" in tempkw.keys():
                        tempkw.pop("reset")
                    cls(**tempkw)
                else:
                    temp = cls._types[tempkw.pop("name")]
                    temp.Update(**tempkw)

    @classmethod
    def new_from_file(cls, filename, skip_lines=0):
        """
        This **function** is used to update the types of the class

        :param filename:  the name of the file
        :param skip_lines: skip the first ``skip_lines`` line(s)
        :return: None
        """
        with open(filename, encoding='utf-8') as f:
            cls.New_From_String(f.read(), skip_lines)

    @classmethod
    def new_from_dict(cls, dic):
        """
        This **function** is used to update the types of the class

        Usage example::

            AtomType.New_From_Dict({"XXX": {"property1[unit1]": XXX1, "property2[unit2]": XXX2}})

        :param dic: the dict of the parameters
        :return: None
        """
        for name, values in dic.items():
            type_already_have = False
            if name in cls._types_different_name.keys():
                name = cls._types_different_name[name].name
                type_already_have = True
            if not type_already_have:
                if "reset" in values.keys():
                    values.pop("reset")
                values["name"] = name
                cls(**values)
            else:
                temp = cls._types[name]
                temp.Update(**values)

    @classmethod
    def get_class_name(cls):
        """
        This **function** gives the ._name of the class

        :return: the name of the class
        """
        return cls._name

    @classmethod
    def clear_type(cls, name=None):
        """
        This **function** clears the instance(s) of the class

        :param name: the instance name to clear. If None, all instances will be cleared
        :return: None
        """
        if name is None:
            cls._types.clear()
            cls._types_different_name.clear()
        else:
            cls._types.pop(name)

    @classmethod
    def set_type(cls, name, toset):
        """
        This **function** sets the instance into the class

        :param name: the instance name
        :param toset: the instance to set
        :return: None
        """
        cls._types[name] = toset

    @classmethod
    def get_type(cls, name):
        """
        This **function** gets the instance of the class

        :param name: the instance name
        :return: the instance to set
        """
        return cls._types[name]

    @classmethod
    def get_all_types(cls):
        """
        This **function** gets the all instances of the class

        :return: a dict mapping the name and the instance
        """
        return cls._types

    @staticmethod
    def _unit_transfer(instance):
        """

        :param instance:
        :return:
        """
        for prop in instance.contents.keys():
            if "[" in prop and "]" in prop and instance.contents[prop] is not None:
                real_prop = prop.split('[')[0]
                instance.contents[real_prop] = instance.contents[prop]
                instance.contents[prop] = None

    def update(self, **kwargs):
        """
        This **function** is used to update the properties of the instance

        :param kwargs: parameters to update
        :return: None
        """
        for key, value in kwargs.items():
            type_func = type(self)._parameters.get(key, None)
            if not type_func:
                raise KeyError(f"{key} is not a property of {type(self)._name}")
            self.contents[key] = type_func(value)
        type(self)._unit_transfer(self)


class AbstractMolecule(ABC):
    """This abstract **class** is used to judge whether a class can be treated as a molecule """


class AtomType(Type):
    """
    This **class** is a subclass of Type, for atom types

    An instance with name "UNKNOWN" is initialized after importing the module

    :param name: the name of the type
    :param kwargs: parameters of the type
    """
    _name = "Atom"
    _parameters = {"name": str, "x": float, "y": float, "z": float}
    _types = Xdict(not_found_message="Atom Type {} not found. Did you import the proper force field?")
    _types_different_name = Xdict(not_found_message="Atom Type {} not found. Did you import the proper force field?")


set_classmethod_alternative_names(AtomType)

AtomType.New_From_String("name\nUNKNOWN")


class ResidueType(Type):
    """
    This **class** is a subclass of Type, for residue types

    :param name: the name of the type
    :param kwargs: parameters of the type
    """
    _name = "Residue"
    _parameters = {"name": str}
    _types = Xdict(not_found_message="Residue Type {} not found. Did you import the proper force field?")
    _types_different_name = Xdict(not_found_message="Residue Type {} not found. Did you import the proper force field?")

    def __init__(self, **kwargs):
        # 力场构建相关
        self.contents = Xdict()
        self.connectivity = Xdict()
        """a dict mapping the atom to its directly connected atom dict"""
        self.built = False
        """indicates the instance built or not"""
        self.bonded_forces = {frc.get_class_name(): [] for frc in GlobalSetting.BondedForces}
        """the bonded forces after building"""

        # 索引相关
        self._name2atom = Xdict()
        self.atoms = []
        """the atoms in the instance"""
        self._atom2name = Xdict()
        self._atom2index = Xdict()
        self._name2index = Xdict()

        super().__init__(**kwargs)

        self._name2atom.not_found_message = "There is no atom named {} in %s"%self.name
        # 连接功能
        self.link = {"head": None, "tail": None, "head_next": None, "tail_next": None,
                     "head_length": 1.5, "tail_length": 1.5, "head_link_conditions": [], "tail_link_conditions": []}
        self.connect_atoms = Xdict()
        """a dict mapping the atom to \
 another dict which mapping the connecting type to the name of the connected atom"""

        set_attribute_alternative_names(self)

    def __getattribute__(self, attr):
        if attr not in ("_name2atom", "contents") and attr in self._name2atom.keys():
            return self._name2atom[attr]
        if getattr(AtomType, "_parameters").get(attr, None) == float:
            return np.sum([getattr(atom, attr) for atom in self.atoms])
        return super().__getattribute__(attr)

    @property
    def head(self):
        """
        the name of the first atom in the head
        """
        return self.link.get("head")

    @head.setter
    def head(self, atom):
        self.link["head"] = atom

    @property
    def tail(self):
        """
        the name of the first atom in the tail
        """
        return self.link.get("tail")

    @tail.setter
    def tail(self, atom):
        self.link["tail"] = atom

    @property
    def head_next(self):
        """
        the name of the second atom in the head
        """
        return self.link.get("head_next")

    @head_next.setter
    def head_next(self, atom):
        self.link["head_next"] = atom

    @property
    def tail_next(self):
        """
        the name of the second atom in the tail
        """
        return self.link.get("tail_next")

    @tail_next.setter
    def tail_next(self, atom):
        self.link["tail_next"] = atom

    @property
    def head_length(self):
        """
        the length of the bond connected to the last residue
        """
        return self.link.get("head_length")

    @head_length.setter
    def head_length(self, atom):
        self.link["head_length"] = atom

    @property
    def tail_length(self):
        """
        the length of the bond connected to the next residue
        """
        return self.link.get("tail_length")

    @tail_length.setter
    def tail_length(self, atom):
        self.link["tail_length"] = atom

    @property
    def head_link_conditions(self):
        """
        the link conditions to the last residue
        """
        return self.link.get("head_link_conditions")

    @property
    def tail_link_conditions(self):
        """
        the link conditions to the next residue
        """
        return self.link.get("tail_link_conditions")

    def name2atom(self, name):
        """
        This **function** convert an atom name to an Atom instance

        :param name: the name
        :return: the Atom instance
        """
        return self._name2atom[name]

    def atom2index(self, atom):
        """
        This **function** convert an Atom instance to its index

        :param atom: the Atom instance
        :return: the index
        """
        return self._atom2index[atom]

    def name2index(self, name):
        """
        This **function** convert an atom name to its index

        :param name: the name
        :return: the index
        """
        return self._name2index[name]

    def add_atom(self, name, atom_type, x, y, z):
        """
        This **function** is used to add an atom to the residue type.

        :param name: the name of the atom
        :param atom_type: the type of the atom
        :param x: the coordinate x
        :param y: the coordinate y
        :param z: the coordinate z
        :return: None
        """
        new_atom = Atom(atom_type, name)
        self.atoms.append(new_atom)
        new_atom.residue = self
        new_atom.x = float(x)
        new_atom.y = float(y)
        new_atom.z = float(z)
        self._name2atom[name] = new_atom
        self._atom2name[new_atom] = name
        self._atom2index[new_atom] = len(self.atoms) - 1
        self._name2index[name] = len(self.atoms) - 1
        self.connectivity[new_atom] = set([])

    def omit_atoms(self, atoms, charge):
        """
        This **function** omits some atoms from the ResidueType

        :param atoms: the atom(s) to omit
        :param charge: the total charge of the residue type after the omission. \
None to use the charge sum of the unomitted atoms
        :return: None
        """
        if not isinstance(atoms, Iterable):
            atoms = [atoms]
        atoms = set(getattr(self, atom) if isinstance(atom, str) else atom for atom in atoms)

        charges = np.array([atom.charge for atom in self.atoms])
        positive_charge = np.sum(charges[charges > 0])
        negative_charge = np.sum(charges[charges < 0])
        if charge is None:
            charge = positive_charge + negative_charge - np.sum([atom.charge for atom in atoms])
        factor = (charge - positive_charge - negative_charge) / (positive_charge - negative_charge)
        for atom in atoms:
            self.atoms.remove(atom)
            atom.residue = None
            self.connectivity.pop(atom)
        self._name2atom.clear()
        self._atom2name.clear()
        self._atom2index.clear()
        self._name2index.clear()
        for index, atom in enumerate(self.atoms):
            self._name2atom[atom.name] = atom
            self._atom2name[atom] = atom.name
            self._atom2index[atom] = index
            self._name2index[atom.name] = index
            self.connectivity[atom] -= atoms
            self.charge += np.sign(self.charge) * factor

    def add_connectivity(self, atom0, atom1):
        """
        This **function** is used to add the connectivity between two atoms to the residue type.

        :param atom0: the atom name or the Atom instance
        :param atom1: the atom name or the Atom instance
        :return: None
        """
        if isinstance(atom0, str):
            atom0 = self.name2atom(atom0)
        if isinstance(atom1, str):
            atom1 = self.name2atom(atom1)
        self.connectivity[atom0].add(atom1)
        self.connectivity[atom1].add(atom0)

    def add_bonded_force(self, bonded_force_entity, typename=None):
        """
        This **function** is used to add the bonded force to the residue type.

        :param bonded_force_entity: the bonded force instance
        :param typename: the bonded force type name. If None, get_class_name will be used to get the name
        :return: None
        """
        if typename is None:
            typename = bonded_force_entity.get_class_name()
        if typename not in self.bonded_forces.keys():
            self.bonded_forces[typename] = []
        self.bonded_forces[typename].append(bonded_force_entity)

    def deepcopy(self, name, forcopy=None):
        """
        This **function** is used to deep copy the instance

        :param name: the new ResidueType name
        :param forcopy: the key to help you find who it is from
        :return: the new instance
        """
        new_restype = ResidueType(name=name)
        new_restype.link = copy.deepcopy(self.link)
        donot_delete = True
        if forcopy is None:
            donot_delete = False
            forcopy = hash(str(time.time()))

        for atom in self.atoms:
            new_restype.Add_Atom(atom.name, atom.type, atom.x, atom.y, atom.z)
            atom.copied[forcopy] = new_restype.atoms[-1]
            atom.copied[forcopy].contents = Xdict()
            atom.copied[forcopy].contents.update(atom.contents)

        for atom, connect_set in self.connectivity.items():
            for aton in connect_set:
                new_restype.Add_Connectivity(atom.copied[forcopy], aton.copied[forcopy])

        for atom in self.atoms:
            atom.copied[forcopy].Extra_Exclude_Atoms(map(lambda aton: aton.copied[forcopy], atom.extra_excluded_atoms))

        if self.built:
            for bond_entities in self.bonded_forces.values():
                for bond_entity in bond_entities:
                    new_restype.Add_Bonded_Force(bond_entity.deepcopy(forcopy))
            new_restype.built = True
            for atom in self.atoms:
                atom.copied[forcopy].linked_atoms = {key: set(map(lambda _atom: _atom.copied[forcopy], value)) for
                                                     key, value in atom.linked_atoms.items()}

        if not donot_delete:
            for atom in self.atoms:
                atom.copied.pop(forcopy)

        return new_restype


class Entity(ABC):
    """
    This **class** is the abstract class of the entities (atoms, bonded forces, residues and so on).

    :param entity_type: the type of the entity
    :param name: the name of the entity
    """
    _count = 0
    _name = None

    def __init__(self, entity_type, name=None):
        self.contents = {**entity_type.contents}
        self._count = type(self)._count
        if not name:
            name = entity_type.name
        type(self)._count += 1
        self.name = name
        """the name of the instance"""
        self.type = entity_type
        """the type of the entity"""

    def __repr__(self):
        return "Entity of " + type(self)._name + ": " + self.name + "(" + str(self._count) + ")"

    def __hash__(self):
        return hash(repr(self))

    def __getattribute__(self, attr):
        if attr != "contents" and attr in self.contents.keys():
            return self.contents[attr]
        return super().__getattribute__(attr)

    def __setattr__(self, attr, value):
        if attr != "contents" and attr in self.contents.keys():
            self.contents[attr] = value
        else:
            super().__setattr__(attr, value)

    @classmethod
    def get_class_name(cls):
        """
        This **function** gives the name of the class

        :return: the name of the class
        """
        return cls._name

    def update(self, **kwargs):
        """
        This **function** is used to update the properties of the instance

        :param kwargs: the properties to update
        :return: None
        """
        for key, value in kwargs.items():
            assert key in self.contents.keys()
            self.contents[key] = getattr(type(self.type), "_parameters")[key](value)
        unit_transfer = getattr(type(self.type), "_unit_transfer")
        unit_transfer(self)


class Atom(Entity):
    """
    This **class** is a subclass of Entity, for atoms

    :param entity_type: a string or a AtomType instance, the type of the entity
    :param name: the name of the entity
    """
    _name = "Atom"
    _count = 0

    def __init__(self, entity_type, name=None):
        # 力场基本信息
        if isinstance(entity_type, str):
            entity_type = AtomType.get_type(entity_type)
        super().__init__(entity_type, name)
        self.x = None
        """the coordinate x of the Atom instance"""
        self.y = None
        """the coordinate y of the Atom instance"""
        self.z = None
        """the coordinate z of the Atom instance"""
        self.bad_coordinate = False
        # 父信息
        self.residue = None
        """the residue which the atom belongs to, maybe a Residue instance or a ResidueType instance"""

        # 成键信息
        self.linked_atoms = {i + 1: set() for i in range(1, GlobalSetting.farthest_bonded_force)}
        """a dict mapping the type and the atoms linked"""
        self.linked_atoms["extra_excluded_atoms"] = set()

        # 复制信息
        self.copied = Xdict()
        """a dict to find who is a copy of the atom"""
        set_attribute_alternative_names(self)

    @property
    def extra_excluded_atoms(self):
        """
        the extra excluded atoms of this atom

        :return: a set of atoms
        """
        return self.linked_atoms["extra_excluded_atoms"]

    def deepcopy(self, forcopy=None):
        """
        This **function** is used to deep copy the instance

        :param forcopy: the key to help you find who it is from
        :return: the new instance
        """
        new_atom = Atom(self.type, self.name)
        new_atom.contents = {**self.contents}
        if forcopy:
            self.copied[forcopy] = new_atom
        return new_atom

    def link_atom(self, link_type, atom):
        """
        This **function** is used to link atoms for building

        :param link_type: the type to link
        :param atom: the atom to link
        :return: None
        """
        if link_type not in self.linked_atoms.keys():
            self.linked_atoms[link_type] = set()
        self.linked_atoms[link_type].add(atom)

    def extra_exclude_atom(self, atom):
        """
        This **function** is used to extra exclude one atom

        :param atom: an Atom instance
        :return: None
        """
        self.extra_excluded_atoms.add(atom)
        atom.extra_excluded_atoms.add(self)

    def extra_exclude_atoms(self, atom_list):
        """
        This **function** is used to extra exclude a list of atoms

        :param atom_list: the atom list
        :return: None
        """
        for atom in atom_list:
            self.Extra_Exclude_Atom(atom)


set_classmethod_alternative_names(Atom)


class Residue(Entity):
    """
    This **class** is a subclass of Entity, for residues

    :param entity_type: a string or a ResidueType instance, the type of the entity
    :param name: the name of the entity
    :param directly_copy: if True, directly copy the Residue instance from the ResidueType instance
    """
    _name = "Residue"
    _count = 0

    def __init__(self, entity_type, name=None, directly_copy=False):
        if isinstance(entity_type, str):
            entity_type = ResidueType.get_type(entity_type)
        super().__init__(entity_type, name)
        self.atoms = []
        """the atoms in the instance"""
        self._name2atom = Xdict()
        self._name2atom.not_found_message = "There is no atom named {} in %s" % self.name
        self._atom2name = Xdict()
        self._atom2index = Xdict()
        self._name2index = Xdict()
        self.connectivity = Xdict()
        """a dict mapping the atom to its directly connected atom dict"""
        self.bonded_forces = {frc.get_class_name(): [] for frc in GlobalSetting.BondedForces}
        """the bonded forces after building"""
        self.built = False
        """indicates the instance built or not"""
        if directly_copy:
            forcopy = hash(int(time.time()))
            for atom in self.type.atoms:
                self.Add_Atom(atom.name, atom.type, atom.x, atom.y, atom.z)
                atom.copied[forcopy] = self.atoms[-1]
            for atom in self.type.atoms:
                for aton in atom.extra_excluded_atoms:
                    atom.copied[forcopy].Extra_Exclude_Atom(aton.copied[forcopy])

        set_attribute_alternative_names(self)

    def __getattribute__(self, attr):
        if attr not in ("_name2atom", "contents") and attr in self._name2atom.keys():
            return self._name2atom[attr]
        if getattr(AtomType, "_parameters").get(attr, None) == float:
            return np.sum([getattr(atom, attr) for atom in self.atoms])
        return super().__getattribute__(attr)

    def set_type(self, new_type, add_missing_atoms=False):
        """
        This **function** is used to change the type of the residue to a new type
        **New From 1.2.6.8**

        :param new_type: the instance or the name of the new residue type
        :param add_missing_atoms: whether to add missing atoms after deleting the terminal atoms
        :return: 1 for success, 0 for failure
        """
        if isinstance(new_type, str):
            new_type = ResidueType.get_type(new_type)
        new_type_atom = Xdict({atom.name: atom for atom in new_type.atoms})
        to_remove = set()
        for atom in self.atoms:
            if atom.name not in new_type_atom:
                to_remove.add(atom)
            else:
                atom.type = new_type_atom[atom.name].type
                atom.charge = new_type_atom[atom.name].charge
                self._name2index[atom.name] -= len(to_remove)
                self._atom2index[atom] -= len(to_remove)
        for atom in to_remove:
            self._name2index.pop(atom.name)
            self._atom2index.pop(atom)
            self._atom2name.pop(atom)
            self._name2atom.pop(atom.name)
            self.atoms.remove(atom)
            atom.residue = None
        self.type = new_type
        self.name = new_type.name
        if add_missing_atoms:
            self.add_missing_atoms()
        return 1

    def unterminal(self, add_missing_atoms=False):
        """
        This **function** is used to turn the terminal residue to be unterminal
        **New From 1.2.6.7**

        :param add_missing_atoms: whether to add missing atoms after deleting the terminal atoms
        :return: 1 for success, 0 for failure
        """
        if self.type.name in GlobalSetting.PDBResidueNameMap["save"]:
            new_type = GlobalSetting.PDBResidueNameMap["save"][self.type.name]
        else:
            return 0
        return self.set_type(new_type, add_missing_atoms)

    def name2atom(self, name):
        """
        This **function** convert an atom name to an Atom instance

        :param name: the name
        :return: the Atom instance
        """
        return self._name2atom[name]

    def atom2index(self, atom):
        """
        This **function** convert an Atom instance to its index

        :param atom: the Atom instance
        :return: the index
        """
        return self._atom2index[atom]

    def name2index(self, name):
        """
        This **function** convert an atom name to its index

        :param name: the name
        :return: the index
        """
        return self._name2index[name]

    def add_atom(self, name, atom_type=None, x=None, y=None, z=None):
        """
        This **function** is used to add an atom to the residue

        :param name: the name of the atom. If an Atom instance is given, only the name and type are used.
        :param atom_type: the type of the atom. If None, it will copy from the ResidueType.
        :param x: the coordinate x. If None, it will copy from the ResidueType.
        :param y: the coordinate y. If None, it will copy from the ResidueType.
        :param z: the coordinate z. If None, it will copy from the ResidueType.
        :return: None
        """
        if isinstance(name, Atom):
            assert atom_type is None
            new_atom = Atom(name.type, name.name)
            new_atom.contents = {**name.contents}
            name = name.name
        else:
            if not atom_type:
                atom_type = self.type.name2atom(name).type
                new_atom = Atom(atom_type, name)
                new_atom.contents = {**self.type.name2atom(name).contents}
            else:
                new_atom = Atom(atom_type, name)
                new_atom.contents = {**self.type.name2atom(name).contents}

        new_atom.residue = self
        self.atoms.append(new_atom)
        if x:
            new_atom.x = float(x)
        if y:
            new_atom.y = float(y)
        if z:
            new_atom.z = float(z)
        self._name2atom[name] = new_atom
        self._atom2name[new_atom] = name
        self._atom2index[new_atom] = len(self.atoms) - 1
        self._name2index[name] = len(self.atoms) - 1
        self.connectivity[new_atom] = set([])

    def add_connectivity(self, atom0, atom1):
        """
        This **function** is used to add the connectivity between two atoms to the residue entity.

        :param atom0: the atom name or the Atom instance
        :param atom1: the atom name or the Atom instance
        :return: None
        """
        if isinstance(atom0, str):
            atom0 = self._name2atom[atom0]
        if isinstance(atom1, str):
            atom1 = self._name2atom[atom1]
        if atom0 in self.connectivity.keys():
            self.connectivity[atom0].add(atom1)
        if atom1 in self.connectivity.keys():
            self.connectivity[atom1].add(atom0)

    def add_bonded_force(self, bonded_force_entity):
        """
        This **function** is used to add the bonded force to the residue entity.

        :param bonded_force_entity: the bonded force instance
        :return: None
        """
        if bonded_force_entity.get_class_name() not in self.bonded_forces.keys():
            self.bonded_forces[bonded_force_entity.get_class_name()] = []
        self.bonded_forces[bonded_force_entity.get_class_name()].append(bonded_force_entity)

    def add_missing_atoms(self):
        """
        This **function** is used to add the missing atoms from the ResidueType to the residue entity.

        :return: None
        """
        t = {atom.name for atom in self.atoms}
        uncertified = {atom.name for atom in self.type.atoms}
        certified_positions = []
        template_positions = []
        for atom in self.type.atoms:
            if atom.name in t:
                uncertified.remove(atom.name)
                template_positions.append([atom.x, atom.y, atom.z])
                self_atom = self.name2atom(atom.name)
                certified_positions.append([self_atom.x, self_atom.y, self_atom.z])
        rotation, center1, center2 = kabsch(template_positions, certified_positions)

        while uncertified:
            movedlist = []
            for atom_name in uncertified:
                temp_atom = getattr(self.type, atom_name)
                self_position = np.array([getattr(temp_atom, i) for i in "xyz"])
                self_position = np.dot(rotation, (self_position - center1)) + center2
                for connected_atom in self.type.connectivity[temp_atom]:
                    if connected_atom.name in t:
                        connected_position = np.array([getattr(connected_atom, i) for i in "xyz"])
                        connected_position = np.dot(rotation, (connected_position - center1)) + center2
                        fact_connected_atom = self._name2atom[connected_atom.name]
                        x_ = self_position[0] - connected_position[0] + fact_connected_atom.x
                        y_ = self_position[1] - connected_position[1] + fact_connected_atom.y
                        z_ = self_position[2] - connected_position[2] + fact_connected_atom.z
                        t.add(atom_name)
                        movedlist.append(atom_name)
                        self.Add_Atom(atom_name, x=x_, y=y_, z=z_)
                        self.atoms[-1].bad_coordinate = True
                        break
            for atom_name in movedlist:
                uncertified.remove(atom_name)

    def deepcopy(self, forcopy=None):
        """
        This **function** is used to deep copy the instance

        :param forcopy: the key to help you find who it is from
        :return: the new instance
        """
        new_residue = Residue(self.type)
        for atom in self.atoms:
            new_residue.Add_Atom(atom)
            new_residue.atoms[-1].contents = dict(atom.contents.items())
            if forcopy:
                atom.copied[forcopy] = new_residue.atoms[-1]

        return new_residue


set_classmethod_alternative_names(Residue)


class ResidueLink:
    """
    This **class** is a class for the link between residues

    :param atom1: the first atom to link
    :param atom2: the second atom to link
    """

    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        """the first atom to link"""
        self.atom2 = atom2
        """the second atom to link"""
        self.built = False
        """indicates the instance built or not"""
        self.bonded_forces = {frc.get_class_name(): [] for frc in GlobalSetting.BondedForces}
        """the bonded forces after building"""
        self.tohash = ResidueLink.get_hash(atom1, atom2, "atom")
        self.residue_tohash = ResidueLink.get_hash(atom1.residue, atom2.residue, "residue")
        set_attribute_alternative_names(self)

    def __repr__(self):
        return "Entity of ResidueLink: " + repr(self.atom1) + "-" + repr(self.atom2)

    def __hash__(self):
        return self.tohash

    def __eq__(self, other):
        return isinstance(other, ResidueLink) and self.tohash == other.tohash

    @staticmethod
    def get_hash(one, other, key="atom"):
        """
        This **function** is used to get the hash value of the ResidueLink

        :param one: one Atom or Residue of the link
        :param other: the other Atom or Residue of the link
        :param key: "atom" or "residue", to specify the key
        :return: the hash value
        """
        if key == "atom":
            tohash = [f"{one}-{other}", f"{other}-{one}"]
        elif key == "residue":
            tohash = [f"{one}-{other}", f"{other}-{one}"]
        else:
            raise ValueError(f'key should be "atom" or "residue", but got {key}')
        tohash.sort()
        return hash(tohash[0])

    def add_bonded_force(self, bonded_force_entity):
        """
        This **function** is used to add the bonded force to the residue link

        :param bonded_force_entity: the bonded force instance
        :return: None
        """
        if bonded_force_entity.get_class_name() not in self.bonded_forces.keys():
            self.bonded_forces[bonded_force_entity.get_class_name()] = []
        self.bonded_forces[bonded_force_entity.get_class_name()].append(bonded_force_entity)

    def deepcopy(self, forcopy):
        """
        This **function** is used to deep copy the instance

        :param forcopy: the key to help you find who it is from
        :return: the new instance
        """
        if self.atom1.copied[forcopy] and self.atom2.copied[forcopy]:
            return ResidueLink(self.atom1.copied[forcopy], self.atom2.copied[forcopy])
        raise Exception("Not Right forcopy")


class Molecule():
    """
    This **class** is a class for molecules

    :param name: a string, the name of the Molecule or a ResidueType instance
    """
    _all = Xdict()
    _save_functions = Xdict()
    _mindsponge_todo = Xdict()

    def __init__(self, name):
        self.name = ""
        """the name of the Molecule"""
        if isinstance(name, ResidueType):
            self.name = name.name
        else:
            self.name = name
        Molecule._all[self.name] = self
        self.residues = []
        """the residues in the molecule"""
        self.atoms = []
        """the atoms in the molecule

        .. NOTE::

            It may not be initialized. Maybe it will become a property in the next version to avoid mistakes.

        """
        self.atom_index = []
        """the atom index"""
        self.residue_links = set()
        """the residue links in the molecule"""
        self._residue_links_map = Xdict(atom=Xdict(), residue=Xdict(),
                                        not_found_message='key should be "atom" or "residue", but got {}')

        self.bonded_forces = Xdict()
        """the bonded forces after building"""
        self.built = False
        """indicates the instance built or not"""
        self.box_length = None
        """the box length of the molecule"""
        self.box_angle = [90.0, 90.0, 90.0]
        """the box angle of the molecule"""
        self._missing_residues = Xdict(not_found_message="These are no missing residues between {}")
        if isinstance(name, ResidueType):
            new_residue = Residue(name)
            for i in name.atoms:
                new_residue.Add_Atom(i)
            self.Add_Residue(new_residue)

        set_attribute_alternative_names(self)

    def __repr__(self):
        return "Entity of Molecule: " + self.name

    def __getattribute__(self, attr):
        if getattr(AtomType, "_parameters").get(attr, None) == float:
            return np.sum([getattr(atom, attr) for res in self.residues for atom in res.atoms])
        return super().__getattribute__(attr)

    @classmethod
    def cast(cls, other, deepcopy=False):
        """
        This **function** casts a Residue, a ResidueType or a Molecule to a Molecule
        **New From 1.2.6.6**

        :param other: a Residue, a ResidueType or a Molecule instance
        :param deepcopy: whether to deepcopy the other instance
        :return: a Molecule instance
        """
        if isinstance(other, Molecule):
            if deepcopy:
                return other.deepcopy()
            return other
        if isinstance(other, Residue):
            new_molecule = Molecule(other.name)
            if deepcopy:
                other = other.deepcopy()
            new_molecule.Add_Residue(other)
            return new_molecule
        if isinstance(other, ResidueType):
            new_molecule = Molecule(other.name)
            res_b = Residue(other)
            for atom in other.atoms:
                res_b.Add_Atom(atom)
            new_molecule.Add_Residue(res_b)
            return new_molecule
        raise TypeError(f"Only Residue, ResidueType or Molecule can be cast to Molecule, but {type(other)} found")

    @classmethod
    def set_save_sponge_input(cls, keyname):
        """
        This **function** is used to set the function when ``Save_SPONGE_Input``.
        It is recommended used as a **decorator**.

        :param keyname: the file prefix to save
        :return: the decorator
        """

        def wrapper(func):
            cls._save_functions[keyname] = func
            return func

        return wrapper

    @classmethod
    def del_save_sponge_input(cls, keyname):
        """
        This **function** is used to delete the function when ``Save_SPONGE_Input``.

        :param keyname: the file prefix to save
        :return: None
        """
        cls._save_functions.pop(keyname)

    @classmethod
    def set_mindsponge_todo(cls, keyname):
        """
        This **function** is used to set the function when ``Save_SPONGE_Input``.
        It is recommended used as a **decorator**.

        :param keyname: the file prefix to save
        :return: the decorator
        """

        def wrapper(func):
            cls._mindsponge_todo[keyname] = func
            return func

        return wrapper

    @classmethod
    def del_mindsponge_todo(cls, keyname):
        """
        This **function** is used to delete the function when ``Save_SPONGE_Input``.

        :param keyname: the file prefix to save
        :return: None
        """
        cls._mindsponge_todo.pop(keyname)

    @staticmethod
    def _set_friends_in_different_residue(molecule, atom1, atom2):
        """

        :param molecule:
        :param atom1:
        :param atom2:
        :return:
        """
        if molecule.atom_index[atom1.residue.atoms[0]] < molecule.atom_index[atom1.residue.atoms[1]]:
            res_index = molecule.atom_index[atom1.residue.atoms[-1]]
            atom1_friends = list(range(res_index + 1))
            atom2_friends = list(range(res_index + 1, len(molecule.atoms)))
        else:
            res_index = molecule.atom_index[atom2.residue.atoms[-1]]
            atom2_friends = list(range(res_index + 1))
            atom1_friends = list(range(res_index + 1, len(molecule.atoms)))
        return atom1_friends, atom2_friends

    @staticmethod
    def _get_head_and_tail(head, tail, molecule, typeatom1, typeatom2, restype, toset, atom1_friends, resatom_):
        """

        :param head:
        :param tail:
        :param molecule:
        :param typeatom1:
        :param typeatom2:
        :param restype:
        :param toset:
        :param atom1_friends:
        :param resatom_:
        :return:
        """
        assert typeatom2 in restype.connectivity[typeatom1]
        to_search = set(restype.connectivity[typeatom1])
        to_search.remove(typeatom2)

        searched = set()
        searched.add(typeatom2)
        while to_search:
            atom0 = to_search.pop()
            if atom0 in searched:
                continue
            if atom0.name == restype.head:
                head = toset
            elif atom0.name == restype.tail:
                tail = toset
            searched.add(atom0)
            atom1_friends.append(molecule.atom_index[resatom_(atom0)])
            to_search |= set(restype.connectivity[atom0]) - searched
        return head, tail

    def find_spacious_direction(self, point, atom_positions=None):
        """
        This **function** is used to find the most spacious (lowest atom density) direction of \
the givin point around the atom positions
        **New From 1.2.6.7**

        :param point: a list of 3 numbers, the point to find the most spacious direction
        :param atom_positions: the atom positions around the point. If None, this will use the positions of the atoms \
in this Molecule
        :return: a numpy array with the shape (3,), the most spacious direction
        """
        point = np.array(point, dtype=np.float32).reshape(3)
        direction = np.zeros_like(point)
        if atom_positions is None:
            atom_positions = np.array([[getattr(atom, i) for i in "xyz"]
                                       for residue in self.residues for atom in residue.atoms],
                                      dtype=np.float32)
        displace = atom_positions - point
        distance = np.linalg.norm(displace, axis=1, keepdims=True)
        filter_ = distance > 1
        displace = displace[np.broadcast_to(filter_, (len(filter_), 3))].reshape(-1, 3)
        distance = distance[distance > 1]
        direction -= np.sum(displace * np.power(distance, -4).reshape(-1, 1))
        direction /= np.linalg.norm(direction)
        return direction

    def add_residue(self, residue):
        """
        This **function** is used to add a residue to the molecule

        :param residue: the residue to add, either a Residue instance or a ResidueType instance
        :return: None
        """
        if isinstance(residue, Residue):
            self.built = False
            self.residues.append(residue)
        elif isinstance(residue, ResidueType):
            self.built = False
            self.residues.append(Residue(residue, directly_copy=True))

    def add_bonded_force(self, bonded_force_entity):
        """
        This **function** is used to add the bonded force to the residue type.

        :param bonded_force_entity: the bonded force instance
        :return: None
        """
        if bonded_force_entity.get_class_name() not in self.bonded_forces.keys():
            self.bonded_forces[bonded_force_entity.get_class_name()] = []
        self.bonded_forces[bonded_force_entity.get_class_name()].append(bonded_force_entity)

    def add_residue_link(self, atom1, atom2):
        """
        This **function** is used to add the connectivity between two atoms of two residues in the molecule.

        :param atom1: the first atom
        :param atom2: the second atom
        :return: None
        """
        self.built = False
        reslink = ResidueLink(atom1, atom2)
        self.residue_links.add(reslink)
        self._residue_links_map["atom"][reslink.tohash] = reslink
        self._residue_links_map["residue"][reslink.residue_tohash] = reslink

    def get_residue_link(self, one, other, key="atom"):
        """
        This **function** is used to get the ResidueLink between two residues

        :param one: one Atom or Residue of the ResidueLink
        :param other: the other Atom or Residue of the ResidueLink
        :param key: "atom" or "residue", to specify the key
        :return: the ResidueLink or None if not found
        """
        tohash = ResidueLink.get_hash(one, other, key)
        return self._residue_links_map[key].get(tohash, None)

    def del_residue_link(self, one, other, key="atom"):
        """
        This **function** is used to delete the ResidueLink between two residues

        :param one: one Atom or Residue of the ResidueLink
        :param other: the other Atom or Residue of the ResidueLink
        :param key: "atom" or "residue", to specify the key
        :return: None
        """
        tohash = ResidueLink.get_hash(one, other, key)
        reslink = self._residue_links_map[key][tohash]
        self.residue_links.remove(reslink)
        self._residue_links_map["atom"].pop(reslink.tohash)
        self._residue_links_map["residue"].pop(reslink.residue_tohash)

    def add_missing_atoms(self):
        """
        This **function** is used to add the missing atoms from the ResidueType instances to the molecule.

        :return: None
        """
        for residue in self.residues:
            residue.Add_Missing_Atoms()

    def set_missing_residues_info(self, start, end, missing_residues):
        """
        This **function** is used to set the information about the missing residues
        **New From 1.2.6.7**

        :param start: the residue or the residue index where the missing residues start. `None` for no starting residue.
        :param end: the residue  or the residue index where the missing residues end. `None` for no ending residue.
        :param missing_residues: the missing residues. The parameter can be a string separated by space, \
or a list of residue names, or a list of ResidueType or None. \
If None, the information will be deleted between start and end
        :return: True for success, False for failure to set
        """
        if start is not None and isinstance(start, int):
            start = self.residues[start]
        if end is not None and isinstance(end, int):
            end = self.residues[end]
        if start is None and end is None:
            raise ValueError("starting and ending residues can not be None at the same time")
        key = (start, end)
        if missing_residues is None:
            if key in self._missing_residues:
                self._missing_residues.pop(key)
                return True
            return False
        if isinstance(missing_residues, str):
            missing_residues = [ResidueType.get_type(res) for res in missing_residues.split()]
        elif isinstance(missing_residues, Iterable):
            for i, res in enumerate(missing_residues):
                if isinstance(res, str):
                    res = ResidueType.get_type(res)
                missing_residues[i] = res
        res = missing_residues[0]
        if start is None and res.name in GlobalSetting.PDBResidueNameMap["head"]:
            missing_residues[0] = ResidueType.get_type(GlobalSetting.PDBResidueNameMap["head"][res.name])
        res = missing_residues[-1]
        if end is None and res.name in GlobalSetting.PDBResidueNameMap["tail"]:
            missing_residues[-1] = ResidueType.get_type(GlobalSetting.PDBResidueNameMap["tail"][res.name])
        self._missing_residues[key] = missing_residues
        return True

    def clear_missing_residues_info(self):
        """
        This **function** is used to clear the information about the missing residues
        **New From 1.2.6.7**

        :return: None
        """
        self._missing_residues.clear()

    def add_missing_residues(self):
        """
        This **function** is used to add the missing residues according to the information of the missing residues.
        The information of the missing residues may be set by `set_missing_residues_info` or `load_pdb`
        **New From 1.2.6.7**

        :return: None
        """
        for (start, end), to_insert in self._missing_residues.items():
            if start is not None and end is not None:
                ref_res = start
                tail = start.name2atom(start.type.tail)
                link_to_tail = end.name2atom(end.type.head)
                insert_index = self.residues.index(start) + 1
                start_position = np.array([getattr(tail, i) for i in "xyz"])
                end_position = np.array([getattr(link_to_tail, i) for i in "xyz"])
                loop_direction = self.find_spacious_direction(start_position)
                self.del_residue_link(tail, link_to_tail)
            elif start is not None and end is None:
                ref_res = start
                start.unterminal()
                tail = start.name2atom(start.type.tail)
                link_to_tail = None
                insert_index = self.residues.index(start) + 1
                start_position = np.array([getattr(tail, i) for i in "xyz"])
                end_position = self.find_spacious_direction(start_position)
                norm = np.linalg.norm(end_position)
                if norm != 0:
                    end_position /= norm
                end_position *= len(to_insert) * 5
                end_position += start_position
                loop_direction = np.zeros_like(start_position)
            elif start is None and end is not None:
                tail = None
                ref_res = end
                end.unterminal()
                link_to_tail = end.name2atom(end.type.head)
                insert_index = 0
                end_position = np.array([getattr(link_to_tail, i) for i in "xyz"])
                start_position = self.find_spacious_direction(end_position)
                norm = np.linalg.norm(start_position)
                if norm != 0:
                    start_position /= norm
                start_position *= len(to_insert) * 5
                start_position += end_position
                loop_direction = np.zeros_like(end_position)
            else:
                raise ValueError("starting and ending residues can not be None at the same time")
            self._add_one_missing_residue(start_position, end_position, to_insert, ref_res,
                                          loop_direction, insert_index, tail, link_to_tail)

    def deepcopy(self):
        """
        This **function** is used to deep copy the instance

        :return: the new instance
        """
        new_molecule = Molecule(self.name)
        forcopy = hash(str(time.time()))
        for res in self.residues:
            new_molecule.Add_Residue(res.deepcopy(forcopy))

        for link in self.residue_links:
            new_link = link.deepcopy(forcopy)
            new_molecule.add_residue_link(new_link.atom1, new_link.atom2)

        for res in self.residues:
            for atom in res.atoms:
                atom.copied[forcopy].Extra_Exclude_Atoms(
                    map(lambda aton: aton.copied[forcopy], atom.extra_excluded_atoms))

        if self.built:
            for bond_entities in self.bonded_forces.values():
                for bond_entity in bond_entities:
                    new_molecule.Add_Bonded_Force(bond_entity.deepcopy(forcopy))
            new_molecule.built = True
            new_molecule.atoms = [atom for residue in new_molecule.residues for atom in residue.atoms]
            new_molecule.atom_index = {new_molecule.atoms[i]: i for i in range(len(new_molecule.atoms))}
            for atom in self.atoms:
                atom.copied[forcopy].linked_atoms = {key: set(map(lambda aton: aton.copied[forcopy], value)) for
                                                     key, value in atom.linked_atoms.items()}

        for res in self.residues:
            for atom in res.atoms:
                atom.copied.pop(forcopy)
        return new_molecule

    def get_atom_coordinates(self):
        """
        This **function** is used to get the atom coordinates

        :return: a numpy array, the coordinates of atoms
        """
        self.atoms = []
        for res in self.residues:
            self.atoms.extend(res.atoms)

        self.atom_index = {self.atoms[i]: i for i in range(len(self.atoms))}
        return np.array([[atom.x, atom.y, atom.z] for atom in self.atoms])

    def divide_into_two_parts(self, atom1, atom2):
        """
        This **function** is used to divide the molecule into two parts

        :param atom1: the first atom
        :param atom2: the second atom
        :return: two numpy arrays, the index of the two parts
        """
        if atom1.residue != atom2.residue:
            atom1_friends_np, atom2_friends_np = self._set_friends_in_different_residue(self, atom1, atom2)
        else:
            atom1_friends = []
            atom2_friends = []
            head = 0
            tail = 0

            restype = atom1.residue.type

            def _restype_atom(atom):
                return restype.name2atom(atom.name)

            def _resatom(atom):
                return atom1.residue.name2atom(atom.name)

            typeatom1 = _restype_atom(atom1)
            typeatom2 = _restype_atom(atom2)

            head, tail = self._get_head_and_tail(head, tail, self, typeatom1, typeatom2, restype, 1, atom1_friends,
                                                 _resatom)
            head, tail = self._get_head_and_tail(head, tail, self, typeatom2, typeatom1, restype, 2, atom2_friends,
                                                 _resatom)

            if atom1.name == restype.head:
                head = 1
            elif atom1.name == restype.tail:
                tail = 1
            if atom2.name == restype.head:
                head = 2
            elif atom2.name == restype.tail:
                tail = 2

            resindex_head = min(self.atom_index[atom1.residue.atoms[0]],
                                self.atom_index[atom2.residue.atoms[0]])
            resindex_tail = max(self.atom_index[atom1.residue.atoms[-1]],
                                self.atom_index[atom2.residue.atoms[-1]])

            if head == 1:
                atom1_friends.extend(list(range(resindex_head)))
            else:
                atom2_friends.extend(list(range(resindex_head)))
            if tail == 1:
                atom1_friends.extend(list(range(resindex_tail + 1, len(self.atoms))))
            else:
                atom2_friends.extend(list(range(resindex_tail + 1, len(self.atoms))))

            atom1_friends_set = set(atom1_friends)
            atom1_friends_set.add(self.atom_index[atom1])
            atom1_friends_np = np.array(list(atom1_friends_set))
            atom2_friends_set = set(atom2_friends)
            atom2_friends_set.add(self.atom_index[atom2])
            atom2_friends_np = np.array(list(atom2_friends_set))
        return atom1_friends_np, atom2_friends_np

    def _add_one_missing_residue(self, start_position, end_position, to_insert,
                                 ref_res, loop_direction, insert_index, tail, link_to_tail):
        """
        add one missing residue
        """
        distance = np.linalg.norm(start_position - end_position)
        height = np.max((0, len(to_insert) * 3 - distance)) / 2
        ref_res = Xdict({atom.name: [atom.x, atom.y, atom.z] for atom in ref_res.atoms})
        for i, restype in enumerate(to_insert):
            res = Residue(restype, directly_copy=True)
            positions = [[getattr(atom, j) for j in "xyz"] for atom in res.atoms]
            p1, p2 = [], []
            for j, atom in enumerate(res.atoms):
                if atom.name in ref_res:
                    p1.append(ref_res[atom.name])
                    p2.append(positions[j])
            rotate_matrix, _, res_center = kabsch(p1, p2)
            fraction = i / len(to_insert)
            translate = start_position + (end_position - start_position) * fraction + \
                        height * np.sin(fraction * np.pi) * loop_direction - res_center
            positions = np.dot(rotate_matrix, np.array(positions).transpose()).transpose() + translate
            for j, atom in enumerate(res.atoms):
                atom.x = positions[j][0]
                atom.y = positions[j][1]
                atom.z = positions[j][2]
                atom.bad_coordinate = True
            self.residues.insert(insert_index + i, res)
            if tail is not None:
                self.add_residue_link(tail, res.name2atom(restype.head))
            if restype.tail is not None:
                tail = res.name2atom(restype.tail)
            else:
                tail = None
        if link_to_tail is not None and tail is not None:
            self.add_residue_link(tail, link_to_tail)


def _link_residue_process_coordinate(molecule, atom1, atom2):
    """

    :param molecule:
    :param atom1:
    :param atom2:
    :return:
    """
    molecule.atoms = [atom for residue in molecule.residues for atom in residue.atoms]
    molecule.atom_index = {atom: i for i, atom in enumerate(molecule.atoms)}
    res_a = atom1.residue
    res_b = atom2.residue
    crd = molecule.get_atom_coordinates()
    atom1_friends, atom2_friends = molecule.divide_into_two_parts(atom1, atom2)
    crd[atom2_friends] += 2000

    bond_length = (res_a.type.tail_length + res_b.type.head_length) / 2
    r0 = crd[molecule.atom_index[atom2]] - crd[molecule.atom_index[atom1]]
    l0 = np.linalg.norm(r0)
    dr = (bond_length / l0 - 1) * r0
    crd[atom2_friends] += dr

    res = res_a
    atom_a = atom1
    atom_b = atom2
    atom_b_friends = atom2_friends
    for link_conditions in res.type.tail_link_conditions:
        atoms = [res.name2atom(atom) for atom in link_conditions["atoms"]]
        parameter = link_conditions["parameter"]
        if len(atoms) == 1:
            r0 = crd[molecule.atom_index[atom_b]] - crd[molecule.atom_index[atoms[0]]]
            l0 = np.linalg.norm(r0)
            dr = (parameter / l0 - 1) * r0
            crd[atom_b_friends] += dr
        elif len(atoms) == 2:
            r_ao = crd[molecule.atom_index[atoms[0]]] - crd[molecule.atom_index[atoms[1]]]
            r_ob = crd[molecule.atom_index[atom_b]] - crd[molecule.atom_index[atoms[1]]]
            angle0 = np.arccos(np.dot(r_ao, r_ob) / np.linalg.norm(r_ao) / np.linalg.norm(r_ob))
            delta_angle = parameter - angle0
            crd[atom_b_friends] = np.dot(crd[atom_b_friends] - crd[molecule.atom_index[atoms[1]]],
                                         get_rotate_matrix(np.cross(r_ao, r_ob), delta_angle)) + \
                                         crd[molecule.atom_index[atoms[1]]]
        elif len(atoms) == 3:
            r12 = crd[molecule.atom_index[atoms[0]]] - crd[molecule.atom_index[atoms[1]]]
            r23 = crd[molecule.atom_index[atoms[2]]] - crd[molecule.atom_index[atoms[1]]]
            r34 = crd[molecule.atom_index[atoms[2]]] - crd[molecule.atom_index[atom_b]]
            r12xr23 = np.cross(r12, r23)
            r34xr23 = np.cross(r34, r23)
            cos = np.dot(r12xr23, r34xr23) / np.linalg.norm(r12xr23) / np.linalg.norm(r34xr23)
            cos = max(-0.999999, min(cos, 0.999999))
            dihedral0 = np.arccos(cos)
            dihedral0 = np.pi - np.copysign(dihedral0, np.cross(r34xr23, r12xr23).dot(r23))
            delta_angle = parameter - dihedral0
            crd[atom_b_friends] = np.dot(crd[atom_b_friends] - crd[molecule.atom_index[atoms[2]]],
                                         get_rotate_matrix(r23, delta_angle)) + crd[molecule.atom_index[atoms[2]]]

    res = res_b
    atom_a = atom2
    atom_b = atom1
    atom_b_friends = atom1_friends
    for link_conditions in res.type.head_link_conditions:
        atoms = [res.name2atom(atom) for atom in link_conditions["atoms"]]
        parameter = link_conditions["parameter"]
        if len(atoms) == 1:
            r0 = crd[molecule.atom_index[atom_b]] - crd[molecule.atom_index[atoms[0]]]
            l0 = np.linalg.norm(r0)
            dr = (parameter / l0 - 1) * r0
            crd[atom_b_friends] += dr
        elif len(atoms) == 2:
            r_ao = crd[molecule.atom_index[atoms[0]]] - crd[molecule.atom_index[atoms[1]]]
            r_ob = crd[molecule.atom_index[atom_b]] - crd[molecule.atom_index[atoms[1]]]
            angle0 = np.arccos(np.dot(r_ao, r_ob) / np.linalg.norm(r_ao) / np.linalg.norm(r_ob))
            delta_angle = parameter - angle0
            crd[atom_b_friends] = np.dot(crd[atom_b_friends] - crd[molecule.atom_index[atoms[1]]],
                                         get_rotate_matrix(np.cross(r_ao, r_ob), delta_angle)) + \
                                         crd[molecule.atom_index[atoms[1]]]
        elif len(atoms) == 3:
            r_oo = crd[molecule.atom_index[atoms[0]]] - crd[molecule.atom_index[atoms[1]]]
            r_oa = crd[molecule.atom_index[atoms[1]]] - crd[molecule.atom_index[atoms[2]]]
            r_ab = crd[molecule.atom_index[atoms[2]]] - crd[molecule.atom_index[atom_b]]
            r12xr23 = np.cross(r_oo, r_oa)
            r34xr23 = np.cross(r_ab, r_oa)
            cos = np.dot(r12xr23, r34xr23) / np.linalg.norm(r12xr23) / np.linalg.norm(r34xr23)
            cos = max(-0.999999, min(cos, 0.999999))
            dihedral0 = np.arccos(cos)
            dihedral0 = np.pi - np.copysign(dihedral0, np.cross(r34xr23, r12xr23).dot(r_oa))
            delta_angle = parameter - dihedral0
            crd[atom_b_friends] = np.dot(crd[atom_b_friends] - crd[molecule.atom_index[atoms[2]]],
                                         get_rotate_matrix(r_oa, delta_angle)) + crd[molecule.atom_index[atoms[2]]]

    if res_a.type.tail_next and res_b.type.head_next:
        atom_a = res_a.name2atom(res_a.type.tail_next)
        atom_b = res_b.name2atom(res_b.type.head_next)
        r_oo = crd[molecule.atom_index[atom_a]] - crd[molecule.atom_index[atom1]]
        r_oa = crd[molecule.atom_index[atom1]] - crd[molecule.atom_index[atom2]]
        r_ab = crd[molecule.atom_index[atom2]] - crd[molecule.atom_index[atom_b]]
        r12xr23 = np.cross(r_oo, r_oa)
        r34xr23 = np.cross(r_ab, r_oa)
        cos = np.dot(r12xr23, r34xr23) / np.linalg.norm(r12xr23) / np.linalg.norm(r34xr23)
        cos = max(-0.999999, min(cos, 0.999999))
        dihedral0 = np.arccos(cos)
        dihedral0 = np.pi - np.copysign(dihedral0, np.cross(r34xr23, r12xr23).dot(r_oa))
        delta_angle = np.pi - dihedral0
        crd[atom2_friends] = np.dot(crd[atom2_friends] - crd[molecule.atom_index[atom2]],
                                    get_rotate_matrix(r_oa, delta_angle)) + crd[molecule.atom_index[atom2]]

    for atom in molecule.atoms:
        i = molecule.atom_index[atom]
        atom.x = crd[i][0]
        atom.y = crd[i][1]
        atom.z = crd[i][2]


set_classmethod_alternative_names(Molecule)
set_classmethod_alternative_names(ResidueLink)
Entity.register(ResidueLink)
Entity.register(Molecule)
AbstractMolecule.register(Residue)
AbstractMolecule.register(ResidueType)
AbstractMolecule.register(Molecule)


def _add(self, other, deepcopy, link):
    """

    :param self:
    :param other:
    :return:
    """
    if isinstance(other, AbstractMolecule):
        new_molecule = Molecule.cast(self, deepcopy=deepcopy)
        new_molecule.built = False
        other_molecule = Molecule.cast(other, deepcopy=True)
        try:
            res_a = new_molecule.residues[-1]
        except IndexError:
            res_a = None
        res_b = other_molecule.residues[0]
        new_molecule.residues += other_molecule.residues
        new_molecule.residue_links |= other_molecule.residue_links
        if link and res_a and res_a.type.tail and res_b.type.head:
            atom1 = res_a.name2atom(res_a.type.tail)
            atom2 = res_b.name2atom(res_b.type.head)
            new_molecule.Add_Residue_Link(atom1, atom2)
            _link_residue_process_coordinate(new_molecule, atom1, atom2)
        return new_molecule
    if other is None:
        return self

    raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" % (type(self), type(other)))


def _mul(self, other, deepcopy):
    """

    :param self:
    :param other:
    :return:
    """
    if isinstance(other, int):
        if other < 1:
            raise ValueError("multiple should be not less than 1")
        if isinstance(self, ResidueType) or deepcopy:
            t = self
        else:
            t = self.deepcopy()
        for _ in range(other - 1):
            t += self
        return t
    raise TypeError("unsupported operand type(s) for *: '%s' and '%s'" % (type(self), type(other)))


ResidueType.__add__ = partialmethod(_add, deepcopy=True, link=True)
ResidueType.__radd__ = partialmethod(_add, deepcopy=True, link=True)
ResidueType.__or__ = partialmethod(_add, deepcopy=True, link=False)
ResidueType.__ror__ = partialmethod(_add, deepcopy=True, link=False)
ResidueType.__mul__ = partialmethod(_mul, deepcopy=True)
ResidueType.__rmul__ = partialmethod(_mul, deepcopy=True)
Molecule.__add__ = partialmethod(_add, deepcopy=True, link=True)
Molecule.__radd__ = partialmethod(_add, deepcopy=True, link=True)
Molecule.__iadd__ = partialmethod(_add, deepcopy=False, link=True)
Molecule.__mul__ = partialmethod(_mul, deepcopy=True)
Molecule.__rmul__ = partialmethod(_mul, deepcopy=True)
Molecule.__imul__ = partialmethod(_mul, deepcopy=False)
Molecule.__or__ = partialmethod(_add, deepcopy=True, link=False)
Molecule.__ror__ = partialmethod(_add, deepcopy=True, link=False)
Molecule.__ior__ = partialmethod(_add, deepcopy=False, link=False)


def generate_new_bonded_force_type(type_name, atoms, properties, is_compulsory, is_multiple=False):
    """
    This **function** is used to generate the subclasses of the Type and the Entity for the bonded force

    :param type_name: the name of the type
    :param atoms: the position of the atoms. For example, "1-2-3-4" for dihedral and "1-3-2-3" for improper.
    :param properties: a dict mapping the name and the type of the force type
    :param is_compulsory: whether it is compulsory for the type
    :param is_multiple: whether it is multiple for one atom list
    :return: a new BondedForceType class
    """

    class BondedForceEntity(Entity):
        """
        This **class** is a subclass of Entity, for bonded forces
        """
        _name = type_name
        _count = 0

        def __init__(self, atoms, entity_type, name=None):
            super().__init__(entity_type, name)
            self.atoms = atoms

        def deepcopy(self, forcopy):
            """
            This **function** is used to deep copy the instance
            :return:
            """
            atoms_ = [atom.copied[forcopy] for atom in self.atoms]
            newone = type(self)(atoms_, self.type, self.name)
            newone.contents = {**self.contents}
            return newone

    temp = [int(i) for i in atoms.split("-")]

    class BondedForceType(Type):
        """
        This **class** is a subclass of Type, for bonded force types
        """
        _name = type_name
        topology_like = temp
        compulsory = is_compulsory
        multiple = is_multiple
        atom_numbers = len(atoms.split("-"))
        topology_matrix = [[temp[i] - j if i > j else 1 for i in range(len(atoms.split("-")))] for j in
                           range(len(atoms.split("-")))]
        _parameters = {
            "name": str,
        }
        entity = BondedForceEntity
        _types = Xdict(not_found_message="Bonded Force Type {} not found. Did you import the proper force field?")
        _types_different_name = Xdict(not_found_message="Bonded Force Type {} not found.\
 Did you import the proper force field?")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            for name in type(self).Same_Force(self.name):
                type(self)._types_different_name[name] = self

            if type(self).multiple:
                for key in self.multiple:
                    self.contents[key + "s"] = [self.contents[key]]
                    self.contents[key] = None
                self.contents["multiple_numbers"] = 1

        @classmethod
        def same_force(cls, atom_list):
            """
            This **function** receives a list of atoms and output all the same force permutations for the list
            :param atom_list:
            :return:
            """
            if isinstance(atom_list, str):
                atom_list_temp = [atom.strip() for atom in atom_list.split("-")]
                temp_ = [atom_list, "-".join(atom_list_temp[::-1])]
            else:
                temp_ = [atom_list, atom_list[::-1]]
            return temp_

        @classmethod
        def set_same_force_function(cls, func):
            cls.Same_Force = classmethod(func)

        def update(self, **kwargs):
            reset = 1
            if "reset" in kwargs.keys():
                reset = int(kwargs.pop("reset"))
            if reset:
                for key in self.contents.keys():
                    if key != "name":
                        self.contents[key] = None
                if type(self).multiple:
                    for key in self.multiple:
                        self.contents[key + "s"] = []
                    self.contents["multiple_numbers"] = 0
            super().update(**kwargs)
            if type(self).multiple:
                for key in type(self).multiple:
                    self.contents[key + "s"].append(self.contents[key])
                    self.contents[key] = None
                self.multiple_numbers += 1

    set_classmethod_alternative_names(BondedForceType)

    BondedForceType.Add_Property(properties)
    BondedForceType.New_From_String("""name
    UNKNOWNS""")

    GlobalSetting.BondedForces.append(BondedForceType)
    GlobalSetting.BondedForcesMap[getattr(BondedForceType, "_name")] = BondedForceType
    for i in atoms.split("-"):
        if int(i) > GlobalSetting.farthest_bonded_force:
            GlobalSetting.farthest_bonded_force = int(i)

    return BondedForceType


def generate_new_pairwise_force_type(type_name, properties):
    """
    This **function** is used to generate the subclasses of the Type and the Entity for the pairwise force

    :param type_name: the name of the force type
    :param properties: a dict mapping the name and the type of the force type
    :return: a new BondedForceType class
    """

    class PairwiseForceType(Type):
        """
        This **class** is a subclass of Type, for pairwise force types
        """
        _name = type_name
        _parameters = {
            "name": str,
        }
        _types = Xdict(not_found_message="{} not found. Did you import the proper force field?")

    set_classmethod_alternative_names(PairwiseForceType)
    PairwiseForceType.Add_Property(properties)

    return PairwiseForceType

set_global_alternative_names(True)
