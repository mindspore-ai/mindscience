"""
This **module** is the basic setting for the force field property of charge
"""
import numpy as np
from ... import Generate_New_Pairwise_Force_Type
from ...helper import Molecule, AtomType, GlobalSetting, set_global_alternative_names, Xdict

AtomType.Add_Property({"LJtype": str})

# pylint: disable=invalid-name
LJType = Generate_New_Pairwise_Force_Type("LJ",
                                          {"epsilon": float, "rmin": float, "sigma": float, "A": float, "B": float})

LJType.Set_Property_Unit("rmin", "distance", "A")
LJType.Set_Property_Unit("sigma", "distance", "A")
LJType.Set_Property_Unit("epsilon", "energy", "kcal/mol")
LJType.Set_Property_Unit("A", "energy·distance^6", "kcal/mol·A^6")
LJType.Set_Property_Unit("B", "energy·distance^12", "kcal/mol·A^12")


@GlobalSetting.Add_Unit_Transfer_Function(LJType)
def _lj_unit_transfer(self):
    """
    This **function** is used to transfer the units of lj

    :param self:
    :return:
    """
    if self.A is not None and self.B is not None:
        if self.B == 0 or self.A == 0:
            self.sigma = 0
            self.epsilon = 0
        else:
            self.sigma = (self.A / self.B) ** (1 / 6)
            self.epsilon = 0.25 * self.B * self.sigma ** (-6)
        self.A = None
        self.B = None
    if self.sigma is not None:
        self.rmin = self.sigma * (4 ** (1 / 12) / 2)
        self.sigma = None


def lorentz_berthelot_for_a(epsilon1, rmin1, epsilon2, rmin2):
    """
    This **function** is used to calculate the A coefficient for Lorentz_Berthelot mix rule

    :param epsilon1: the epsilon parameter of the first atom
    :param rmin1: the rmin parameter of the first atom
    :param epsilon2: the epsilon parameter of the second atom
    :param rmin2: the rmin parameter of the second atom
    :return: the A coefficient for the atom pair
    """
    return np.sqrt(epsilon1 * epsilon2) * ((rmin1 + rmin2) ** 12)


def lorentz_berthelot_for_b(epsilon1, rmin1, epsilon2, rmin2):
    """
    This **function** is used to calculate the B coefficient for Lorentz_Berthelot mix rule

    :param epsilon1: the epsilon parameter of the first atom
    :param rmin1: the rmin parameter of the first atom
    :param epsilon2: the epsilon parameter of the second atom
    :param rmin2: the rmin parameter of the second atom
    :return: the B coefficient for the atom pair
    """
    return np.sqrt(epsilon1 * epsilon2) * 2 * ((rmin1 + rmin2) ** 6)


def _find_ab_lj(ljtypes, stat=True):
    """

    :param ljtypes:
    :param stat:
    :return:
    """
    coefficients_a = []
    coefficients_b = []

    for i, lj_single_i in enumerate(ljtypes):
        lj_i = LJType.get_type(lj_single_i + "-" + lj_single_i)
        if stat:
            j_max = len(ljtypes)
        else:
            j_max = i + 1
        for j in range(j_max):
            lj_j = LJType.get_type(ljtypes[j] + "-" + ljtypes[j])
            finded = False
            findnames = [lj_single_i + "-" + ljtypes[j], ljtypes[j] + "-" + lj_single_i]
            for findname in findnames:
                if findname in LJType.get_all_types():
                    finded = True
                    lj_ij = LJType.get_type(findname)
                    coefficients_a.append(
                        LJType.combining_method_A(lj_ij.epsilon, lj_ij.rmin, lj_ij.epsilon, lj_ij.rmin))
                    coefficients_b.append(
                        LJType.combining_method_B(lj_ij.epsilon, lj_ij.rmin, lj_ij.epsilon, lj_ij.rmin))
                    break
            if not finded:
                coefficients_a.append(LJType.combining_method_A(lj_i.epsilon, lj_i.rmin, lj_j.epsilon, lj_j.rmin))
                coefficients_b.append(LJType.combining_method_B(lj_i.epsilon, lj_i.rmin, lj_j.epsilon, lj_j.rmin))
    return coefficients_a, coefficients_b


def _get_checks(ljtypes, coefficients_a, coefficients_b):
    """

    :param ljtypes:
    :param coefficients_a:
    :param coefficients_b:
    :return:
    """
    checks = Xdict()
    count = 0
    for i in range(len(ljtypes)):
        check_string_a = ""
        check_string_b = ""
        for _ in range(len(ljtypes)):
            check_string_a += "%16.7e" % coefficients_a[count] + " "
            check_string_b += "%16.7e" % coefficients_b[count] + " "
            count += 1

        checks[i] = check_string_a + check_string_b
    return checks


def _judge_same_type(ljtypes, checks):
    """

    :param ljtypes:
    :param checks:
    :return:
    """
    same_type = {i: i for i in range(len(ljtypes))}
    for i in range(len(ljtypes) - 1, -1, -1):
        for j in range(i + 1, len(ljtypes)):
            if checks[i] == checks[j]:
                same_type[j] = i
    return same_type


def _get_real_lj(ljtypes, same_type):
    """

    :param ljtypes:
    :param same_type:
    :return:
    """
    real_ljtypes = []
    tosub = 0
    for i, lj_single_i in enumerate(ljtypes):
        if same_type[i] == i:
            real_ljtypes.append(lj_single_i)
            same_type[i] -= tosub
        else:
            same_type[i] = same_type[same_type[i]]
            tosub += 1
    return real_ljtypes


def write_lj(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    ljtypes = []
    ljtypemap = Xdict()
    for atom in self.atoms:
        if atom.LJtype not in ljtypemap.keys():
            ljtypemap[atom.LJtype] = len(ljtypes)
            ljtypes.append(atom.LJtype)

    coefficients_a, coefficients_b = _find_ab_lj(ljtypes)

    checks = _get_checks(ljtypes, coefficients_a, coefficients_b)

    same_type = _judge_same_type(ljtypes, checks)

    real_ljtypes = _get_real_lj(ljtypes, same_type)

    real_as, real_bs = _find_ab_lj(real_ljtypes, False)

    towrite = "%d %d\n\n" % (len(self.atoms), len(real_ljtypes))
    count = 0
    for i in range(len(real_ljtypes)):
        for _ in range(i + 1):
            towrite += "%16.7e" % real_as[count] + " "
            count += 1
        towrite += "\n"
    towrite += "\n"

    count = 0
    for i in range(len(real_ljtypes)):
        for _ in range(i + 1):
            towrite += "%16.7e" % real_bs[count] + " "
            count += 1
        towrite += "\n"
    towrite += "\n"
    towrite += "\n".join(["%d" % (same_type[ljtypemap[atom.LJtype]]) for atom in self.atoms])
    return towrite


Molecule.Set_Save_SPONGE_Input("LJ")(write_lj)


#pylint: disable=unused-argument
def lj_todo(self, sys_kwarg, ene_kwarg, use_pbc):
    """
    This **function** is used to get MindSponge system and energy

    :param self: the Molecule instance
    :param sys_kwarg: a dict, the kwarg for system
    :param ene_kwarg: a dict, the kwarg for force field
    :return: None
    """
    from mindsponge.potential import LennardJonesEnergy
    if "lj" not in ene_kwarg:
        ene_kwarg["lj"] = Xdict()
        ene_kwarg["lj"]["function"] = lambda system, ene_kwarg: LennardJonesEnergy(
            epsilon=ene_kwarg["lj"]["epsilon"], use_pbc=use_pbc,
            sigma=ene_kwarg["lj"]["sigma"],
            length_unit='A', energy_unit='kcal/mol')
        ene_kwarg["lj"]["epsilon"] = []
        ene_kwarg["lj"]["sigma"] = []
    ljtypes = [LJType.get_type(atom.LJtype + "-" + atom.LJtype) for atom in self.atoms]
    ene_kwarg["lj"]["epsilon"].append([ljtype.epsilon for ljtype in ljtypes])
    ene_kwarg["lj"]["sigma"].append([ljtype.rmin / (4 ** (1 / 12) / 2) for ljtype in ljtypes])


Molecule.Set_MindSponge_Todo("LJ")(lj_todo)

set_global_alternative_names()
