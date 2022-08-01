"""
This **module** is the basic setting for the force field format of 3-parameter non bonded 1-4 interactions
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, GlobalSetting, set_global_alternative_names

# pylint: disable=invalid-name
NB14Type = Generate_New_Bonded_Force_Type("nb14_extra", "1-4",
                                          {"epsilon": float, "rmin": float, "sigma": float, "A": float, "B": float,
                                           "kee": float, "nb14_ee_factor": float}, False)

NB14Type.Set_Property_Unit("rmin", "distance", "A")
NB14Type.Set_Property_Unit("sigma", "distance", "A")
NB14Type.Set_Property_Unit("epsilon", "energy", "kcal/mol")
NB14Type.Set_Property_Unit("A", "energy·distance^6", "kcal/mol·A^6")
NB14Type.Set_Property_Unit("B", "energy·distance^12", "kcal/mol·A^12")

NB14Type.topology_matrix = [[1, -4],
                            [1, 1]]


def get_nb14_extra_lj(atom1, atom2):
    """
    This **function** is used to get the LJ parameters for the extra nb14 interactions

    :param atom1: the Atom instance
    :param atom2: the Atom instance
    :return: the A and B coefficients of LJ
    """
    from . import lj_base
    lj_type = lj_base.LJType

    a = 0
    b = 0

    lj_i = lj_type.get_type(atom1.LJtype + "-" + atom1.LJtype)
    lj_j = lj_type.get_type(atom2.LJtype + "-" + atom2.LJtype)
    finded = False
    findnames = [atom1.LJtype + "-" + atom2.LJtype,
                 atom2.LJtype + "-" + atom1.LJtype]

    for findname in findnames:
        if findname in lj_type.get_all_types():
            finded = True
            lj_ij = lj_type.get_type(findname)
            a = lj_type.combining_method_A(lj_ij.epsilon, lj_ij.rmin, lj_ij.epsilon, lj_ij.rmin)
            b = lj_type.combining_method_B(lj_ij.epsilon, lj_ij.rmin, lj_ij.epsilon, lj_ij.rmin)
            break
    if not finded:
        a = lj_type.combining_method_A(lj_i.epsilon, lj_i.rmin, lj_j.epsilon, lj_j.rmin)
        b = lj_type.combining_method_B(lj_i.epsilon, lj_i.rmin, lj_j.epsilon, lj_j.rmin)
    return a, b


def exclude_to_nb14_extra(molecule, atom1, atom2):
    """
    This **function** is used to calculate nb14_extra instead of non-bonded interactions for atom1 and atom2

    :param molecule: the Molecule instance
    :param atom1: the Atom instance
    :param atom2: the Atom instance
    :return: None
    """
    new_force = NB14Type.entity([atom1, atom2], NB14Type.get_type("UNKNOWNS"))
    a, b = Get_NB14EXTRA_AB(new_force.atoms[0], new_force.atoms[1])
    new_force.A = nb14_bond.kLJ * a
    new_force.B = nb14_bond.kLJ * b
    new_force.kee = nb14_bond.kee
    atom1.Extra_Exclude_Atom(atom2)

    molecule.Add_Bonded_Force(new_force)


def nb14_to_nb14_extra(molecule):
    """
    This **function** is used to convert nb14 to nb14_extra

    :param molecule: the Molecule instance
    :return: None
    """
    # 处理nb14
    # A、B中的nb14全部变为nb14_extra
    while molecule.bonded_forces.get("nb14", []):
        nb14_bond = molecule.bonded_forces["nb14"].pop()
        new_force = NB14Type.entity(nb14_bond.atoms, NB14Type.get_type("UNKNOWNS"), nb14_bond.name)

        a, b = get_nb14_extra_lj(new_force.atoms[0], new_force.atoms[1])

        new_force.A = nb14_bond.kLJ * a
        new_force.B = nb14_bond.kLJ * b
        new_force.kee = nb14_bond.kee
        new_force.nb14_ee_factor = nb14_bond.kee * new_force.atoms[0].charge * new_force.atoms[1].charge
        molecule.Add_Bonded_Force(new_force)


@GlobalSetting.Add_Unit_Transfer_Function(NB14Type)
def _lj_unit_transfer(self):
    """
    This **function** is used to transfer the units of lj

    :param self:
    :return:
    """
    if self.sigma is not None:
        self.rmin = self.sigma * (4 ** (1 / 12) / 2)
        self.sigma = None
    if self.rmin is not None and self.epsilon is not None:
        self.A = self.epsilon * (2 * self.rmin) ** 12
        self.B = self.epsilon * 2 * ((2 * self.rmin) ** 6)
        self.rmin = None
        self.epsilon = None


@Molecule.Set_Save_SPONGE_Input("nb14_extra")
def write_nb14(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    bonds = []
    for bond in self.bonded_forces.get("nb14_extra", []):
        order = list(range(2))
        if bond.A != 0 or bond.B != 0 or bond.kee != 0:
            if self.atom_index[bond.atoms[order[0]]] > self.atom_index[bond.atoms[order[-1]]]:
                temp_order = order[::-1]
            else:
                temp_order = order

            bonds.append("%d %d %.6e %.6e %.6e" % (self.atom_index[bond.atoms[temp_order[0]]],
                                                   self.atom_index[bond.atoms[temp_order[1]]],
                                                   bond.A, bond.B, bond.kee))

    if bonds:
        towrite = "%d\n" % len(bonds)
        bonds.sort(key=lambda x: list(map(int, x.split()[:2])))
        towrite += "\n".join(bonds)

        return towrite
    return None

set_global_alternative_names()
