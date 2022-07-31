"""
This **module** is the basic setting for the force field format of periodic proper and improper dihedral
"""
from itertools import permutations
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, set_global_alternative_names

# pylint: disable=invalid-name
ImproperType = Generate_New_Bonded_Force_Type("harmonic_improper", "1-3-2-3", {"k": float, "phi0": float}, False)

ImproperType.topology_matrix = [[1, 3, 2, 3],
                                [1, 1, 2, 3],
                                [1, 1, 1, 2],
                                [1, 1, 1, 1]]

ImproperType.Set_Property_Unit("k", "energy·angle^-2", "kcal/mol·rad^-2")
ImproperType.Set_Property_Unit("phi0", "angle", "rad")


@ImproperType.Set_Same_Force_Function
def improper_same_force(_, atom_list):
    """
    This **function** is used to return the same force type for an atom list
    :param _:
    :param atom_list:
    :return:
    """
    temp = []
    if isinstance(atom_list, str):
        atom_list_temp = [atom.strip() for atom in atom_list.split("-")]
        center_atom = atom_list_temp.pop(2)
        for atom_permutation in permutations(atom_list_temp):
            atom_permutation = list(atom_permutation)
            atom_permutation.insert(2, center_atom)
            temp.append("-".join(atom_permutation))
    else:
        atom_list_temp = list(atom_list)
        center_atom = atom_list_temp.pop(2)
        for atom_permutation in permutations(atom_list_temp):
            atom_permutation = list(atom_permutation)
            atom_permutation.insert(2, center_atom)
            temp.append(atom_permutation)
    return temp


@Molecule.Set_Save_SPONGE_Input("improper_dihedral")
def write_dihedral(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    dihedrals = []

    for dihedral in self.bonded_forces.get("harmonic_improper", []):
        if dihedral.k != 0:
            temp_order = [2, 0, 1, 3]
            dihedrals.append("%d %d %d %d %f %f" % (self.atom_index[dihedral.atoms[temp_order[0]]],
                                                    self.atom_index[dihedral.atoms[temp_order[1]]],
                                                    self.atom_index[dihedral.atoms[temp_order[2]]],
                                                    self.atom_index[dihedral.atoms[temp_order[3]]], dihedral.k,
                                                    dihedral.phi0))

    if dihedrals:
        towrite = "%d\n" % len(dihedrals)
        dihedrals.sort(key=lambda x: list(map(float, x.split())))
        towrite += "\n".join(dihedrals)

        return towrite
    return None

set_global_alternative_names()
