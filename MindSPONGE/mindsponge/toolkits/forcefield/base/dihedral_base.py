"""
This **module** is the basic setting for the force field format of periodic proper and improper dihedral
"""
from itertools import permutations
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, set_global_alternative_names, Xdict

# pylint: disable=invalid-name
ProperType = Generate_New_Bonded_Force_Type("dihedral", "1-2-3-4", {"k": float, "phi0": float, "periodicity": int},
                                            True, ["k", "phi0", "periodicity"])

ProperType.Set_Property_Unit("k", "energy", "kcal/mol")
ProperType.Set_Property_Unit("phi0", "angle", "rad")

# pylint: disable=invalid-name
ImproperType = Generate_New_Bonded_Force_Type("improper", "1-3-2-3", {"k": float, "phi0": float, "periodicity": int},
                                              False)
ImproperType.topology_matrix = [[1, 3, 2, 3],
                                [1, 1, 2, 3],
                                [1, 1, 1, 2],
                                [1, 1, 1, 1]]

ImproperType.Set_Property_Unit("k", "energy", "kcal/mol")
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


@Molecule.Set_Save_SPONGE_Input("dihedral")
def write_dihedral(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    dihedrals = []
    for dihedral in self.bonded_forces.get("dihedral", []):
        order = list(range(4))
        if self.atom_index[dihedral.atoms[order[0]]] > self.atom_index[dihedral.atoms[order[-1]]]:
            temp_order = order[::-1]
        else:
            temp_order = order
        for i in range(dihedral.multiple_numbers):
            if dihedral.ks[i] != 0:
                dihedrals.append("%d %d %d %d %d %f %f" % (self.atom_index[dihedral.atoms[temp_order[0]]]
                                                           , self.atom_index[dihedral.atoms[temp_order[1]]],
                                                           self.atom_index[dihedral.atoms[temp_order[2]]]
                                                           , self.atom_index[dihedral.atoms[temp_order[3]]],
                                                           dihedral.periodicitys[i], dihedral.ks[i], dihedral.phi0s[i]))

    for dihedral in self.bonded_forces.get("improper", []):
        order = list(range(4))
        if dihedral.k != 0:
            if self.atom_index[dihedral.atoms[order[0]]] > self.atom_index[dihedral.atoms[order[-1]]]:
                temp_order = order[::-1]
            else:
                temp_order = order
            dihedrals.append("%d %d %d %d %d %f %f" % (self.atom_index[dihedral.atoms[temp_order[0]]]
                                                       , self.atom_index[dihedral.atoms[temp_order[1]]],
                                                       self.atom_index[dihedral.atoms[temp_order[2]]]
                                                       , self.atom_index[dihedral.atoms[temp_order[3]]],
                                                       dihedral.periodicity, dihedral.k, dihedral.phi0))

    if dihedrals:
        towrite = "%d\n" % len(dihedrals)
        dihedrals.sort(key=lambda x: list(map(float, x.split())))
        towrite += "\n".join(dihedrals)

        return towrite
    return None


#pylint: disable=unused-argument
@Molecule.Set_MindSponge_Todo("dihedral")
def _do(self, sys_kwarg, ene_kwarg, use_pbc):
    """

    :return:
    """
    from mindsponge.potential import DihedralEnergy
    if "dihedral" not in ene_kwarg:
        ene_kwarg["dihedral"] = Xdict()
        ene_kwarg["dihedral"]["function"] = lambda system, ene_kwarg: DihedralEnergy(
            index=ene_kwarg["dihedral"]["index"], use_pbc=use_pbc,
            force_constant=ene_kwarg["dihedral"]["force_constant"],
            periodicity=ene_kwarg["dihedral"]["periodicity"],
            phase=ene_kwarg["dihedral"]["phase"],
            energy_unit="kcal/mol")
        ene_kwarg["dihedral"]["index"] = []
        ene_kwarg["dihedral"]["force_constant"] = []
        ene_kwarg["dihedral"]["periodicity"] = []
        ene_kwarg["dihedral"]["phase"] = []
    dihedrals = []
    force_constants = []
    periodicitys = []
    phases = []
    for dihedral in self.bonded_forces.get("dihedral", []):
        for i in range(dihedral.multiple_numbers):
            if dihedral.ks[i] != 0:
                dihedrals.append([self.atom_index[atom] for atom in dihedral.atoms])
                force_constants.append(dihedral.ks[i] * 2)
                periodicitys.append(dihedral.periodicitys[i])
                phases.append(dihedral.phi0s[i])
    for dihedral in self.bonded_forces.get("improper", []):
        if dihedral.k != 0:
            dihedrals.append([self.atom_index[atom] for atom in dihedral.atoms])
            force_constants.append(dihedral.k * 2)
            periodicitys.append(dihedral.periodicity)
            phases.append(dihedral.phi0)
    ene_kwarg["dihedral"]["index"].append(dihedrals)
    ene_kwarg["dihedral"]["force_constant"].append(force_constants)
    ene_kwarg["dihedral"]["periodicity"].append(periodicitys)
    ene_kwarg["dihedral"]["phase"].append(phases)

set_global_alternative_names()
