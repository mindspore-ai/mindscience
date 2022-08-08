"""
This **module** is the basic setting for the force field format of 2-parameter non bonded 1-4 interactions
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, set_global_alternative_names, Xdict

# pylint: disable=invalid-name
NB14Type = Generate_New_Bonded_Force_Type("nb14", "1-4", {"kLJ": float, "kee": float}, True)

NB14Type.topology_matrix = [[1, -4],
                            [1, 1]]


@Molecule.Set_Save_SPONGE_Input("nb14")
def write_nb14(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    bonds = []
    for bond in self.bonded_forces.get("nb14", []):
        order = list(range(2))
        if bond.kLJ != 0 and bond.kee != 0:
            if self.atom_index[bond.atoms[order[0]]] > self.atom_index[bond.atoms[order[-1]]]:
                temp_order = order[::-1]
            else:
                temp_order = order
            bonds.append("%d %d %f %f" % (self.atom_index[bond.atoms[temp_order[0]]]
                                          , self.atom_index[bond.atoms[temp_order[1]]], bond.kLJ, bond.kee))

    if bonds:
        towrite = "%d\n" % len(bonds)
        bonds.sort(key=lambda x: list(map(int, x.split()[:2])))
        towrite += "\n".join(bonds)

        return towrite
    return None


#pylint: disable=unused-argument, relative-beyond-top-level
@Molecule.Set_MindSponge_Todo("nb14")
def _do(self, sys_kwarg, ene_kwarg, use_pbc):
    """

    :return:
    """
    from mindsponge.potential import NonbondPairwiseEnergy
    from .nb14_extra_base import get_nb14_extra_lj
    if "nb14" not in ene_kwarg:
        ene_kwarg["nb14"] = Xdict()
        ene_kwarg["nb14"]["function"] = lambda sys_kwarg, ene_kwarg: NonbondPairwiseEnergy(
            index=ene_kwarg["nb14"]["index"], use_pbc=use_pbc,
            qiqj=ene_kwarg["nb14"]["qiqj"],
            epsilon_ij=ene_kwarg["nb14"]["epsilon_ij"],
            sigma_ij=ene_kwarg["nb14"]["sigma_ij"],
            r_scale=ene_kwarg["nb14"]["r_scale"],
            r6_scale=ene_kwarg["nb14"]["r6_scale"],
            r12_scale=ene_kwarg["nb14"]["r12_scale"],
            length_unit="A", energy_unit="kcal/mol")
        ene_kwarg["nb14"]["index"] = []
        ene_kwarg["nb14"]["r_scale"] = []
        ene_kwarg["nb14"]["r6_scale"] = []
        ene_kwarg["nb14"]["r12_scale"] = []
        ene_kwarg["nb14"]["qiqj"] = []
        ene_kwarg["nb14"]["sigma_ij"] = []
        ene_kwarg["nb14"]["epsilon_ij"] = []
    nb14s = []
    r_scales = []
    r6_scales = []
    r12_scales = []
    qiqj = []
    sigma_ij = []
    epsilon_ij = []
    for bond in self.bonded_forces.get("nb14", []):
        if bond.kee != 1 and bond.kLJ != 1:
            atom1, atom2 = bond.atoms
            a, b = get_nb14_extra_lj(atom1, atom2)
            nb14s.append([self.atom_index[atom] for atom in bond.atoms])
            qiqj.append(atom1.charge * atom2.charge)
            if a != 0:
                sigma_ij.append((a/b)**(1/6))
                epsilon_ij.append(b*b/a/4)
            else:
                sigma_ij.append(0)
                epsilon_ij.append(0)
            r_scales.append(bond.kee)
            r6_scales.append(bond.kLJ)
            r12_scales.append(bond.kLJ)
    ene_kwarg["nb14"]["index"].append(nb14s)
    ene_kwarg["nb14"]["r_scale"].append(r_scales)
    ene_kwarg["nb14"]["r6_scale"].append(r6_scales)
    ene_kwarg["nb14"]["r12_scale"].append(r12_scales)
    ene_kwarg["nb14"]["qiqj"].append(qiqj)
    ene_kwarg["nb14"]["sigma_ij"].append(sigma_ij)
    ene_kwarg["nb14"]["epsilon_ij"].append(epsilon_ij)

set_global_alternative_names()
