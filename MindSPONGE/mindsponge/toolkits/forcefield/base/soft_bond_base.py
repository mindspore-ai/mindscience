"""
This **module** is the basic setting for the force field format of soft bond
"""
from ...helper import Molecule, set_global_alternative_names
from . import bond_base

bond_base.BondType.Add_Property({"from_AorB": int})


@Molecule.Set_Save_SPONGE_Input("bond_soft")
def write_bond(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    bonds = []
    for bond in self.bonded_forces.get("bond_soft", []):
        order = list(range(2))
        if bond.k != 0:
            if self.atom_index[bond.atoms[order[0]]] > self.atom_index[bond.atoms[order[-1]]]:
                temp_order = order[::-1]
            else:
                temp_order = order
            bonds.append("%d %d %f %f %d" % (self.atom_index[bond.atoms[temp_order[0]]]
                                             , self.atom_index[bond.atoms[temp_order[1]]], bond.k, bond.b,
                                             bond.from_AorB))

    if bonds:
        towrite = "%d\n" % len(bonds)
        bonds.sort(key=lambda x: list(map(int, x.split()[:2])))
        towrite += "\n".join(bonds)

        return towrite
    return None

set_global_alternative_names()
