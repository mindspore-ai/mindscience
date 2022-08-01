"""
This **module** is the basic setting for the force field format of 2-parameter non bonded 1-4 interactions
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, set_global_alternative_names

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

set_global_alternative_names()
