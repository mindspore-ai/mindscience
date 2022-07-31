"""
This **module** is the basic setting for the force field format of virtual atoms
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, GlobalSetting, set_global_alternative_names

# pylint: disable=invalid-name
VirtualType2 = Generate_New_Bonded_Force_Type("vatom2", "1",
                                              {"atom0": int, "atom1": int, "atom2": int, "k1": float, "k2": float},
                                              False)
GlobalSetting.VirtualAtomTypes["vatom2"] = 3


@Molecule.Set_Save_SPONGE_Input("virtual_atom")
def write_virtual_atoms(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    vatoms = []
    for vatom in self.bonded_forces.get("vatom2", []):
        vatoms.append("2 %d %d %d %d %f %f" % (self.atom_index[vatom.atoms[0]],
                                               self.atom_index[vatom.atoms[0]] + vatom.atom0,
                                               self.atom_index[vatom.atoms[0]] + vatom.atom1,
                                               self.atom_index[vatom.atoms[0]] + vatom.atom2,
                                               vatom.k1, vatom.k2))

    if vatoms:
        towrite = ""
        towrite += "\n".join(vatoms)

        return towrite
    return None

set_global_alternative_names()
