"""
This **module** is the basic setting for the force field property of mass
"""
from ...helper import Molecule, AtomType, set_global_alternative_names

AtomType.Add_Property({"mass": float})


@Molecule.Set_Save_SPONGE_Input("mass")
def write_mass(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    towrite = "%d\n" % (len(self.atoms))
    towrite += "\n".join(["%.3f" % (atom.mass) for atom in self.atoms])
    return towrite

set_global_alternative_names()
