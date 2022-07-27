"""
This **module** is the basic setting for the force field property of mass
"""
from ...helper import Molecule, AtomType

AtomType.Add_Property({"mass": float})


@Molecule.Set_Save_SPONGE_Input("mass")
def write_mass(self):
    """
    This **function** is used to write SPONGE input file
    :param self:
    :return:
    """
    towrite = "%d\n" % (len(self.atoms))
    towrite += "\n".join(["%.3f" % (atom.mass) for atom in self.atoms])
    return towrite
