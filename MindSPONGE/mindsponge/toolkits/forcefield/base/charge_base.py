"""
This **module** is the basic setting for the force field property of charge
"""
from ...helper import Molecule, AtomType

AtomType.Add_Property({"charge": float})

AtomType.Set_Property_Unit("charge", "charge", "e")


@Molecule.Set_Save_SPONGE_Input("charge")
def write_charge(self):
    """
    This **function** is used to write SPONGE input file
    :param self:
    :return:
    """
    towrite = "%d\n" % (len(self.atoms))
    towrite += "\n".join(["%.6f" % (atom.charge * 18.2223) for atom in self.atoms])
    return towrite
