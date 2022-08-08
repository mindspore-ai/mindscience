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


#pylint: disable=unused-argument
@Molecule.Set_MindSponge_Todo("mass")
def _do_mass(self, sys_kwarg, ene_kwarg, use_pbc):
    """

    :return:
    """
    if "atom_mass" not in sys_kwarg:
        sys_kwarg["atom_mass"] = []
    sys_kwarg["atom_mass"].append([atom.mass for atom in self.atoms])

set_global_alternative_names()
