"""
This **module** is the basic setting for the force field property of charge
"""
import numpy as np
from ...helper import Molecule, AtomType, set_global_alternative_names, Xdict

AtomType.Add_Property({"charge": float})

AtomType.Set_Property_Unit("charge", "charge", "e")


@Molecule.Set_Save_SPONGE_Input("charge")
def write_charge(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    towrite = "%d\n" % (len(self.atoms))
    towrite += "\n".join(["%.6f" % (atom.charge * 18.2223) for atom in self.atoms])
    return towrite


@Molecule.Set_MindSponge_Todo("charge")
def _do(self, sys_kwarg, ene_kwarg, use_pbc):
    """

    :return:
    """
    from mindsponge.potential import CoulombEnergy
    if "atom_charge" not in sys_kwarg:
        sys_kwarg["atom_charge"] = []
    sys_kwarg["atom_charge"].append([atom.charge for atom in self.atoms])
    if "charge" not in ene_kwarg:
        ene_kwarg["charge"] = Xdict()
        ene_kwarg["charge"]["function"] = lambda system, ene_kwarg: CoulombEnergy(
            atom_charge=system.atom_charge, length_unit='A', energy_unit='kcal/mol',
            use_pbc=use_pbc, use_pme=use_pbc,
            nfft=system.pbc_box.asnumpy().astype(int)//4*4,
            exclude_index=np.array(sys_kwarg["exclude"]))


set_global_alternative_names()
