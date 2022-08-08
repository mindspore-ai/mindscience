"""
This **module** is the basic setting for the force field format of harmonic bond
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, set_global_alternative_names, Xdict

# pylint: disable=invalid-name
BondType = Generate_New_Bonded_Force_Type("bond", "1-2", {"k": float, "b": float}, True)

BondType.Set_Property_Unit("k", "energy·distance^-2", "kcal/mol·A^-2")
BondType.Set_Property_Unit("b", "distance", "A")


@Molecule.Set_Save_SPONGE_Input("bond")
def write_bond(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    bonds = []
    for bond in self.bonded_forces.get("bond", []):
        order = list(range(2))
        if bond.k != 0:
            if self.atom_index[bond.atoms[order[0]]] > self.atom_index[bond.atoms[order[-1]]]:
                temp_order = order[::-1]
            else:
                temp_order = order
            bonds.append("%d %d %f %f" % (self.atom_index[bond.atoms[temp_order[0]]]
                                          , self.atom_index[bond.atoms[temp_order[1]]], bond.k, bond.b))

    if bonds:
        towrite = "%d\n" % len(bonds)
        bonds.sort(key=lambda x: list(map(int, x.split()[:2])))
        towrite += "\n".join(bonds)

        return towrite
    return None


@Molecule.Set_MindSponge_Todo("bond")
def _do(self, sys_kwarg, ene_kwarg, use_pbc):
    """

    :return:
    """
    from mindsponge.potential import BondEnergy
    if "bond" not in sys_kwarg:
        sys_kwarg["bond"] = []
    if "bond" not in ene_kwarg:
        ene_kwarg["bond"] = Xdict()
        ene_kwarg["bond"]["function"] = lambda system, ene_kwarg: BondEnergy(
            index=system.bond, use_pbc=use_pbc,
            force_constant=ene_kwarg["bond"]["force_constant"],
            bond_length=ene_kwarg["bond"]["bond_length"],
            length_unit="A", energy_unit="kcal/mol")
        ene_kwarg["bond"]["force_constant"] = []
        ene_kwarg["bond"]["bond_length"] = []
    bonds = []
    force_constants = []
    bond_lengths = []
    for bond in self.bonded_forces.get("bond", []):
        if bond.k == 0:
            continue
        bonds.append([self.atom_index[bond.atoms[0]], self.atom_index[bond.atoms[1]]])
        force_constants.append(bond.k * 2)
        bond_lengths.append(bond.b)
    sys_kwarg["bond"].append(bonds)
    ene_kwarg["bond"]["force_constant"].append(force_constants)
    ene_kwarg["bond"]["bond_length"].append(bond_lengths)


set_global_alternative_names()
