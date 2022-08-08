"""
This **module** is the basic setting for the force field format of harmonic angle
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, set_global_alternative_names, Xdict

# pylint: disable=invalid-name
AngleType = Generate_New_Bonded_Force_Type("angle", "1-2-3", {"k": float, "b": float}, True)

AngleType.Set_Property_Unit("k", "energy·angle^-2", "kcal/mol·rad^-2")
AngleType.Set_Property_Unit("b", "angle", "rad")


@Molecule.Set_Save_SPONGE_Input("angle")
def write_angle(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    angles = []
    for angle in self.bonded_forces.get("angle", []):
        order = list(range(3))
        if angle.k != 0:
            if self.atom_index[angle.atoms[order[0]]] > self.atom_index[angle.atoms[order[-1]]]:
                temp_order = order[::-1]
            else:
                temp_order = order
            angles.append("%d %d %d %f %f" % (self.atom_index[angle.atoms[temp_order[0]]]
                                              , self.atom_index[angle.atoms[temp_order[1]]],
                                              self.atom_index[angle.atoms[temp_order[2]]], angle.k, angle.b))

    if angles:
        towrite = "%d\n" % len(angles)
        angles.sort(key=lambda x: list(map(int, x.split()[:3])))
        towrite += "\n".join(angles)
        return towrite

    return None


#pylint: disable=unused-argument
@Molecule.Set_MindSponge_Todo("angle")
def _do(self, sys_kwarg, ene_kwarg, use_pbc):
    """

    :return:
    """
    from mindsponge.potential import AngleEnergy
    if "angle" not in ene_kwarg:
        ene_kwarg["angle"] = Xdict()
        ene_kwarg["angle"]["function"] = lambda system, ene_kwarg: AngleEnergy(
            index=ene_kwarg["angle"]["index"], use_pbc=use_pbc,
            force_constant=ene_kwarg["angle"]["force_constant"],
            bond_angle=ene_kwarg["angle"]["bond_angle"],
            energy_unit="kcal/mol")
        ene_kwarg["angle"]["index"] = []
        ene_kwarg["angle"]["force_constant"] = []
        ene_kwarg["angle"]["bond_angle"] = []
    bonds = []
    force_constants = []
    bond_angles = []
    for bond in self.bonded_forces.get("angle", []):
        if bond.k == 0:
            continue
        bonds.append([self.atom_index[atom] for atom in bond.atoms])
        force_constants.append(bond.k * 2)
        bond_angles.append(bond.b)
    ene_kwarg["angle"]["index"].append(bonds)
    ene_kwarg["angle"]["force_constant"].append(force_constants)
    ene_kwarg["angle"]["bond_angle"].append(bond_angles)

set_global_alternative_names()
