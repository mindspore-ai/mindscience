"""
This **module** is the basic setting for the force field format of harmonic angle
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule

# pylint: disable=invalid-name
AngleType = Generate_New_Bonded_Force_Type("angle", "1-2-3", {"k": float, "b": float}, True)

AngleType.Set_Property_Unit("k", "energy·angle^-2", "kcal/mol·rad^-2")
AngleType.Set_Property_Unit("b", "angle", "rad")


@Molecule.Set_Save_SPONGE_Input("angle")
def write_angle(self):
    """
    This **function** is used to write SPONGE input file
    :param self:
    :return:
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
