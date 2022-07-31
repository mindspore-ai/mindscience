"""
This **module** is the basic setting for the force field format of Urey Bradley angle
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, set_global_alternative_names

# pylint: disable=invalid-name
UreyBradleyType = Generate_New_Bonded_Force_Type("Urey_Bradley", "1-2-3",
                                                 {"k": float, "b": float, "kUB": float, "r13": float}, True)

UreyBradleyType.Set_Property_Unit("k", "energy·angle^-2", "kcal/mol·rad^-2")
UreyBradleyType.Set_Property_Unit("b", "angle", "rad")
UreyBradleyType.Set_Property_Unit("kUB", "energy·distance^-2", "kcal/mol·A^-2")
UreyBradleyType.Set_Property_Unit("r13", "distance", "A")


@Molecule.Set_Save_SPONGE_Input("urey_bradley")
def write_angle(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    angles = []
    for angle in self.bonded_forces.get("Urey_Bradley", []):
        order = list(range(3))
        if angle.k != 0 or angle.kUB != 0:
            if self.atom_index[angle.atoms[order[0]]] > self.atom_index[angle.atoms[order[-1]]]:
                temp_order = order[::-1]
            else:
                temp_order = order
            angles.append("%d %d %d %f %f %f %f" % (self.atom_index[angle.atoms[temp_order[0]]]
                                                    , self.atom_index[angle.atoms[temp_order[1]]],
                                                    self.atom_index[angle.atoms[temp_order[2]]], angle.k, angle.b,
                                                    angle.kUB, angle.r13))

    if angles:
        towrite = "%d\n" % len(angles)
        angles.sort(key=lambda x: list(map(int, x.split()[:3])))
        towrite += "\n".join(angles)
        return towrite
    return None

set_global_alternative_names()
