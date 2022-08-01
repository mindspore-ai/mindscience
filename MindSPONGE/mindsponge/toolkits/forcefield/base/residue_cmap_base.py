"""
This **module** is the basic setting for the force field format of atom-specific cmap
"""
from ... import Generate_New_Bonded_Force_Type
from ...helper import Molecule, Xdict, set_global_alternative_names

# pylint: disable=invalid-name
CMapType = Generate_New_Bonded_Force_Type("residue_specific_cmap", "1-2-3-4-5", {}, False)

CMapType.Residue_Map = {}


@Molecule.Set_Save_SPONGE_Input("cmap")
def write_cmap(self):
    """
    This **function** is used to write SPONGE input file

    :param self: the Molecule instance
    :return: the string to write
    """
    cmaps = []
    resolutions = []
    used_types = []
    used_types_map = Xdict()
    cmap = None
    for cmap in self.bonded_forces.get("residue_specific_cmap", []):
        resname = cmap.atoms[2].residue.type.name
        if resname in CMapType.Residue_Map.keys():
            if CMapType.Residue_Map[resname]["count"] not in used_types_map.keys():
                used_types_map[CMapType.Residue_Map[resname]["count"]] = len(used_types)
                used_types.append(CMapType.Residue_Map[resname]["parameters"])
                resolutions.append(str(CMapType.Residue_Map[resname]["resolution"]))
            cmaps.append("%d %d %d %d %d %d" % (self.atom_index[cmap.atoms[0]], self.atom_index[cmap.atoms[1]],
                                                self.atom_index[cmap.atoms[2]], self.atom_index[cmap.atoms[3]],
                                                self.atom_index[cmap.atoms[4]],
                                                used_types_map[CMapType.Residue_Map[resname]["count"]]))

    if cmap:
        towrite = "%d %d\n" % (len(cmaps), len(resolutions))
        towrite += " ".join(resolutions) + "\n\n"

        for i in range(len(used_types_map)):
            resol = int(resolutions[i])
            for j in range(resol):
                for k in range(resol):
                    towrite += "%f " % used_types[i][j * 24 + k]
                towrite += "\n"
            towrite += "\n"

        towrite += "\n".join(cmaps)

        return towrite
    return None

set_global_alternative_names()
