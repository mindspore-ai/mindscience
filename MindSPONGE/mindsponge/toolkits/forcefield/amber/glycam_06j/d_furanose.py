"""
This **module** sets the residue types of D-furanose
"""
from itertools import product
from ....helper import source


def _init():
    """
    initialize the module
    """
    source("...glycam_06j")
    load_mol2(os.path.join(AMBER_DATA_DIR, "glycam_06j", "d_furanose.mol2"), as_template=True)
    all_types = ResidueType.get_all_types()
    for code, terminal, comformation in product("ABCDEFGHJKLMNOPQRSTUVWXYZ", "012346789PQRSTUVWXYZ", "UD"):
        resname = terminal + code + comformation
        if resname not in all_types:
            continue
        res = ResidueType.get_type(resname)
        tail_dihedral = 120 if comformation == "D" else -120
        if terminal == "1":
            res.head = "O1"
            res.head_next = "C1"
            res.head_length = 1.4
            res.head_link_conditions.append({"atoms": ["C1", "O1"], "parameter": 111 / 180 * np.pi})
            res.head_link_conditions.append({"atoms": ["O5", "C1", "O1"], "parameter": 60 / 180 * np.pi})
        elif code in "CPBJ":
            res.tail = "C2"
            res.tail_next = "C3"
            res.tail_length = 1.4
            res.tail_link_conditions.append({"atoms": ["O5", "C2"], "parameter": 110 / 180 * np.pi})
            res.tail_link_conditions.append({"atoms": ["C3", "O5", "C2"], "parameter": tail_dihedral / 180 * np.pi})
        else:
            res.tail = "C1"
            res.tail_next = "C2"
            res.tail_length = 1.4
            res.tail_link_conditions.append({"atoms": ["O4", "C1"], "parameter": 110 / 180 * np.pi})
            res.tail_link_conditions.append({"atoms": ["C2", "O4", "C1"], "parameter": tail_dihedral / 180 * np.pi})
        if terminal in "2ZYXTSP" or (terminal == "R" and code not in "VYW"):
            setHead(res, 2)
        elif terminal in "3WVQR":
            SetHead(res, 3)
        elif terminal in "4U":
            Set_Head(res, 4)
        elif terminal != "1":
            set_head(res, int(terminal))


_init()
