"""
This **module** sets the residue types of glycoprotein
"""

from ....helper import source

source("...glycam_06j")

load_mol2(os.path.join(AMBER_DATA_DIR, "glycam_06j", "glycoprotein.mol2"), as_template=True)

for resname in ["OLP", "OLT", "NLN", "OLS"]:
    res = ResidueType.get_type(resname)
    nres = ResidueType.get_type("N" + resname)
    cres = ResidueType.get_type("C" + resname)
    res.head, cres.head = "N", "N"
    res.head_length, cres.head_length = 1.3, 1.3
    res.head_next, cres.head_next = "CA", "CA"
    res.tail, nres.tail = "C", "C"
    res.tail_next, nres.tail_next = "CA", "CA"
    res.tail_length, nres.tail_length = 1.3, 1.3

    res.head_link_conditions.append({"atoms": ["CA", "N"], "parameter": 120 / 180 * np.pi})
    cres.head_link_conditions.append({"atoms": ["CA", "N"], "parameter": 120 / 180 * np.pi})

    if resname != "OLP":
        res.head_link_conditions.append({"atoms": ["H", "CA", "N"], "parameter": -np.pi})
        cres.head_link_conditions.append({"atoms": ["H", "CA", "N"], "parameter": -np.pi})
    else:
        res.head_link_conditions.append({"atoms": ["HA", "CA", "N"], "parameter": 0})
        cres.head_link_conditions.append({"atoms": ["HA", "CA", "N"], "parameter": 0})

    res.tail_link_conditions.append({"atoms": ["CA", "C"], "parameter": 120 / 180 * np.pi})
    res.tail_link_conditions.append({"atoms": ["O", "CA", "C"], "parameter": -np.pi})
    nres.tail_link_conditions.append({"atoms": ["CA", "C"], "parameter": 120 / 180 * np.pi})
    nres.tail_link_conditions.append({"atoms": ["O", "CA", "C"], "parameter": -np.pi})

    GlobalSetting.Add_PDB_Residue_Name_Mapping("head", resname, "N" + resname)
    GlobalSetting.Add_PDB_Residue_Name_Mapping("tail", resname, "C" + resname)
