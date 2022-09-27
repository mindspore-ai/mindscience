"""
This **module** set the basic configuration for OL15 (DNA)
"""
from ...helper import source, Xprint

source("....")
amber = source("...amber")

amber.load_parameters_from_parmdat("parm10.dat")
amber.load_parameters_from_frcmod("parmbsc1.frcmod")

load_mol2(os.path.join(AMBER_DATA_DIR, "DNA.mol2"), as_template=True)

for i in "ATCG":
    i = "D" + i
    res = ResidueType.get_type(i)
    res5 = ResidueType.get_type(i + "5")
    res3 = ResidueType.get_type(i + "3")
    res.tail = "O3'"
    res.tail_next = "C3'"
    res5.tail = "O3'"
    res5.tail_next = "C3'"

    res.head = "P"
    res.head_next = "OP2"
    res3.head = "P"
    res3.head_next = "OP2"

    res.tail_link_conditions.append({"atoms": ["C3'", "O3'"], "parameter": 120 / 180 * np.pi})
    res.tail_link_conditions.append({"atoms": ["H3'", "C3'", "O3'"], "parameter": -54 / 180 * np.pi})
    res5.tail_link_conditions.append({"atoms": ["C3'", "O3'"], "parameter": 120 / 180 * np.pi})
    res5.tail_link_conditions.append({"atoms": ["H3'", "C3'", "O3'"], "parameter": -54 / 180 * np.pi})

    res.head_link_conditions.append({"atoms": ["OP2", "P"], "parameter": 108 / 180 * np.pi})
    res.head_link_conditions.append({"atoms": ["O5'", "OP2", "P"], "parameter": 113 / 180 * np.pi})
    res3.head_link_conditions.append({"atoms": ["OP2", "P"], "parameter": 108 / 180 * np.pi})
    res3.head_link_conditions.append({"atoms": ["O5'", "OP2", "P"], "parameter": 113 / 180 * np.pi})

    GlobalSetting.Add_PDB_Residue_Name_Mapping("head", res.name, res5.name)
    GlobalSetting.Add_PDB_Residue_Name_Mapping("tail", res.name, res3.name)

Xprint("""Reference for OL15:
  Marie Zgarbová, Jiří Šponer, Michal Otyepka, Thomas E Cheatham 3rd, Rodrigo Galindo-Murillo, and Petr Jurečka
    Refinement of the Sugar-Phosphate Backbone Torsion Beta for AMBER Force Fields Improves the Description of 
      Z- and B-DNA
    Journal of Chemical Theory and Computation 2015 11 (8), 5723-5736
    DOI: 10.1021/acs.jctc.5b00716
""")
