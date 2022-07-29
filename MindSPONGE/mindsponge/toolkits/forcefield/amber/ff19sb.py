"""
This **module** set the basic configuration for ff19sb
"""
from ...helper import source, Xprint, set_real_global_variable
from ..base import residue_cmap_base

source("....")
amber = source("...amber")

residue_cmap_base.CMapType.New_From_String(r"""
name
C-N-XC-C-N
""")

amber.load_parameters_from_parmdat("parm19.dat")
amber.load_parameters_from_frcmod("ff19SB.frcmod", include_cmap=True)

load_mol2(os.path.join(AMBER_DATA_DIR, "ff19SB.mol2"))

ResidueType.set_type("HIS", ResidueType.get_type("HIE"))
ResidueType.set_type("NHIS", ResidueType.get_type("NHIE"))
ResidueType.set_type("CHIS", ResidueType.get_type("CHIE"))

set_real_global_variable("HIS", ResidueType.get_type("HIS"))
set_real_global_variable("NHIS", ResidueType.get_type("NHIS"))
set_real_global_variable("CHIS", ResidueType.get_type("CHIS"))

residues = "ALA ARG ASN ASP CYS CYX GLN GLU GLY HID HIE HIP ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL HIS".split()

for res in residues:
    ResidueType.get_type(res).head_next = "CA"
    ResidueType.get_type(res).head_length = 1.3
    ResidueType.get_type(res).tail_next = "CA"
    ResidueType.get_type(res).tail_length = 1.3
    ResidueType.get_type(res).head_link_conditions.append({"atoms": ["CA", "N"], "parameter": 120 / 180 * np.pi})
    ResidueType.get_type(res).head_link_conditions.append({"atoms": ["H", "CA", "N"], "parameter": -np.pi})
    ResidueType.get_type(res).tail_link_conditions.append({"atoms": ["CA", "C"], "parameter": 120 / 180 * np.pi})
    ResidueType.get_type(res).tail_link_conditions.append({"atoms": ["O", "CA", "C"], "parameter": -np.pi})

    ResidueType.get_type("N" + res).tail_next = "CA"
    ResidueType.get_type("N" + res).tail_length = 1.3
    ResidueType.get_type("N" + res).tail_link_conditions.append({"atoms": ["CA", "C"], "parameter": 120 / 180 * np.pi})
    ResidueType.get_type("N" + res).tail_link_conditions.append({"atoms": ["O", "CA", "C"], "parameter": -np.pi})

    ResidueType.get_type("C" + res).head_next = "CA"
    ResidueType.get_type("C" + res).head_length = 1.3
    ResidueType.get_type("C" + res).head_link_conditions.append({"atoms": ["CA", "N"], "parameter": 120 / 180 * np.pi})
    ResidueType.get_type("C" + res).head_link_conditions.append({"atoms": ["H", "CA", "N"], "parameter": -np.pi})

    GlobalSetting.Add_PDB_Residue_Name_Mapping("head", res, "N" + res)
    GlobalSetting.Add_PDB_Residue_Name_Mapping("tail", res, "C" + res)

ResidueType.get_type("ACE").tail_next = "CH3"
ResidueType.get_type("ACE").tail_length = 1.3
ResidueType.get_type("ACE").tail_link_conditions.append({"atoms": ["CH3", "C"], "parameter": 120 / 180 * np.pi})
ResidueType.get_type("ACE").tail_link_conditions.append({"atoms": ["O", "CH3", "C"], "parameter": -np.pi})

ResidueType.get_type("NME").head_next = "CH3"
ResidueType.get_type("NME").head_length = 1.3
ResidueType.get_type("NME").head_link_conditions.append({"atoms": ["CH3", "N"], "parameter": 120 / 180 * np.pi})
ResidueType.get_type("NME").head_link_conditions.append({"atoms": ["H", "CH3", "N"], "parameter": -np.pi})

GlobalSetting.HISMap["DeltaH"] = "HD1"
GlobalSetting.HISMap["EpsilonH"] = "HE2"
GlobalSetting.HISMap["HIS"].update({"HIS": {"HID": "HID", "HIE": "HIE", "HIP": "HIP"},
                                    "CHIS": {"HID": "CHID", "HIE": "CHIE", "HIP": "CHIP"},
                                    "NHIS": {"HID": "NHID", "HIE": "NHIE", "HIP": "NHIP"}})

ResidueType.get_type("CYX").connect_atoms["ssbond"] = "SG"

Xprint("""Reference for ff19SB:
  Chuan Tian, Koushik Kasavajhala, Kellon A. A. Belfon, Lauren Raguette, He Huang, Angela N. Migues, John Bickel, Yuzhang Wang, Jorge Pincay, Qin Wu, and Carlos Simmerling
    ff19SB: Amino-Acid-Specific Protein Backbone Parameters Trained against Quantum Mechanics Energy Surfaces in Solution
    The Journal of Chemical Physics 2020 16 (1), 528-552, 
    DOI: 10.1021/acs.jctc.9b00591
""")
# pylint:disable=undefined-variable
