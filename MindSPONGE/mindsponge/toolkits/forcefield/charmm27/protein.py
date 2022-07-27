"""
This **package** sets the protein configuration of charmm27 force field
"""
from ...helper import remove_real_global_variable, source, Xprint

source("....")
source("...charmm27")

load_mol2(os.path.join(CHARMM27_DATA_DIR, "protein.mol2"))

ResidueType.types.pop("HYP")
ResidueType.types.pop("CHYP")
ResidueType.types.pop("NHE")

remove_real_global_variable("HYP")
remove_real_global_variable("CHYP")
remove_real_global_variable("NHE")

residues = "ALA ARG ASN ASP CYS CYX GLN GLU GLY HID HIE HIP ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()

for res in residues:
    ResidueType.types[res].head_next = "CA"
    ResidueType.types[res].head_length = 1.3
    ResidueType.types[res].tail_next = "CA"
    ResidueType.types[res].tail_length = 1.3
    ResidueType.types[res].head_link_conditions.append({"atoms": ["CA", "N"], "parameter": 120 / 180 * np.pi})
    ResidueType.types[res].head_link_conditions.append({"atoms": ["H", "CA", "N"], "parameter": -np.pi})
    ResidueType.types[res].tail_link_conditions.append({"atoms": ["CA", "C"], "parameter": 120 / 180 * np.pi})
    ResidueType.types[res].tail_link_conditions.append({"atoms": ["O", "CA", "C"], "parameter": -np.pi})

    ResidueType.types["N" + res].tail_next = "CA"
    ResidueType.types["N" + res].tail_length = 1.3
    ResidueType.types["N" + res].tail_link_conditions.append({"atoms": ["CA", "C"], "parameter": 120 / 180 * np.pi})
    ResidueType.types["N" + res].tail_link_conditions.append({"atoms": ["O", "CA", "C"], "parameter": -np.pi})

    ResidueType.types["C" + res].head_next = "CA"
    ResidueType.types["C" + res].head_length = 1.3
    ResidueType.types["C" + res].head_link_conditions.append({"atoms": ["CA", "N"], "parameter": 120 / 180 * np.pi})
    ResidueType.types["C" + res].head_link_conditions.append({"atoms": ["H", "CA", "N"], "parameter": -np.pi})

    GlobalSetting.Add_PDB_Residue_Name_Mapping("head", res, "N" + res)
    GlobalSetting.Add_PDB_Residue_Name_Mapping("tail", res, "C" + res)

ResidueType.types["ACE"].tail_next = "CH3"
ResidueType.types["ACE"].tail_length = 1.3
ResidueType.types["ACE"].tail_link_conditions.append({"atoms": ["CH3", "C"], "parameter": 120 / 180 * np.pi})
ResidueType.types["ACE"].tail_link_conditions.append({"atoms": ["O", "CH3", "C"], "parameter": -np.pi})

ResidueType.types["NME"].head_next = "CH3"
ResidueType.types["NME"].head_length = 1.3
ResidueType.types["NME"].head_link_conditions.append({"atoms": ["CH3", "N"], "parameter": 120 / 180 * np.pi})
ResidueType.types["NME"].head_link_conditions.append({"atoms": ["H", "CH3", "N"], "parameter": -np.pi})

GlobalSetting.HISMap["DeltaH"] = "HD1"
GlobalSetting.HISMap["EpsilonH"] = "HE2"
GlobalSetting.HISMap["HIS"].update({"HIS": {"HID": "HID", "HIE": "HIE", "HIP": "HIP"},
                                    "CHIS": {"HID": "CHID", "HIE": "CHIE", "HIP": "CHIP"},
                                    "NHIS": {"HID": "NHID", "HIE": "NHIE", "HIP": "NHIP"}})

ResidueType.types["CYX"].connect_atoms["ssbond"] = "SG"

Xprint("""Reference for protein of charmm27:
    MacKerell, Jr., A. D., Feig, M., Brooks, C.L., III, Extending the
    treatment of backbone energetics in protein force fields: limitations
    of gas-phase quantum mechanics in reproducing protein conformational
    distributions in molecular dynamics simulations, Journal of
    Computational Chemistry, 25: 1400-1415, 2004.

and 

    MacKerell, Jr., A. D.,  et al. All-atom
    empirical potential for molecular modeling and dynamics Studies of
    proteins.  Journal of Physical Chemistry B, 1998, 102, 3586-3616.

""")
# pylint:disable=undefined-variable
