"""
This **module** set the basic configuration for lipid17
"""
from ...helper import source, Xprint

source("....")
amber = source("...amber")
amber.load_parameters_from_parmdat("lipid17.dat")
load_mol2(os.path.join(AMBER_DATA_DIR, "lipid17.mol2"), as_template=True)

for res in "LAL PA MY OL ST AR DHA".split():
    ResidueType.get_type(res).head = "C12"
    ResidueType.get_type(res).tail = "C12"
    ResidueType.get_type(res).head_next = "C13"
    ResidueType.get_type(res).tail_next = "C13"
    ResidueType.get_type(res).head_length = 1.5
    ResidueType.get_type(res).tail_length = 1.5
    ResidueType.get_type(res).head_link_conditions.append({"atoms": ["H2R", "C12"], "parameter": 109.5 / 180 * np.pi})
    ResidueType.get_type(res).head_link_conditions.append(
        {"atoms": ["H2S", "H2R", "C12"], "parameter": -120 / 180 * np.pi})
    ResidueType.get_type(res).tail_link_conditions.append({"atoms": ["H2R", "C12"], "parameter": 109.5 / 180 * np.pi})
    ResidueType.get_type(res).tail_link_conditions.append(
        {"atoms": ["H2S", "H2R", "C12"], "parameter": -120 / 180 * np.pi})

for res in "PC PE PS PGR PH-".split():
    ResidueType.get_type(res).head = "C11"
    ResidueType.get_type(res).tail = "C21"
    ResidueType.get_type(res).head_next = "O11"
    ResidueType.get_type(res).tail_next = "O21"
    ResidueType.get_type(res).head_length = 1.5
    ResidueType.get_type(res).tail_length = 1.5
    ResidueType.get_type(res).head_link_conditions.append({"atoms": ["O11", "C11"], "parameter": 120 / 180 * np.pi})
    ResidueType.get_type(res).head_link_conditions.append({"atoms": ["O12", "O11", "C11"], "parameter": np.pi})
    ResidueType.get_type(res).tail_link_conditions.append({"atoms": ["O21", "C21"], "parameter": 120 / 180 * np.pi})
    ResidueType.get_type(res).tail_link_conditions.append({"atoms": ["O22", "O21", "C21"], "parameter": np.pi})

Xprint("""Reference for lipid17:
  Gould, I.R., Skjevik A.A., Dickson, C.J., Madej, B.D., Walker, R.C.
    Lipid17: A Comprehensive amber Force Field for the Simulation of Zwitterionic and Anionic Lipids
    2018, in prep.
    
  Dickson, C.J., Madej, B.D., Skjevik, A.A., Betz, R.M., Teigen, K., Gould, I.R., Walker, R.C. 
    Lipid14: The Amber Lipid Force Field.
    Journal of Chemical Theory and Computation 2014 10(2), 865-879,
    DOI: 10.1021/ct4010307
""")
# pylint:disable=undefined-variable
