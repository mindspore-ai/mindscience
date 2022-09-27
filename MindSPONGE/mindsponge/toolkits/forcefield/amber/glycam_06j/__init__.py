"""
This **package** sets the basic configuration for GLYCAM-06j
"""
from ....helper import source, Xprint

source("....")
amber = source("...amber")

amber.load_parameters_from_parmdat(os.path.join("glycam_06j", "GLYCAM_06j.dat"))

load_mol2(os.path.join(AMBER_DATA_DIR, "glycam_06j", "terminal.mol2"), as_template=True)

ROH = ResidueType.get_type("ROH")
ROH.head = "O1"
ROH.head_next = "HO1"
ROH.head_length = 1.3

OME = ResidueType.get_type("OME")
OME.head = "O"
OME.head_next = "CH3"
OME.head_length = 1.3


def set_head(res, n):
    """
    This **function** sets the head of the carbohydrate residue type to the n-th oxygen atom
    :param res:
    :param n:
    :return:
    """
    head_dihedral = -60 if (n <= 6 and res.name[-1] not in "DU") else -180
    res.head = f"O{n}"
    res.head_next = f"C{n}"
    res.tail_length = 1.4
    res.head_link_conditions.clear()
    res.head_link_conditions.append({"atoms": [f"C{n}", f"O{n}"], "parameter": 111 / 180 * np.pi})
    res.head_link_conditions.append({"atoms": [f"C{n-1}", f"C{n}", f"O{n}"], "parameter": head_dihedral / 180 * np.pi})


set_global_alternative_names(set_head)

Xprint("""Reference for GLYCAM-06j:
  Kirschner, K.N., Yongye, A.B., Tschampel, S.M., González-Outeiriño, J., Daniels, C.R., Foley, B.L. and Woods
    GLYCAM06: A generalizable biomolecular force field. Carbohydrates.
    The Journal of Computational Chemistry 2008 29, 622-655, 
    DOI: 10.1002/jcc.20820
""")
