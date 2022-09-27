"""
This **package** sets the tips3p configuration of charmm27 force field
"""
from ...helper import source, Xprint, set_real_global_variable

source("....")
source("...charmm27")

AtomType.New_From_String(
    """
    name mass    charge[e]  LJtype
    HW   1.008    0.417       HW
    OW   16      -0.834       OW
    """)

bond_base.BondType.New_From_String(r"""
name   k[kcal/mol·A^-2]   b[A]
OW-HW  553                0.9572
HW-HW  553                1.5136
""")

ub_angle_base.UreyBradleyType.New_From_String(r"""
name      k   kUB  
HW-OW-HW  0   0
OW-HW-HW  0   0
""")

lj_base.LJType.New_From_String(r"""
name    sigma[nm]   epsilon[kJ/mol]
OW-OW   0.315057422683    0.6363864
HW-HW   0.0400013524445 0.192464
""")

TIPS3P = load_mol2(os.path.join(CHARMM27_DATA_DIR, "tip3p.mol2"), as_template=True)

load_mol2(os.path.join(CHARMM27_DATA_DIR, "atomic_ions.mol2"), as_template=True)

set_real_global_variable("WAT", TIPS3P)

Xprint("""Reference for CHARMM modified tip3p:
  to do
""")
# pylint:disable=undefined-variable
