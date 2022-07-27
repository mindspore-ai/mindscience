"""
This **module** set the basic configuration for ff14sb
"""
from ...helper import source, Xprint, set_real_global_variable

source("....")
amber = source("...amber")

AtomType.New_From_String(
    """
    name mass    charge[e]  LJtype
    HW   1.008    0.4238      HW
    OW   16      -0.8476      OW
    """)

bond_base.BondType.New_From_String(r"""
name   k[kcal/mol·A^-2]   b[A]
OW-HW  553                1.0000
HW-HW  553                1.6330
""")

angle_base.AngleType.New_From_String(r"""
name      k   b
HW-OW-HW  0   0
OW-HW-HW  0   0
""")

lj_base.LJType.New_From_String(r"""
name    epsilon[kcal/mol]   rmin[A]
OW-OW   0.1553              1.7767
HW-HW   0                   0
""")

SPCE = load_mol2(os.path.join(AMBER_DATA_DIR, "spce.mol2"))

amber.load_parameters_from_frcmod("ions1lm_126_spce.frcmod")
amber.load_parameters_from_frcmod("ionsjc_spce.frcmod")
amber.load_parameters_from_frcmod("ions234lm_126_spce.frcmod")

load_mol2(os.path.join(AMBER_DATA_DIR, "atomic_ions.mol2"))

set_real_global_variable("WAT", SPCE)

Xprint("""Reference for spce:
1. Water:
  H. J. C. Berendsen, J. R. Grigera, and T. P. Straatsma
    The missing term in effective pair potentials
    The Journal of Physical Chemistry 1987 91 (24), 6269-6271
    DOI: 10.1021/j100308a038

2. Li+, Na+, K+, Rb+, Cs+, F-, Cl-, Br-, I-:
  In Suk Joung and Thomas E. Cheatham
    Determination of Alkali and Halide Monovalent Ion Parameters for Use in Explicitly Solvated Biomolecular Simulations
    The Journal of Physical Chemistry B 2008 112 (30), 9020-9041
    DOI: 10.1021/jp8001614

3. Ag+, Tl+, Cu+:
  Pengfei Li, Lin Frank Song, and Kenneth M. Merz
    Systematic Parameterization of Monovalent Ions Employing the Nonbonded Model
    Journal of Chemical Theory and Computation 2015 11 (4), 1645-1657, 
    DOI: 10.1021/ct500918t
    
4. Divalent Ions(Ba2+, Mg2+...)
  Pengfei Li and Kenneth M. Merz
    Taking into Account the Ion-Induced Dipole Interaction in the Nonbonded Model of Ions
    Journal of Chemical Theory and Computation 2014 10 (1), 289-297
    DOI: 10.1021/ct400751u

5. Trivalent and Tetravalent Cations(Al3+, Fe3+, Hf4+...)
  Pengfei Li, Lin Frank Song, and Kenneth M. Merz
    Parameterization of Highly Charged Metal Ions Using the 12-6-4 LJ-Type Nonbonded Model in Explicit Water
    The Journal of Physical Chemistry B 2015 119 (3), 883-895
    DOI: 10.1021/jp505875v  
""")
# pylint:disable=undefined-variable
