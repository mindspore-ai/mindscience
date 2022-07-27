"""
This **module** set the basic configuration for ff14sb
"""
from ...helper import source, Xprint, set_real_global_variable

source("....")
amber = source("...amber")

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

angle_base.AngleType.New_From_String(r"""
name      k   b
HW-OW-HW  0   0
OW-HW-HW  0   0
""")

lj_base.LJType.New_From_String(r"""
name    epsilon[kcal/mol]   rmin[A]
OW-OW   0.152               1.7683
HW-HW   0                   0
""")

TIP3P = load_mol2(os.path.join(AMBER_DATA_DIR, "tip3p.mol2"))

amber.load_parameters_from_frcmod("ions1lm_126_tip3p.frcmod")
amber.load_parameters_from_frcmod("ionsjc_tip3p.frcmod")
amber.load_parameters_from_frcmod("ions234lm_126_tip3p.frcmod")

load_mol2(os.path.join(AMBER_DATA_DIR, "atomic_ions.mol2"))

set_real_global_variable("WAT", TIP3P)

Xprint("""Reference for tip3p:
1. Water:
  William L. Jorgensen, Jayaraman Chandrasekhar, and Jeffry D. Madura
    Comparison of simple potential functions for simulating liquid water
    The Journal of Chemical Physics 1983 79, 926-935, 
    DOI: 10.1063/1.445869

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
