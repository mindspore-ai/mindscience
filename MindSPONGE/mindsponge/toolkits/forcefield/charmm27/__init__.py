"""
This **package** sets the basic configuration of charmm27 force field
"""
import os
from ... import GlobalSetting, load_ffitp, AtomType, ResidueType, set_global_alternative_names
from ..base import charge_base, mass_base, lj_base, bond_base, ub_angle_base, \
    dihedral_base, nb14_base, nb14_extra_base, improper_base, \
    virtual_atom_base, atom_cmap_base, exclude_base

CHARMM27_DATA_DIR = os.path.dirname(__file__)

lj_base.LJType.combining_method_A = lj_base.Lorentz_Berthelot_For_A
lj_base.LJType.combining_method_B = lj_base.Lorentz_Berthelot_For_B

GlobalSetting.Set_Invisible_Bonded_Forces(["improper"])

dihedral_base.ProperType.New_From_String(r"""
name        k reset  phi0 periodicity
X-X-X-X     0 0      0    0
""")
exclude_base.Exclude(4)


def load_parameter_from_ffitp(filename, prefix=True):
    """
    This **function** is used to get amber force field parameters from GROMACS ffitp

    :param filename: the name of the input file
    :param prefix: whether add the CHARMM27_DATA_DIR to the filename
    :return: None
    """
    if prefix:
        filename = os.path.join(CHARMM27_DATA_DIR, filename)
    output = load_ffitp(filename)

    AtomType.New_From_String(output["atomtypes"])
    bond_base.BondType.New_From_String(output["bonds"])
    dihedral_base.ProperType.New_From_String(output["dihedrals"])
    lj_base.LJType.New_From_String(output["LJ"])
    ub_angle_base.UreyBradleyType.New_From_String(output["Urey-Bradley"])
    improper_base.ImproperType.New_From_String(output["impropers"])
    nb14_extra_base.NB14Type.New_From_String(output["nb14_extra"])
    nb14_base.NB14Type.New_From_String(output["nb14"])
    atom_cmap_base.CMapType.New_From_Dict(output["cmaps"])


load_parameter_from_ffitp("forcefield.itp")

set_global_alternative_names()
