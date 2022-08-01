"""
This **package** sets the basic configuration of amber force field
"""
import os
from ...helper import set_global_alternative_names
from ... import AtomType, load_parmdat, load_frcmod

from ..base import charge_base, mass_base, lj_base, bond_base, angle_base, dihedral_base, nb14_base,\
    virtual_atom_base, exclude_base

AMBER_DATA_DIR = os.path.dirname(__file__)

lj_base.LJType.combining_method_A = lj_base.Lorentz_Berthelot_For_A
lj_base.LJType.combining_method_B = lj_base.Lorentz_Berthelot_For_B

nb14_base.NB14Type.New_From_String(r"""
name    kLJ     kee
X-X     0.5     0.833333
""")

exclude_base.Exclude(4)


def load_parameters_from_parmdat(filename, prefix=True):
    """
    This **function** is used to get amber force field parameters from parmdat files

    :param filename: the name of the input file
    :param prefix: whether add the AMBER_DATA_DIR to the filename
    :return:
    """
    if prefix:
        filename = os.path.join(AMBER_DATA_DIR, filename)
    atoms, bonds, angles, propers, impropers, ljs = load_parmdat(filename)
    AtomType.New_From_String(atoms)
    bond_base.BondType.New_From_String(bonds)
    angle_base.AngleType.New_From_String(angles)
    dihedral_base.ProperType.New_From_String(propers)
    dihedral_base.ImproperType.New_From_String(impropers)
    lj_base.LJType.New_From_String(ljs)


def load_parameters_from_frcmod(filename, include_cmap=False, prefix=True):
    """
    This **function** is used to get amber force field parameters from frcmod files

    :param filename: the name of the input file
    :param include_cmap: whether include cmap
    :param prefix: whether add the AMBER_DATA_DIR to the filename
    :return: None
    """
    if prefix:
        filename = os.path.join(AMBER_DATA_DIR, filename)
    atoms, bonds, angles, propers, impropers, ljs, cmap = load_frcmod(filename)

    AtomType.New_From_String(atoms)
    bond_base.BondType.New_From_String(bonds)
    angle_base.AngleType.New_From_String(angles)
    dihedral_base.ProperType.New_From_String(propers)
    dihedral_base.ImproperType.New_From_String(impropers)
    lj_base.LJType.New_From_String(ljs)

    if include_cmap:
        from ..base import residue_cmap_base
        residue_cmap_base.CMapType.Residue_Map.update(cmap)


set_global_alternative_names()
