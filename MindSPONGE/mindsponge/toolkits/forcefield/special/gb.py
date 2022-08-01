"""
This **module** set the generalized Born force field
"""
from ...helper import source, Xprint, Guess_Element_From_Mass, set_global_alternative_names

source("....")

AtomType.Add_Property({"GB_radii": float})
AtomType.Set_Property_Unit("GB_radii", "distance", "A")
AtomType.Add_Property({"GB_scaler": float})


def bondi_radii(atom):
    """
    This **function** receives an atom and sets the Bondi radii

    :param atom: the Atom instance
    :return: None
    """
    temp_dict = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "F": 1.47,
                 "P": 1.8, "S": 1.8, "Cl": 1.75, "Br": 1.85, "I": 1.98}
    element = Guess_Element_From_Mass(atom.mass)
    atom.GB_radii = temp_dict.get(element, 1.5)
    temp_dict2 = {"H": 0.85, "C": 0.72, "N": 0.79, "O": 0.85, "F": 0.88,
                  "P": 0.86, "S": 0.96}
    atom.GB_scaler = temp_dict2.get(element, 0.8)


def modified_bondi_radii(atom):
    """
    This **function** receives an atom and sets the modified Bondi radii

    :param atom: the Atom instance
    :return: None
    """
    temp_dict = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.5, "F": 1.5,
                 "Si": 2.1, "P": 1.85, "S": 1.8, "Cl": 1.7, "Br": 1.85, "I": 1.98}
    element = Guess_Element_From_Mass(atom.mass)
    atom.GB_radii = temp_dict.get(element, 1.5)
    temp_dict2 = {"H": 0.85, "C": 0.72, "N": 0.79, "O": 0.85, "F": 0.88,
                  "P": 0.86, "S": 0.96}
    atom.GB_scaler = temp_dict2.get(element, 0.8)
    if element == "H":
        for atom_a in atom.residue.connectivity[atom]:
            element_a = Guess_Element_From_Mass(atom_a.mass)
            if element_a in ("C", "N"):
                atom.GB_radii = 1.3
            elif elementA in ('S', 'O', 'H'):
                atom.GB_radii = 0.8
            break


AVAILABLE_RADIUS_SET = {"bondi_radii": bondi_radii, "modified_bondi_radii": modified_bondi_radii}


def _show_reference(radius_set):
    """

    :param radius_set:
    :return:
    """
    # pylint: disable=comparison-with-callable
    if radius_set == bondi_radii:
        Xprint("""Reference for Bondi radii:
         A. Bondi
         van der Waals Volumes and Radii
         Journal of Physical Chemistry 1964 68 (3) 441-451
         DOI: 10.1021/j100785a001
     """)
    # pylint: disable=comparison-with-callable
    elif radius_set == modified_bondi_radii:
        Xprint("""Reference for modified Bondi radii:
    Vickie Tsui, David A. Case
    Theory and Applications of the Generalized Born Solvation Model in Macromolecular Simulations
    Biopolymers 2001 56 (4) 275-291
    DOI: 10.1002/1097-0282(2000)56:4<275::AID-BIP10024>3.0.CO;2-E
""")


def write_gb_radii_and_scaler(self):
    """
    This **function** is used to write gb radii and scaler when saving SPONGE inputs

    :param self: the Molecule instance
    :return: the string to write
    """
    towrite = "%d\n" % (len(self.atoms))
    towrite += "\n".join(["%.4f %.4f" % (atom.GB_radii, atom.GB_scaler) for atom in self.atoms])
    return towrite


def set_gb_radius(mol, radius_set=modified_bondi_radii):
    """
    This **function** is used to set GB radius for the molecule

    :param mol: the molecule, either a Molecule instance, a ResidueType instance or a Residue instance
    :param radius_set: a function, which receives an Atom instance and sets the radius
    :return: None
    """
    _show_reference(radius_set)
    if isinstance(mol, Molecule):
        for residue in mol.residues:
            for atom in residue.atoms:
                radius_set(atom)
    else:
        for atom in mol.atoms:
            radius_set(atom)
    mol.box_length = [999, 999, 999]
    Molecule.Set_Save_SPONGE_Input("gb")(write_gb_radii_and_scaler)


set_global_alternative_names()
