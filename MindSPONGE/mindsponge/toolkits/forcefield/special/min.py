"""
This **module** is used to save parameters for special minimization
"""
from ...helper import source, set_global_alternative_names

source("....")


def write_zero_mass_for_hydrogen(self):
    """
    This **function** sets the mass of hydrogen and no lj atoms to zero, to freeze them

    :param self: the Molecule instance
    :return: the string to write
    """
    towrite = "%d\n" % (len(self.atoms))
    towrite += "\n".join(
        ["%.3f" % (1) if (atom.mass < 3.999 or atom.LJtype == "ZERO_LJ_ATOM" or
                          atom.bad_coordinate) else "%.3f" % (0) for atom in self.atoms])
    return towrite


def write_zero_lj(self):
    """
    This **function** sets all the lj parameters to zero

    :param self: the Molecule instance
    :return: the string to write
    """
    towrite = "%d %d\n\n" % (len(self.atoms), 1)
    for _ in range(1):
        for _ in range(1):
            towrite += "%16.7e" % 0 + " "
        towrite += "\n"
    towrite += "\n"

    for _ in range(1):
        for _ in range(1):
            towrite += "%16.7e" % 0 + " "
        towrite += "\n"
    towrite += "\n"
    towrite += "\n".join(["%d" % (0) for atom in self.atoms])
    return towrite


def write_zero_charge(self):
    """
    This **function** sets all the charge parameters to zero

    :param self: the Molecule instance
    :return: the string to write
    """
    towrite = "%d\n" % (len(self.atoms))
    towrite += "\n".join(["%.6f" % (0) for _ in self.atoms])
    return towrite


def save_min_bonded_parameters():
    """
    This **function** saves parameters to only minimize bonded force when saving SPONGE inputs

    :return: None
    """
    Molecule.Set_Save_SPONGE_Input("fake_mass")(write_zero_mass_for_hydrogen)
    Molecule.Set_Save_SPONGE_Input("fake_LJ")(write_zero_lj)
    Molecule.Set_Save_SPONGE_Input("fake_charge")(write_zero_charge)


def do_not_save_min_bonded_parameters():
    """
    This **function** does not save parameters to only minimize bonded force when saving SPONGE inputs

    :return: None
    """
    Molecule.Del_Save_SPONGE_Input("fake_mass")
    Molecule.Del_Save_SPONGE_Input("fake_LJ")
    Molecule.Del_Save_SPONGE_Input("fake_charge")


set_global_alternative_names()
