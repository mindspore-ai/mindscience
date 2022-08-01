"""
This **module** is used to provide basic common math functions
"""
import numpy as np
from .namespace import  set_global_alternative_names


def get_rotate_matrix(r0, angle):
    """
    This **function** is used to get the rotation matrix

    :param r0: the axis to rotate around
    :param angle: the angle to rotate
    :return: the rotation matrix
    """
    cost = np.cos(angle)
    cost_one = 1 - cost
    sint = np.sin(angle)
    r0 /= np.linalg.norm(r0)
    return np.array([[r0[0] * r0[0] * cost_one + cost, r0[0] * r0[1] * cost_one - r0[2] * sint,
                      r0[0] * r0[2] * cost_one + r0[1] * sint],
                     [r0[0] * r0[1] * cost_one + r0[2] * sint, r0[1] * r0[1] * cost_one + cost,
                      r0[1] * r0[2] * cost_one - r0[0] * sint],
                     [r0[0] * r0[2] * cost_one - r0[1] * sint, r0[1] * r0[2] * cost_one + r0[0] * sint,
                      r0[2] * r0[2] * cost_one + cost]]).transpose()


def get_fibonacci_grid(n, origin, radius):
    """
    This **function** is used to get the sphere grid by fibonacci algorithm

    :param n: the number of grids
    :param origin: the origin of the sphere
    :param radius: the radius of the sphere
    :return: a numpy array of grid coordinates
    """
    n_ = np.arange(1, n + 1)
    factorn = (np.sqrt(5) - 1) * np.pi * n_
    out = np.zeros((n, 3))
    out[:, 2] = (2 * n_ - 1) / n - 1
    sqrtz = np.sqrt(1 - out[:, 2] * out[:, 2])
    out[:, 0] = sqrtz * np.cos(factorn)
    out[:, 1] = sqrtz * np.sin(factorn)
    out *= radius
    out += origin
    return out


def guess_element_from_mass(mass):
    """
    This **function** is used to guess element from its mass

    :param mass: an int or a float, the mass
    :return: the element name
    """
    elements = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc",
                "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
                "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
                "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
                "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
                "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
                "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
                "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
                "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
                "Ds", "Rg"]
    masses = [0.00000, 1.00794, 4.00260, 6.941, 9.012182, 10.811,
              12.0107, 14.0067, 15.9994, 18.9984032, 20.1797,
              22.989770, 24.3050, 26.981538, 28.0855, 30.973761,
              32.065, 35.453, 39.948, 39.0983, 40.078, 44.955910,
              47.867, 50.9415, 51.9961, 54.938049, 55.845, 58.9332,
              58.6934, 63.546, 65.409, 69.723, 72.64, 74.92160,
              78.96, 79.904, 83.798, 85.4678, 87.62, 88.90585,
              91.224, 92.90638, 95.94, 98.0, 101.07, 102.90550,
              106.42, 107.8682, 112.411, 114.818, 118.710, 121.760,
              127.60, 126.90447, 131.293, 132.90545, 137.327,
              138.9055, 140.116, 140.90765, 144.24, 145.0, 150.36,
              151.964, 157.25, 158.92534, 162.500, 164.93032,
              167.259, 168.93421, 173.04, 174.967, 178.49, 180.9479,
              183.84, 186.207, 190.23, 192.217, 195.078, 196.96655,
              200.59, 204.3833, 207.2, 208.98038, 209.0, 210.0, 222.0,
              223.0, 226.0, 227.0, 232.0381, 231.03588, 238.02891,
              237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0,
              258.0, 259.0, 262.0, 261.0, 262.0, 266.0, 264.0, 269.0,
              268.0, 271.0, 272.0]
    if 3.8 > mass > 0.0:
        index = 1
    elif 208.99 > mass > 207.85:
        index = 83
    elif  58.8133 > mass > 56.50:
        index = 27
    else:
        index = 0
        for j in range(0, 111):
            if abs(mass - masses[j]) < 0.65:
                index = j
                break
    return elements[index]

set_global_alternative_names()
