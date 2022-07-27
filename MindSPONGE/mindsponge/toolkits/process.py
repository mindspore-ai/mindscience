"""
This **module** is used to process topology and conformations
"""
import numpy as np
from .helper import get_rotate_matrix, ResidueType, Molecule, Residue


def impose_bond(molecule, atom1, atom2, length):
    """

    :param molecule:
    :param atom1:
    :param atom2:
    :param length:
    :return:
    """
    crd = molecule.get_atom_coordinates()
    _, atom2_friends = molecule.divide_into_two_parts(atom1, atom2)
    r0 = crd[molecule.atom_index[atom2]] - crd[molecule.atom_index[atom1]]
    l0 = np.linalg.norm(r0)
    if l0 == 0:
        crd[molecule.atom_index[atom2]] += (1 / 3) ** (0.5)
        r0 = crd[molecule.atom_index[atom2]] - crd[molecule.atom_index[atom1]]
        l0 = np.linalg.norm(r0)
    dr = (length / l0 - 1) * r0
    crd[atom2_friends] += dr
    for atom in molecule.atoms:
        i = molecule.atom_index[atom]
        atom.x = crd[i][0]
        atom.y = crd[i][1]
        atom.z = crd[i][2]


def impose_angle(molecule, atom1, atom2, atom3, angle):
    """

    :param molecule:
    :param atom1:
    :param atom2:
    :param atom3:
    :param angle:
    :return:
    """
    crd = molecule.get_atom_coordinates()
    _, atom3_friends = molecule.divide_into_two_parts(atom2, atom3)
    r12 = crd[molecule.atom_index[atom1]] - crd[molecule.atom_index[atom2]]
    r23 = crd[molecule.atom_index[atom3]] - crd[molecule.atom_index[atom2]]
    angle0 = np.arccos(np.dot(r12, r23) / np.linalg.norm(r23) / np.linalg.norm(r12))
    delta_angle = angle - angle0
    crd[atom3_friends] = np.dot(crd[atom3_friends] - crd[molecule.atom_index[atom2]],
                                get_rotate_matrix(np.cross(r12, r23), delta_angle)) + crd[molecule.atom_index[atom2]]
    for atom in molecule.atoms:
        i = molecule.atom_index[atom]
        atom.x = crd[i][0]
        atom.y = crd[i][1]
        atom.z = crd[i][2]


def impose_dihedral(molecule, atom1, atom2, atom3, atom4, dihedral):
    """

    :param molecule:
    :param atom1:
    :param atom2:
    :param atom3:
    :param atom4:
    :param dihedral:
    :return:
    """
    crd = molecule.get_atom_coordinates()
    _, atom4_friends = molecule.divide_into_two_parts(atom2, atom3)
    r12 = crd[molecule.atom_index[atom1]] - crd[molecule.atom_index[atom2]]
    r23 = crd[molecule.atom_index[atom3]] - crd[molecule.atom_index[atom2]]
    r34 = crd[molecule.atom_index[atom3]] - crd[molecule.atom_index[atom4]]
    r12xr23 = np.cross(r12, r23)
    r23xr34 = np.cross(r23, r34)
    cos = np.dot(r12xr23, r23xr34) / np.linalg.norm(r12xr23) / np.linalg.norm(r23xr34)
    cos = max(-0.999999, min(cos, 0.999999))
    dihedral0 = np.arccos(cos)
    dihedral0 = np.pi - np.copysign(dihedral0, np.cross(r23xr34, r12xr23).dot(r23))
    delta_angle = dihedral - dihedral0
    crd[atom4_friends] = np.dot(crd[atom4_friends] - crd[molecule.atom_index[atom3]],
                                get_rotate_matrix(r23, delta_angle)) + crd[molecule.atom_index[atom3]]
    for atom in molecule.atoms:
        i = molecule.atom_index[atom]
        atom.x = crd[i][0]
        atom.y = crd[i][1]
        atom.z = crd[i][2]


def _add_solvent_box(molecule, new_molecule, molmin, molmax, solshape, distance, tolerance, solcrd):
    """

    :param molecule:
    :param new_molecule:
    :param molmin:
    :param molmax:
    :param solshape:
    :param distance:
    :param tolerance:
    :param solcrd:
    :return:
    """
    x0 = molmin[0] - solshape[0] - distance[0]
    while x0 < molmax[0] + distance[3] + solshape[0]:
        y0 = molmin[1] - solshape[1] - distance[1]
        while y0 < molmax[1] + distance[4] + solshape[1]:
            z0 = molmin[2] - solshape[2] - distance[2]
            while z0 < molmax[2] + distance[5] + solshape[2]:
                if (molmax[0] + tolerance + solshape[0] > x0 > molmin[0] - tolerance - solshape[0] and
                        molmax[1] + tolerance + solshape[1] > y0 > molmin[1] - tolerance - solshape[1] and
                        molmax[2] + tolerance + solshape[2] > z0 > molmin[2] - tolerance - solshape[2]):
                    z0 += solshape[2]
                    continue
                for atom in new_molecule.atoms:
                    i = new_molecule.atom_index[atom]
                    atom.x = solcrd[i][0] + x0
                    atom.y = solcrd[i][1] + y0
                    atom.z = solcrd[i][2] + z0
                molecule |= new_molecule
                z0 += solshape[2]
            y0 += solshape[1]
        x0 += solshape[0]


def add_solvent_box(molecule, solvent, distance, tolerance=3):
    """

    :param molecule:
    :param solvent:
    :param distance:
    :param tolerance:
    :return:
    """
    if isinstance(distance, (float, int)):
        distance = [distance] * 6
    elif not isinstance(distance, list):
        raise Exception("parameter distance should be a list, an int or a float")

    if len(distance) == 3:
        distance = distance + distance
    elif len(distance) != 6:
        raise Exception("the length of parameter distance should be 3 or 6")

    if isinstance(molecule, ResidueType):
        new_molecule = Molecule(molecule.name)
        res_a = Residue(molecule)
        for atom in molecule.atoms:
            res_a.Add_Atom(atom)
        new_molecule.Add_Residue(res_a)
        for key, value in sys.modules['__main__'].__dict__.items():
            if value == molecule:
                sys.modules['__main__'].__dict__[key] = new_molecule
        molecule = new_molecule

    molcrd = molecule.get_atom_coordinates()
    molmin = np.min(molcrd, axis=0)
    molmax = np.max(molcrd, axis=0)
    if isinstance(solvent, ResidueType):
        new_molecule = Molecule(solvent.name)
        res_a = Residue(solvent)
        for atom in solvent.atoms:
            res_a.Add_Atom(atom)
        new_molecule.Add_Residue(res_a)
    else:
        new_molecule = solvent.deepcopy()
    solcrd = new_molecule.get_atom_coordinates()
    solmin = np.min(solcrd, axis=0)
    solmax = np.max(solcrd, axis=0)
    solshape = solmax - solmin + tolerance

    _add_solvent_box(molecule, new_molecule, molmin, molmax, solshape, distance, tolerance, solcrd)


def h_mass_repartition(molecules, repartition_mass=1.1, repartition_rate=3, exclude_residue_name="WAT"):
    """

    :param molecules:
    :param repartition_mass:
    :param repartition_rate:
    :param exclude_residue_name:
    :return:
    """
    for res in molecules.residues:
        if res.name == exclude_residue_name:
            continue
        for atom in res.atoms:
            if atom.mass <= repartition_mass:
                connect_atoms = res.type.connectivity[res.type.name2atom(atom.name)]
                assert len(connect_atoms) == 1
                origin_mass = atom.mass
                atom.mass *= repartition_rate
                delta_mass = atom.mass - origin_mass
                for heavy_atom in connect_atoms:
                    res.name2atom(heavy_atom.name).mass -= delta_mass


def solvent_replace(molecule, select, toreplace):
    """

    :param molecule:
    :param select:
    :param toreplace:
    :return:
    """
    solvents = []
    for i in range(len(molecule.residues)):
        if select(molecule.residues[i]):
            solvents.append(i)

    np.random.shuffle(solvents)
    count = 0
    for key, value in toreplace.items():
        assert isinstance(key, ResidueType) or (isinstance(key, Molecule) and len(key.residues) == 1)
        if isinstance(key, Molecule):
            key = key.residues[0].type

        tempi = solvents[count:count + value]
        count += value
        for i in tempi:
            new_residue = Residue(key)
            crd_o = [molecule.residues[i].atoms[0].x, molecule.residues[i].atoms[0].y, molecule.residues[i].atoms[0].z]
            crd0 = [key.atoms[0].x, key.atoms[0].y, key.atoms[0].z]
            for atom in key.atoms:
                new_residue.Add_Atom(atom, x=atom.x + crd_o[0] - crd0[0],
                                     y=atom.y + crd_o[1] - crd0[1], z=atom.z + crd_o[2] - crd0[2])
            molecule.residues[i] = new_residue


def main_axis_rotate(molecule, direction_long=None, direction_middle=None, direction_short=None):
    """

    :param molecule:
    :param direction_long:
    :param direction_middle:
    :param direction_short:
    :return:
    """
    if direction_long is None:
        direction_long = [0, 0, 1]
    if direction_middle is None:
        direction_middle = [0, 1, 0]
    if direction_short is None:
        direction_short = [1, 0, 0]
    molcrd = molecule.get_atom_coordinates()
    eye = np.zeros((3, 3))
    mass_of_center = np.zeros(3)
    total_mass = 0
    for i, atom in enumerate(molecule.atoms):
        xi, yi, zi = molcrd[i]
        total_mass += atom.mass
        mass_of_center += atom.mass * np.array([xi, yi, zi])
    mass_of_center /= total_mass

    for i, atom in enumerate(molecule.atoms):
        xi, yi, zi = molcrd[i] - mass_of_center
        eye += atom.mass * np.array([[yi * yi + zi * zi, -xi * yi, -xi * zi],
                                     [-xi * yi, xi * xi + zi * zi, -yi * zi],
                                     [-xi * zi, -yi * zi, xi * xi + yi * yi]])

    eigval, eigvec = np.linalg.eig(eye)
    t = np.argsort(eigval)
    matrix0 = np.vstack([direction_short, direction_middle, direction_long])
    rotation_matrix = np.dot(matrix0, np.linalg.inv(np.vstack((eigvec[:, t[2]], eigvec[:, t[1]], eigvec[:, t[0]]))))
    molcrd = np.dot(molcrd - mass_of_center, rotation_matrix) + mass_of_center
    for i, atom in enumerate(molecule.atoms):
        atom.x = molcrd[i][0]
        atom.y = molcrd[i][1]
        atom.z = molcrd[i][2]
