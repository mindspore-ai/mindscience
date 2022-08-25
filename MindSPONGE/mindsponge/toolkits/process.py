"""
This **module** is used to process topology and conformations
"""
import os.path
from abc import ABC, abstractmethod

import numpy as np
from .helper import get_rotate_matrix, ResidueType, Molecule, Residue, set_global_alternative_names, Xdict, \
    GlobalSetting, set_attribute_alternative_names
from .build import save_sponge_input
from .load import load_coordinate
from .forcefield.special.min import save_min_bonded_parameters, do_not_save_min_bonded_parameters
from .mdrun import run


def impose_bond(molecule, atom1, atom2, length):
    """
    This **function** is used to impose the distance in ``molecule`` between ``atom1`` and ``atom2`` to ``length``

    usage example::

        import Xponge
        import Xponge.forcefield.amber.ff14sb
        mol = ALA*10
        Impose_Bond(mol, mol.residues[0].CA, mol.residues[0].C, 1.2)

    .. ATTENTION::

        `atom1` and `atom2` should be bonded if they are in one residue

    :param molecule: a ``Molecule`` instance
    :param atom1: the base atom, which will not change its coordinate
    :param atom2: the atom to change its coordinate to fit the length
    :param length: distance in the unit of angstrom
    :return: None
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
    This **function** is used to impose the angle in ``molecule`` between ``atom1``, ``atom2`` and ``atom3`` \
 to ``angle``.

    .. ATTENTION::

        The pairs of ``atom1`` - ``atom2`` and ``atom2`` - ``atom3``  should be bonded.

    :param molecule: a ``Molecule`` instance
    :param atom1: the base atom, which will not change its coordinate
    :param atom2: the base atom, which will not change its coordinate
    :param atom3: the atom to change its coordinate to fit the angle
    :param angle: angle in the unit of rad
    :return: None
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
    This **function** is used to impose the dihedral in ``molecule`` between ``atom1``, ``atom2``, ``atom3`` \
 and ``atom4`` to ``dihedral``.

    .. ATTENTION::

        The pairs of ``atom1`` - ``atom2``,  ``atom2`` - ``atom3``  and ``atom3`` - ``atom4`` should be bonded.

    :param molecule: a ``Molecule`` instance
    :param atom1: the base atom, which will not change its coordinate
    :param atom2: the base atom, which will not change its coordinate
    :param atom3: the atom to change its coordinate to fit the angle
    :param atom4: the atom to change its coordinate to fit the angle
    :param dihedral: dihedral angle in the unit of rad
    :return: None
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
    This **function** add a box full of solvents to a molecule

    :param molecule: the molecule to add the box, either a ``Molecule`` or a ``ResidueType``
    :param solvent: the solvent molecule, either a ``Molecule`` or a ``ResidueType``
    :param distance: the distance between the ``molecule`` and the box in the unit of Angstrom. \
 This can be an ``int`` or a ``float``, and it can be also a list of them with the length 3 or 6, \
 which represents the 3 or 6 directions respectively.
    :param tolerance: the distance between two molecules. 3 for default.
    :return: None
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
    This **function** repartition the mass of light atoms to the connected heavy atoms. \
 This can help the simulation run with a time step of 4 fs.

    .. ATTENTION::

        Many functions use mass to guess the element, the mass repartition may cause error. So call this function \
 at the final step please unless you know what you are doing.

    :param molecules: a ``Molecule``
    :param repartition_mass: if the mass of the atom is not greater than this value, it will be seen as a light atom. \
 1.1 for default and in the unit of Dalton.
    :param repartition_rate: The mass of the light atom will multiplied by this value.
    :param exclude_residue_name: the residue name which will not do the repartition. "WAT" for default.
    :return: None
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


def solvent_replace(molecule, select, toreplace, sort=True):
    """
    This **function** replaces the solvent to some other molecules randomly.

    usage example::

        import Xponge
        import Xponge.forcefield.amber.ff14sb
        import Xponge.forcefield.amber.tip3p
        mol = ALA*10
        Add_Solvent_Box(mol, WAT, 10)
        Solvent_Replace(mol, WAT, {K:10, CL:10})
        #Solvent_Replace(mol, lambda res:res.name == "WAT", {K:10, CL:10})

    :param molecule: a ``Molecule`` instance
    :param select: a **function** to decide which residues should be replaced, \
or a Residue, a ResidueType or a Molecule with only one Residue, \
which the residues to be replaced have the same name
    :param toreplace: a dict, which stores the mapping of molecules to replace and the number of molecules. \
 Every molecule should be a ``ResidueType``, a ``Residue`` or a ``Molecule`` with only one ``Residue``.
    :param sort: whether to sort the residues after replacing
    :return: None
    """
    solvents = []
    for_sort = Xdict()
    if not callable(select):
        if isinstance(select, Molecule):
            select = select.residues[0]
        resname = select.name
        # pylint: disable=unnecessary-lambda-assignment
        select = lambda res: res.name == resname
    for i, resi in enumerate(molecule.residues):
        if select(resi):
            solvents.append(i)
            for_sort[resi] = float("inf")
        else:
            for_sort[resi] = float("-inf")

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
            for_sort[new_residue] = count
    if sort:
        molecule.residues.sort(key=lambda res: for_sort[res])


def main_axis_rotate(molecule, direction_long=None, direction_middle=None, direction_short=None):
    """
    This **function** rotates the main axis of the molecule to the desired direction

    :param molecule: a ``Molecule`` instance
    :param direction_long: a list of three ``int`` or ``float`` to represent the direction vector. \
The long main axis will rotate to this direction.
    :param direction_middle: a list of three ``int`` or ``float`` to represent the direction vector. \
The middle main axis will rotate to this direction.
    :param direction_short: a list of three ``int`` or ``float`` to represent the direction vector. \
The short main axis will rotate to this direction.
    :return: None
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


def get_peptide_from_sequence(sequence, charged_terminal=True):
    """
    This **function** is used to get a peptide from the sequence

    :param sequence: a string, the serial
    :param charged_terminal: whether to change the terminal residues to the corresponding charged residue
    :return: a Molecule instance, the peptide
    """
    assert isinstance(sequence, str) and len(sequence) > 1
    temp_dict = Xdict({"A": "ALA", "G": "GLY", "V": "VAL", "L": "LEU", "I": "ILE", "P": "PRO",
                       "F": "PHE", "Y": "TYR", "W": "TRP", "S": "SER", "T": "THR", "C": "CYS",
                       "M": "MET", "N": "ASN", "Q": "GLN", "D": "ASP", "E": "GLU", "K": "LYS",
                       "R": "ARG", "H": "HIS"}, not_found_message="{} is not an abbreviation for an amino acid")
    temp_dict2 = Xdict({key: ResidueType.get_type(value) for key, value in temp_dict.items()},
                       not_found_message="{} is not an abbreviation for an amino acid")
    if charged_terminal:
        head = "N" + temp_dict[sequence[0]]
        tail = "C" + temp_dict[sequence[-1]]
    else:
        head = temp_dict[sequence[0]]
        tail = temp_dict[sequence[-1]]

    toret = ResidueType.get_type(head)

    for i in sequence[1:-1]:
        toret = toret + temp_dict2[i]
    toret += ResidueType.get_type(tail)
    return toret


def optimize(mol, step=5000, only_bad_coordinate=True):
    """
    This **function** is used to optimize the structure of the Molecule instance

    :param mol: the molecule to optimize
    :param step: the step to minimize
    :param only_bad_coordinate: whether to optimize all the atoms or the atoms whose coordinates are bad
    :return: None
    """
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tempdir:
        temp_prefix = os.path.join(tempdir, "temp")
        temp_out = os.path.join(tempdir, "min")
        save_min_bonded_parameters()
        save_sponge_input(mol, temp_prefix)
        do_not_save_min_bonded_parameters()
        all_to_use = f"""SPONGE_NOPBC -default_in_file_prefix {temp_prefix} -rst {temp_out} -crd {temp_prefix}.dat
        -box {temp_prefix}.box -mdout {temp_out}.out -mdinfo {temp_out}.info -mode minimization 
        -step_limit {step} """
        if only_bad_coordinate:
            all_to_use += f"-mass_in_file {temp_prefix + '_fake_mass.txt'} "
        if GlobalSetting.verbose < 2:
            print_to = f" > {os.devnull}"
        else:
            print_to = ""
        run(all_to_use + "-dt 1e-8" + print_to)

        all_to_use += f"-coordinate_in_file {temp_out+'_coordinate.txt'} "
        for i in [7, 6, 5, 4, 3]:
            run(all_to_use + f"-dt 1e-{i} {print_to}")
        load_coordinate(temp_out+"_coordinate.txt", mol)


class Region(ABC):
    """
    This **abstract class** is used to define a region
    **New From 1.2.6.4**

    :param *regions: the regions
    :param do: eigher 'union' or 'intersect'
    """
    def __init__(self):
        self.side = "in"
        self._side = True
        set_attribute_alternative_names(self)

    @abstractmethod
    def __contains__(self, item):
        pass

    def set_side(self, side):
        """
        This **function** is used to set the side of the region"

        :param side: either "in" or "out"
        :return: None
        """
        self.side = side
        if side == "in":
            self._side = True
        elif side == "out":
            self._side = False
        else:
            raise ValueError("side should be 'in' or 'out'")



class IntersectRegion:
    """
    This **class** is used to get the interset region of some regions
    **New From 1.2.6.4**

    :param *regions: the regions
    """
    def __init__(self, *regions):
        self.regions = regions

    def __contains__(self, item):
        for region in self.regions:
            if item not in region:
                return False
        return True


class UnionRegion:
    """
    This **class** is used to get the union region of some regions
    **New From 1.2.6.4**

    :param *regions: the regions
    """
    def __init__(self, *regions):
        self.regions = regions

    def __contains__(self, item):
        for region in self.regions:
            if item in region:
                return True
        return False


Region.register(UnionRegion)
Region.register(IntersectRegion)


class BlockRegion(Region):
    """
    This **class** is used to define a block region
    **New From 1.2.6.4**

    :param x_low: the lowest x coordinate of the block region
    :param y_low: the lowest y coordinate of the block region
    :param z_low: the lowest z coordinate of the block region
    :param x_high: the highest x coordinate of the block region
    :param y_high: the highest y coordinate of the block region
    :param z_high: the highest z coordinate of the block region
    :param side: either "in" or "out"
    """
    def __init__(self, x_low, y_low, z_low, x_high, y_high, z_high, side="in"):
        self.x_low = x_low
        self.y_low = y_low
        self.z_low = z_low
        self.x_high = x_high
        self.y_high = y_high
        self.z_high = z_high
        super().__init__()
        self.set_side(side)

    def __contains__(self, item):
        ans = self.x_low < item[0] < self.x_high and self.y_low < item[1] < self.y_high \
              and self.z_low < item[2] < self.z_high
        return ans if self._side else not ans


class SphereRegion(Region):
    """
    This **class** is used to define a sphere region
    **New From 1.2.6.4**

    :param x: the x coordinate of the sphere origin
    :param y: the y coordinate of the sphere origin
    :param z: the z coordinate of the sphere origin
    :param r: the radius of the sphere
    :param side: either "in" or "out"
    """
    def __init__(self, x, y, z, r, side="in"):
        self.x = x
        self.y = y
        self.z = z
        self._r2 = r * r
        super().__init__()
        self.set_side(side)

    def __contains__(self, item):
        ans = (item[0] - self.x) ** 2 + (item[1] - self.y) ** 2 + (item[2] - self.z) ** 2
        ans = ans < self._r2
        return ans if self._side else not ans


class FrustumRegion(Region):
    """
    This **class** is used to define a frustum region
    **New From 1.2.6.4**

    :param x1: the x coordinate of the first circle origin
    :param y1: the y coordinate of the first circle origin
    :param z1: the z coordinate of the first circle origin
    :param r1: the radius of the first circle origin
    :param x2: the x coordinate of the second circle origin
    :param y2: the y coordinate of the second circle origin
    :param z2: the z coordinate of the second circle origin
    :param r2: the radius of the second circle origin
    :param side: either "in" or "out"
    """
    def __init__(self, x1, y1, z1, r1, x2, y2, z2, r2, side="in"):
        self.r1 = r1
        self.r2 = r2
        self.o1 = np.array([x1, y1, z1], dtype=np.float32)
        self.o2 = np.array([x2, y2, z2], dtype=np.float32)
        self.axis = self.o2 - self.o1
        self.length = np.linalg.norm(self.axis)
        self.axis /= self.length
        self.k = (r2 - r1) / self.length
        super().__init__()
        self.set_side(side)

    def __contains__(self, item):
        crd = np.array(item) - self.o1
        length = np.linalg.norm(crd)
        projection = np.dot(crd, self.axis)
        distance = length * length - projection * projection
        r = self.r1 + self.k * projection
        ans = self.length > projection > 0 and distance < r * r
        return ans if self._side else not ans


class PrismRegion(Region):
    """
    This **class** is used to define a prism (parallelepiped) region
    **New From 1.2.6.4**

    :param x0: the x coordinate of the origin
    :param y0: the y coordinate of the origin
    :param z0: the z coordinate of the origin
    :param x1: the x coordinate of the first basis vector
    :param y1: the y coordinate of the first basis vector
    :param z1: the z coordinate of the first basis vector
    :param x2: the x coordinate of the second basis vector
    :param y2: the y coordinate of the second basis vector
    :param zz: the z coordinate of the second basis vector
    :param x3: the x coordinate of the third basis vector
    :param y3: the y coordinate of the third basis vector
    :param z3: the z coordinate of the third basis vector
    :param side: either "in" or "out"
    """
    def __init__(self, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, side="in"):
        self.l0 = np.array([x0, y0, z0], dtype=np.float32)
        self.l1 = np.array([x1, y1, z1], dtype=np.float32)
        self.l2 = np.array([x2, y2, z2], dtype=np.float32)
        self.l3 = np.array([x3, y3, z3], dtype=np.float32)
        self.n3 = np.cross(self.l1, self.l2)
        self.n3 /= np.linalg.norm(self.n3)
        self.n2 = np.cross(self.l3, self.l1)
        self.n2 /= np.linalg.norm(self.n2)
        self.n1 = np.cross(self.l2, self.l3)
        self.n1 /= np.linalg.norm(self.n1)
        self.length = np.array([np.dot(self.l1, self.n1), np.dot(self.l2, self.n2), np.dot(self.l3, self.n3)])
        assert np.all(self.length > 0), "The basis vectors should mmet the right-handed axis system requirements"
        super().__init__()
        self.set_side(side)

    def __contains__(self, item):
        crd = np.array(item) - self.l0
        ans = 0 <= np.dot(crd, self.n1) < self.length[0] and 0 <= np.dot(crd, self.n2) < self.length[1] and \
              0 <= np.dot(crd, self.n3) < self.length[2]

        return ans if self._side else not ans


class Lattice:
    """
    This **class** is used to help with the process of the lattice
    **New From 1.2.6.4**

    :param style: the style of the lattice. "custom", "template:NAME" or the key values in Lattice.styles
        "sc", "fcc", "bcc", "hcp", "diamond" are in the default template styles.
    :param basis_molecule: the unit to repeat, a Residue, a ResidueType or a Molecule
    :param scale: the scale of the lattice
    :param origin: a list with 3 numbers, the origin of the lattice. [0,0,0] for default.
    :param cell_length: the length of the unit cell. [1,1,1] for default.
    :param cell_angle: the angle of the unit cell. [90,90,90] for default.
    :param basis_position: a list of lists, every sublist has 3 numbers for coordinates.
    :param spacing: a list with 3 numbers, the spacing distance in x, y, z directions.
    """
    styles = Xdict(not_found_message="There is no lattice style named {}")

    def __init__(self, style="custom", basis_molecule=None, scale=None, origin=None, cell_length=None,
                 cell_angle=None, basis_position=None, spacing=None):
        self.basis_molecule = basis_molecule
        if style == "custom" or style.startswith("template:"):
            self.scale = 1
            self.origin = [0, 0, 0]
            self.cell_length = [1, 1, 1]
            self.cell_angle = np.array([90, 90, 90], dtype=float)
            self.spacing = [0, 0, 0]
            self.basis_position = []
            if self.scale is not None:
                self.scale = scale
            if origin is not None:
                self.origin = origin
            if cell_length is not None:
                self.cell_length = cell_length
            if cell_angle is not None:
                self.cell_angle = np.array(cell_angle)
            assert np.all(self.cell_angle <= 90), "the cell angle should be not greater than 90 degree"
            if spacing is not None:
                self.spacing = spacing
            if basis_position is not None:
                self.basis_position = basis_position
        else:
            old_style = Lattice.styles[style]
            self.scale = scale
            self.origin = old_style.origin
            self.cell_length = old_style.cell_length
            self.cell_angle = old_style.cell_angle
            self.basis_position = old_style.basis_position
            self.spacing = old_style.spacing
        if not style.startswith("template:") and self.basis_molecule is None:
            raise ValueError("basis molecule should not be None for a non-template lattice")
        if not style.startswith("template:") and self.scale is None:
            raise ValueError("scale should not be None for a non-template lattice")
        if style.startswith("template:"):
            style_name = style.split(":")[1].strip()
            Lattice.styles[style_name] = self

        set_attribute_alternative_names(self)

    @staticmethod
    def _judge_region(x1, y1, z1, x2, y2, z2, region, mol, basis_mol, res_len):
        """

        :param x1:
        :param y1:
        :param z1:
        :param x2:
        :param y2:
        :param z2:
        :param region:
        :param mol:
        :param basis_mol:
        :param res_len:
        :return:
        """
        if (x2, y2, z2) in region:
            mol |= basis_mol
            for res in mol.residues[res_len:]:
                for atom in res.atoms:
                    atom.x = atom.x - x1 + x2
                    atom.y = atom.y - y1 + y2
                    atom.z = atom.z - z1 + z2

    def create(self, box, region, mol=None):
        """
        This **function** is used to create basis molecules to the region in the box

        :param box: the box of the system
        :param region: the region to create the basis_molecule
        :param mol: if ``mol`` the Molecule instance is provided, basis molecules will be added to ``mol``
        :return: a new Molecule instance, or the Molecule instance ``mol``
        """
        if not isinstance(box, BlockRegion) or box.side == "out":
            raise ValueError("Box should only be a BlockRegion with side == 'in' !")
        if not mol:
            mol = Molecule("unnamed")
        mol.box_length = [box.x_high - box.x_low, box.y_high - box.y_low, box.z_high - box.z_low]
        cell_angle = self.cell_angle * np.pi / 180.0
        c0, c1, c2 = np.cos(cell_angle)
        _, _, s2 = np.sin(cell_angle)
        basis_mol = self.basis_molecule
        res_len = -1
        if isinstance(basis_mol, Molecule):
            basis_mol.atoms = [atom for residue in basis_mol.residues for atom in residue.atoms]
            res_len = -len(basis_mol.residues)
        dl1x = self.scale * (self.cell_length[0] + self.spacing[0])

        dl2x = self.scale * (self.cell_length[1] + self.spacing[1])
        dl2y = dl2x * s2
        dl2x *= c2

        dl3x = self.scale * (self.cell_length[2] + self.spacing[2])
        dl3y = dl3x * (c0 - c2 * c1) / s2
        dl3z = dl3x * np.sqrt(1 - c2 * c2 - c1 * c1 - c0 * c0 + 2 * c0 * c1 * c2) / s2
        dl3x *= c1
        bps = np.array([[dl1x * basis[0] + dl2x * basis[1] + dl3x * basis[2],
                         dl2y * basis[1] + dl3y * basis[2], dl3z * basis[2]] for basis in self.basis_position])
        x_init = 0
        y_init = 0
        z0 = box.z_low + self.origin[2]
        x1 = basis_mol.atoms[0].x
        y1 = basis_mol.atoms[0].y
        z1 = basis_mol.atoms[0].z
        while z0 < box.z_high:
            y0 = box.y_low + self.origin[1] + y_init
            while y0 < box.y_high:
                x0 = box.x_low + self.origin[0] + x_init
                while x0 < box.x_high:
                    for basis in bps:
                        x2 = basis[0] + x0
                        y2 = basis[1] + y0
                        z2 = basis[2] + z0
                        self._judge_region(x1, y1, z1, x2, y2, z2, region, mol, basis_mol, res_len)
                    x0 += dl1x
                x_init += dl2x
                x_init %= dl1x
                y0 += dl2y
            x_init += dl3x
            x_init %= dl1x
            y_init += dl3y
            y_init %= dl2y
            z0 += dl3z
        return mol


SIMPLE_CUBIC_LATTICE = Lattice("template:sc", basis_position=[[0, 0, 0]])

BODY_CENTERED_CUBIC_LATTICE = Lattice("template:bcc", basis_position=[[0, 0, 0], [0.5, 0.5, 0.5]])

FACE_CENTERED_CUBIC_LATTICE = Lattice("template:fcc", basis_position=[[0, 0, 0], [0.5, 0, 0.5],
                                                                      [0, 0.5, 0.5], [0.5, 0.5, 0]])

HEXAGONAL_CLOSE_PACKED_LATTICE = Lattice("template:hcp",
                                         basis_position=[[0, 0, 0], [1/3, 1/3, 0.5]],
                                         cell_angle=[90, 90, 60],
                                         cell_length=[1, 1, 2/3*np.sqrt(6)])

DIAMOND_LATTICE = Lattice("template:diamond", basis_position=[[0, 0, 0], [0, 0.5, 0.5],
                                                              [0.5, 0, 0.5], [0.5, 0.5, 0],
                                                              [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
                                                              [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]])

set_global_alternative_names()
