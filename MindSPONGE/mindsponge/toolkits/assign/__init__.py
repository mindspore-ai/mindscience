"""
This **package** is used to assign the properties for atoms, residues and molecules
"""
from collections import OrderedDict
from itertools import groupby
import numpy as np
from ..helper import AtomType, ResidueType, Xopen, Xdict, set_real_global_variable, set_attribute_alternative_names, \
    set_global_alternative_names, Guess_Element_From_Mass


class AssignRule:
    """
    This **class** is to be the rule to determine the atom type for one atom

    :param name: the name of the rule
    """
    all = Xdict(not_found_message="AssignRule {} not found. Did you import the proper force field?")

    def __init__(self, name):
        self.name = name
        AssignRule.all[name] = self
        self.rules = OrderedDict()
        set_attribute_alternative_names(self)

    def add_rule(self, atomtype):
        """
        This **function** is used as a **decorator** to add the atom type - judge function

        :param atomtype: a string or an AtomType instance
        :return: a **function**, which wraps a judge function (receiving the Assign instance and the atom index \
and giving True or False as a result)
        """
        if isinstance(atomtype, str):
            atomtype = AtomType.get_type(atomtype)
        elif not isinstance(atomtype, AtomType):
            raise TypeError("atomtype should be a string or AtomType")

        def wrapper(rule_function):
            self.rules[atomtype] = rule_function

        return wrapper


class _RING():
    """
    This **class** is used to help with the ring assignment.
    """
    def __init__(self, atom_list):
        min_index = np.argmin(atom_list)
        self.atoms = atom_list[min_index:] + atom_list[:min_index]
        reverse_list = self.atoms[::-1]
        reverse_list = reverse_list[-1:] + reverse_list[:-1]
        if reverse_list[1] < self.atoms[1]:
            self.atoms = reverse_list
        self.tohash = "-".join(["%d" % atom for atom in self.atoms])
        self.is_pure_aromatic_ring = None
        self.is_pure_aliphatic_ring = None
        self.is_planar_ring = None
        self.out_plane_double_bond = None

    def __repr__(self):
        return self.tohash

    def __hash__(self):
        return hash(self.tohash)

    def __eq__(self, other):
        return isinstance(other, _RING) and self.tohash == other.tohash

    @staticmethod
    def add_rings_basic_marker(assign, rings):
        """

        :param assign:
        :param rings:
        :return:
        """
        for ring in rings:
            for atom in ring.atoms:
                assign.Add_Atom_Marker(atom, "RG")
                assign.Add_Atom_Marker(atom, "RG%d" % len(ring.atoms))

    @staticmethod
    def check_rings_type(assign, rings):
        """

        :param assign:
        :param rings:
        :return:
        """
        for ring in rings:
            ring.check_pure_aromatic(assign)
            ring.check_pure_aliphatic_and_planar(assign)
            ring.check_out_plane_double_bond(assign)

            if not ring.is_pure_aromatic_ring:
                for atom in ring.atoms:
                    if ring.is_pure_aliphatic_ring:
                        assign.Add_Atom_Marker(atom, "AR5")
                    elif ring.is_planar_ring:
                        if ring.out_plane_double_bond:
                            assign.Add_Atom_Marker(atom, "AR3")
                        else:
                            assign.Add_Atom_Marker(atom, "AR2")
                    else:
                        assign.Add_Atom_Marker(atom, "AR4")

    @staticmethod
    def get_rings(assign):
        """

        :param assign:
        :return:
        """
        current_path = []
        current_path_sons = Xdict()
        current_work = []
        current_path_father = Xdict()
        have_found_rings = set([])
        for atom0 in range(len(assign.atoms)):
            current_path.append(atom0)
            current_work.extend([[atom, atom0] for atom in assign.bonds[atom0].keys()])
            current_path_sons[atom0] = len(assign.bonds[atom0])
            current_path_father = []
            while current_path:
                work_atom, from_atom = current_work.pop()
                current_path.append(work_atom)
                current_path_father.append(from_atom)
                bond_atom = []
                for atom in assign.bonds[work_atom].keys():
                    if atom != from_atom:
                        try:
                            index = current_path.index(atom)
                            have_found_rings.add(_RING(current_path[index:]))
                        except ValueError:
                            bond_atom.append([atom, work_atom])

                if len(current_path) < 9:
                    current_path_sons[work_atom] = len(bond_atom)
                    current_work.extend(bond_atom)

                else:
                    current_path_sons[work_atom] = 0

                for atom in current_path[::-1]:
                    if current_path_sons[atom] == 0:
                        pop_atom = current_path.pop()
                        current_path_sons.pop(pop_atom)
                        if current_path_father:
                            father_atom = current_path_father.pop()
                            current_path_sons[father_atom] -= 1
        return have_found_rings

    def get_3_neighbors(self):
        """

        :return:
        """
        for i, atom in enumerate(self.atoms):
            yield self.atoms[i - 2], self.atoms[i - 1], atom

    def check_pure_aromatic(self, assign):
        """

        :param assign:
        :return:
        """
        if len(self.atoms) == 6:
            self.is_pure_aromatic_ring = True
            for atom in self.atoms:
                if not assign.Atom_Judge(atom, "C3") and not assign.Atom_Judge(atom, "N2") and not assign.Atom_Judge(
                        atom, "N3"):
                    self.is_pure_aromatic_ring = False
                    break
                if assign.Atom_Judge(atom, "N3"):
                    temp = 0
                    for bonded_atom, bond_order in assign.bonds[atom].items():
                        temp += bond_order
                    if temp == 3:
                        self.is_pure_aromatic_ring = False
                        break
                for bonded_atom, bond_order in assign.bonds[atom].items():
                    if bond_order == 2 and "RG" not in assign.atom_marker[bonded_atom].keys():
                        self.is_pure_aromatic_ring = False
                        break
                if not self.is_pure_aromatic_ring:
                    break
        else:
            self.is_pure_aromatic_ring = False

    def check_pure_aliphatic_and_planar(self, assign):
        """

        :param assign:
        :return:
        """
        self.is_pure_aliphatic_ring = True
        self.is_planar_ring = True
        for atom in self.atoms:
            if self.is_pure_aromatic_ring:
                assign.Add_Atom_Marker(atom, "AR1")
                for i in range(6):
                    assign.Add_Bond_Marker(self.atoms[i - 1], self.atoms[i], "AB")
            if not assign.Atom_Judge(atom, "C4"):
                self.is_pure_aliphatic_ring = False
            if (not assign.Atom_Judge(atom, "C3") and not assign.Atom_Judge(atom, "N2")
                    and not assign.Atom_Judge(atom, "N3") and not assign.Atom_Judge(atom, "O2")
                    and not assign.Atom_Judge(atom, "S2") and not assign.Atom_Judge(atom, "P2")
                    and not assign.Atom_Judge(atom, "P3")):
                self.is_planar_ring = False

    def check_out_plane_double_bond(self, assign):
        """

        :param assign:
        :return:
        """
        self.out_plane_double_bond = False
        for atom in self.atoms:
            for bonded_atom, order in assign.bonds[atom].items():
                if assign.atoms[bonded_atom] != "C" and order == 2 and bonded_atom not in self.atoms:
                    self.out_plane_double_bond = True


class Assign():
    """
    This **class** is used to assign properties for atoms, which is called an "assignment"

    :param name: the name of the molecule
    """
    XX = set("CNOPS")
    XA = set("OS")
    XB = set("NP")
    XC = set(["F", "Cl", "Br", "I"])
    XD = set("SP")
    XE = set(["N", "O", "F", "Cl", "Br", "S", "I"])

    def __init__(self, name="ASN"):
        self.name = name
        self.atom_numbers = 0
        self.atoms = []
        self.names = []
        self.element_details = []
        self.coordinate = None
        self.charge = None
        self.atom_types = Xdict()
        self.atom_marker = Xdict()
        self.bonds = Xdict()
        self.ar_bonds = Xdict()
        self.am_bonds = Xdict()
        self.bond_marker = Xdict()
        set_attribute_alternative_names(self)

    def add_index_to_name(self):
        """
        This **function** adds the atom index to the atom name

        :return: None
        """
        for i in range(self.atom_numbers):
            self.names[i] += str(i)

    def atom_judge(self, atom, string):
        """
        This **function** helps judge whether the atom belongs to the mask. For example, "O2" means an oxygen atom \
connected to two other atoms, "N4" means a nitrogen atom connected to four other atoms.

        :param atom: the index of the atom
        :param string: a string mask  of a list of string masks.
        :return:
        """
        assert isinstance(string, (list, str))
        if isinstance(string, str):
            todo = [string]
        else:
            todo = string
        judge = False
        for s in todo:
            element, links = [''.join(list(g)) for k, g in groupby(s, key=lambda x: x.isdigit())]
            if self.atoms[atom] == element and len(self.bonds[atom]) == int(links):
                judge = True
                break
        return judge

    def add_atom(self, element, x, y, z, name="", charge=0.0):
        """
        This **function** adds an atom to the Assign

        :param element: the chemical symbol for the element. "O" - oxygen, "H" - hydrogen for example.
        :param x: the x coordinate
        :param y: the y coordinate
        :param z: the z coordinate
        :param name: the name of the atom
        :param charge: the charge of the atom
        :return: None
        """
        if "." in element:
            element, element_detail = element.split(".")
            element_detail = "." + element_detail
        else:
            element_detail = ""
        self.element_details.append(element_detail)
        self.atoms.append(element)
        self.bonds[self.atom_numbers] = Xdict()
        self.bond_marker[self.atom_numbers] = Xdict()
        self.atom_marker[self.atom_numbers] = Xdict()
        self.atom_types[self.atom_numbers] = None
        self.atom_numbers += 1
        self.names.append(name)
        if self.coordinate is None:
            self.coordinate = np.array([[float(x), float(y), float(z)]])
        else:
            self.coordinate = np.vstack((self.coordinate, np.array([x, y, z])))
        if self.charge is None:
            self.charge = np.array([charge])
        else:
            self.charge = np.hstack((self.charge, np.array([charge])))

    def add_atom_marker(self, atom, marker):
        """
        This **function** adds a marker to an atom

        :param atom: the atom index
        :param marker: the marker
        :return: None
        """
        if marker in self.atom_marker[atom].keys():
            self.atom_marker[atom][marker] += 1
        else:
            self.atom_marker[atom][marker] = 1

    def add_bond(self, atom1, atom2, order=-1):
        """
        This **function** adds a bond to two atoms

        :param atom1: the index of the first atom
        :param atom2: the index of the the second atom
        :param order: the bond order
        :return: None
        """
        self.bonds[atom1][atom2] = order
        self.bond_marker[atom1][atom2] = set([])
        self.bonds[atom2][atom1] = order
        self.bond_marker[atom2][atom1] = set([])

    def add_bond_marker(self, atom1, atom2, marker, only1=False):
        """
        This **function** adds a marker to a bond

        :param atom1: the index of the first atom
        :param atom2: the index of the the second atom
        :param marker: the marker
        :param only1: only add the marker to the atom1 - atom2 bond instead of the atom2 - atom1 bond
        :return: None
        """
        self.bond_marker[atom1][atom2].add(marker)
        if marker in self.atom_marker[atom1]:
            self.atom_marker[atom1][marker] += 1
        else:
            self.atom_marker[atom1][marker] = 1
        if not only1:
            self.bond_marker[atom2][atom1].add(marker)
            if marker in self.atom_marker[atom2]:
                self.atom_marker[atom2][marker] += 1
            else:
                self.atom_marker[atom2][marker] = 1

    def determine_equal_atoms(self):
        """
        This **function** dertermines the chemical equalvalent atoms

        .. NOTE::

            The pyckage RDKit is needed for this **function**

        :return: a list of equalvalent atom index lists
        """
        from ..helper.rdkit import Find_Equal_Atoms
        return Find_Equal_Atoms(self)

    def determine_ring_and_bond_type(self):
        """
        This **function** determine the ring and the bond type

        :return: None
        """
        have_found_rings = _RING.get_rings(self)
        _RING.add_rings_basic_marker(self, have_found_rings)
        _RING.check_rings_type(self, have_found_rings)

        for atom in range(len(self.atoms)):
            dlo = 0
            noto = 0
            for atom2, order in self.bonds[atom].items():
                if self.Atom_Judge(atom2, "O1"):
                    dlo += 1
                else:
                    noto += 1
            if dlo >= 1 >= noto:
                for atom2, order in self.bonds[atom].items():
                    if self.Atom_Judge(atom2, "O1"):
                        self.Add_Bond_Marker(atom, atom2, "DLB")
            for atom2, order in self.bonds[atom].items():
                if "DLB" in self.bond_marker[atom][atom2]:
                    self.Add_Bond_Marker(atom, atom2, "DL", True)
                    self.Add_Bond_Marker(atom, atom2, "sb", True)
                elif order == 1:
                    self.Add_Bond_Marker(atom, atom2, "sb", True)
                    if "AB" not in self.bond_marker[atom][atom2]:
                        self.Add_Bond_Marker(atom, atom2, "SB", True)
                elif order == 2:
                    self.Add_Bond_Marker(atom, atom2, "db", True)
                    if "AB" not in self.bond_marker[atom][atom2]:
                        self.Add_Bond_Marker(atom, atom2, "DB", True)
                else:
                    self.Add_Bond_Marker(atom, atom2, "tb", True)

    def determine_atom_type(self, rule):
        """
        This **function** determines the atom type.

        .. ATTENTION::

            Before determining the atom type, the ring and bond type should be determined. If you use the function \
``Get_Assignment_From_XXX``, the ring and bond type has been determined by the function, but when you build the Assign \
instance yourself, remember to determine the ring and bond type!

        :param rule: a string or an AssignRule instance
        :return: None
        """
        if isinstance(rule, str):
            rule = AssignRule.all[rule]

        for i in range(len(self.atoms)):
            find_type = False
            for atom_type, type_rule in rule.rules.items():
                if type_rule(i, self):
                    self.atom_types[i] = atom_type
                    find_type = True
                    break

            assert find_type, "No atom type found for assignment %s of atom #%d" % (self.name, i)

    def to_residuetype(self, name, charge=None):
        """
        This **function** converts the Assign instance to the ResidueType instance

        :param name: the name of the ResidueType instance
        :param charge: the charge of atoms. If set to None, internal charge will be used
        :return: the ResidueType instance
        """
        temp = ResidueType(name=name)
        if not charge:
            if self.charge is None:
                charge = np.zeros(self.atom_numbers)
            else:
                charge = self.charge
        count = Xdict()
        for i in range(self.atom_numbers):
            assert self.atom_types[i] is not None
            if self.names[i]:
                atom_name = self.names[i]
            elif self.atoms[i] in count.keys():
                atom_name = self.atoms[i] + "%d" % count[self.atoms[i]]
                self.names[i] = atom_name
                count[self.atoms[i]] += 1
            else:
                count[self.atoms[i]] = 1
                atom_name = self.atoms[i]
                self.names[i] = atom_name
            temp.Add_Atom(atom_name, self.atom_types[i], x=self.coordinate[i][0],
                          y=self.coordinate[i][1], z=self.coordinate[i][2])
            temp.atoms[-1].charge = charge[i]
        for i, bondi in self.bonds.items():
            for j in bondi.keys():
                temp.Add_Connectivity(temp.atoms[i], temp.atoms[j])
        set_real_global_variable(name, temp)
        return temp

    def calculate_charge(self, method, **parameters):
        """
        This **function** calculates the partial charge for every atom

        :param method: the method to calculate the charge
        :param parameters: the parameters to calculate the charge
        :return: None
        """
        method = method.upper()
        if method == "RESP":
            from . import resp
            self.charge = resp.RESP_Fit(self, basis=parameters.get("basis", "6-31g*"), opt=parameters.get("opt", False),
                                        charge=parameters.get("charge", 0), spin=parameters.get("spin", 0),
                                        extra_equivalence=parameters.get("extra_equivalence", []),
                                        grid_density=parameters.get("grid_density", 6),
                                        grid_cell_layer=parameters.get("grid_cell_layer", 4),
                                        a1=parameters.get("a1", 0.0005),
                                        a2=parameters.get("a2", 0.001), two_stage=parameters.get("two_stage", True),
                                        only_esp=parameters.get("only_esp", False),
                                        radius=parameters.get("radius", None))
        else:
            raise ValueError("methods should be one of the following: 'RESP'")

    def save_as_pdb(self, filename):
        """
        This **function** saves the instance as a pdb file

        :param filename: the name of the output file
        :return: None
        """
        if not isinstance(filename, str):
            raise TypeError("filename needed to save an assignment to a pdb file")
        towrite = towrite = "REMARK   Generated By Xponge (Assignment)\n"
        count = Xdict()
        for i in range(self.atom_numbers):
            if self.names[i]:
                atom_name = self.names[i]
            elif self.atoms[i] in count.keys():
                atom_name = self.atoms[i] + "%d" % count[self.atoms[i]]
                self.names[i] = atom_name
                count[self.atoms[i]] += 1
            else:
                count[self.atoms[i]] = 1
                atom_name = self.atoms[i]
                self.names[i] = atom_name
        for i, atom in enumerate(self.atoms):
            towrite += "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%17s%2s\n" % (i + 1, self.names[i],
                                                                                 self.name, " ", 1,
                                                                                 self.coordinate[i][0],
                                                                                 self.coordinate[i][1],
                                                                                 self.coordinate[i][2], " ", atom)

        for i in range(self.atom_numbers):
            bonded_atoms = list(self.bonds[i].keys())
            bonded_atoms.sort()
            bonded_atoms = [bonded_atoms[i:i + 4] for i in range(0, len(bonded_atoms), 4)]
            if bonded_atoms:
                for atoms in bonded_atoms:
                    towrite += "CONECT %4d" % (i + 1)
                    for atom in atoms:
                        towrite += " %4d" % (atom + 1)
                    towrite += "\n"

        f = Xopen(filename, "w")
        f.write(towrite)
        f.close()

    def save_as_mol2(self, filename):
        """
        This **function** saves the instance as a mol2 file

        :param filename: the name of the output file
        :return: None
        """
        if not isinstance(filename, str):
            raise TypeError("filename needed to save an assignment as a mol2 file")
        bonds = []
        for i in range(self.atom_numbers):
            for j, order in self.bonds[i].items():
                if i < j:
                    if i in self.ar_bonds.keys() and j in self.ar_bonds[i]:
                        bonds.append("%6d %6d ar\n" % (i + 1, j + 1))
                    elif i in self.am_bonds.keys() and j in self.am_bonds[i]:
                        bonds.append("%6d %6d am\n" % (i + 1, j + 1))
                    else:
                        bonds.append("%6d %6d %1d\n" % (i + 1, j + 1, order))
        bonds.sort(key=lambda x: list(map(int, x.split()[:2])))
        count = Xdict()
        for i in range(self.atom_numbers):
            if self.names[i]:
                atom_name = self.names[i]
            elif self.atoms[i] in count.keys():
                atom_name = self.atoms[i] + "%d" % count[self.atoms[i]]
                self.names[i] = atom_name
                count[self.atoms[i]] += 1
            else:
                count[self.atoms[i]] = 1
                atom_name = self.atoms[i]
                self.names[i] = atom_name
        towrite = "@<TRIPOS>MOLECULE\n%s\n %d %d 1 0 1\nSMALL\nUSER_CHARGES\n" % (
            self.name, self.atom_numbers, len(bonds))
        towrite += "@<TRIPOS>ATOM\n"
        for i, atom in enumerate(self.atoms):
            towrite += "%6d %4s %8.4f %8.4f %8.4f   %-8s %5d %8s %10.6f\n" % (
                i + 1, self.names[i], self.coordinate[i][0], self.coordinate[i][1], self.coordinate[i][2],
                atom + self.element_details[i], 1, self.name, self.charge[i])

        towrite += "@<TRIPOS>BOND\n"
        for i, bond in enumerate(bonds):
            towrite += "%6d %s" % (i + 1, bond)
        towrite += "@<TRIPOS>SUBSTRUCTURE\n"
        towrite += "%5d %8s %6d ****               0 ****  **** \n" % (1, self.name, 1)

        f = Xopen(filename, "w")
        f.write(towrite)
        f.close()


def get_assignment_from_pubchem(parameter, keyword):
    """
    This **function** gets an Assign instance from PubChem

    usage example::

        a1 = Get_Assignment_From_PubChem("ethane", "name")
        a2 = Get_Assignment_From_PubChem("CC", "smiles")

    :param parameter: the parameter to search on PubChem
    :param keyword: the keyword to search on PubChem
    :return: the Assign instance
    """
    import pubchempy as pcp
    cs = pcp.get_compounds(parameter, keyword, record_type='3d')
    if not cs:
        raise pcp.NotFoundError
    if len(cs) == 1:
        assign = Assign()
        c = cs[0]
        for atom in c.atoms:
            assign.Add_Atom(atom.element, atom.x, atom.y, atom.z)
        for bond in c.bonds:
            assign.Add_Bond(bond.aid1 - 1, bond.aid2 - 1, bond.order)
        assign.Determine_Ring_And_Bond_Type()
        return assign
    raise NotImplementedError


def get_assignment_from_pdb(filename, determine_bond_order=True, only_residue=""):
    """
    This **function** gets an Assign instance from a pdb file

    :param filename: the name of the input file
    :param determine_bond_order: whether determine the bond order automatically
    :param only_residue: only get the residue with the name same as ``only_residue``
    :return: the Assign instance
    :raise NotImplementedError: it has not been implemented when determine_bond_order is True
    """
    assign = Assign()
    index_atom_map = Xdict()
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if only_residue:
                    resname = line[17:20].strip()
                    if resname != only_residue:
                        continue
                index = int(line[6:11])
                index_atom_map[index] = assign.atom_numbers
                atom_name = line[12:16].strip()
                element = line[76:78].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                assign.Add_Atom(element, x, y, z, atom_name)
            if line.startswith("CONECT"):
                atom = int(line[6:11])
                if atom not in index_atom_map.keys():
                    continue
                for bonded_atom_i in range(11, 31, 5):
                    try:
                        temp = line[bonded_atom_i:bonded_atom_i + 5]
                        bonded_atom = int(temp)
                    except ValueError:
                        break
                    if bonded_atom in index_atom_map.keys():
                        assign.Add_Bond(index_atom_map[atom], index_atom_map[int(bonded_atom)])
    if determine_bond_order:
        raise NotImplementedError
    return assign


def get_assignment_from_residuetype(restype):
    """
    This **function** gets an Assign instance from a ResidueType instance

    :param restype: the ResidueType instance
    :return: the Assign instance
    """
    assign = Assign()
    for atom in restype.atoms:
        assign.Add_Atom(Guess_Element_From_Mass(atom.mass), atom.x, atom.y, atom.z, atom.name)
    for atom in restype.atoms:
        i = restype.atom2index(atom)
        for atomb in restype.connectivity[atom]:
            j = restype.atom2index(atomb)
            if i < j:
                assign.Add_Bond(i, j)
    return assign


def _deal_with_ar_bonds(assign):
    """

    :param assign:
    :return:
    """
    ar_bonds_atoms = list(assign.ar_bonds.keys())
    ar_bonds_atoms.sort(key=lambda x: (x, len(assign.ar_bonds[x])))
    doubled = Xdict()
    checked = Xdict()
    for ar_atom in ar_bonds_atoms:
        assign.ar_bonds[ar_atom].sort(key=lambda x: (x, len(assign.ar_bonds[x])))
        doubled[ar_atom] = False
        checked[ar_atom] = False

    working_space = []
    while ar_bonds_atoms:
        working_space.append(ar_bonds_atoms.pop())
        while working_space:
            work_atom = working_space.pop()
            if checked[work_atom]:
                continue
            checked[work_atom] = True
            for neighbor in assign.ar_bonds[work_atom]:
                if not checked[neighbor]:
                    working_space.append(neighbor)
            for neighbor in assign.ar_bonds[work_atom][::-1]:
                if doubled[work_atom]:
                    break
                if not doubled[neighbor] and not doubled[work_atom]:
                    assign.bonds[neighbor][work_atom] = 2
                    doubled[neighbor] = True
                    doubled[work_atom] = True


def get_assignment_from_mol2(filename):
    """
    This **function** gets an Assign instance from a mol2 file

    :param filename: the name of the input file
    :return: the Assign instance
    """
    with open(filename) as f:
        flag = None
        subflag = None
        for line in f:
            if not line.strip():
                continue
            if line.startswith("@<TRIPOS>"):
                flag = line[9:].strip()
            elif flag == "MOLECULE":
                if subflag is None:
                    assign = Assign(line.strip())
                    subflag = "0"
            # 处理原子信息
            elif flag == "ATOM":
                words = line.split()
                atom_name = words[1]
                element = words[5]
                x = float(words[2])
                y = float(words[3])
                z = float(words[4])
                charge = float(words[8])
                assign.Add_Atom(element, x, y, z, atom_name, charge)
            elif flag == "BOND":
                words = line.split()
                if words[3] in "1234567890":
                    assign.Add_Bond(int(words[1]) - 1, int(words[2]) - 1, int(words[3]))
                elif words[3] == "ar":
                    atom1 = int(words[1]) - 1
                    atom2 = int(words[2]) - 1
                    assign.Add_Bond(atom1, atom2, 1)
                    if atom1 not in assign.ar_bonds.keys():
                        assign.ar_bonds[atom1] = [atom2]
                    else:
                        assign.ar_bonds[atom1].append(atom2)
                    if atom2 not in assign.ar_bonds.keys():
                        assign.ar_bonds[atom2] = [atom1]
                    else:
                        assign.ar_bonds[atom2].append(atom1)
                elif words[3] == "am":
                    atom1 = int(words[1]) - 1
                    atom2 = int(words[2]) - 1
                    assign.Add_Bond(atom1, atom2, 1)
                    if atom1 not in assign.am_bonds.keys():
                        assign.am_bonds[atom1] = [atom2]
                    else:
                        assign.am_bonds[atom1].append(atom2)
                    if atom2 not in assign.am_bonds.keys():
                        assign.am_bonds[atom2] = [atom1]
                    else:
                        assign.am_bonds[atom2].append(atom1)
                else:
                    raise NotImplementedError(
                        "No implemented method to process bond #%s type %s" % (words[0], words[3]))

    _deal_with_ar_bonds(assign)
    assign.Determine_Ring_And_Bond_Type()
    return assign

set_global_alternative_names()
