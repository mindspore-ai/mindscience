"""
This **module** is used to load and read
"""
import os

import numpy as np

from .helper import Molecule, Residue, ResidueType, AtomType, GlobalSetting, Xdict, Xopen, \
    set_real_global_variable, set_global_alternative_names


##########################################################################
# General Format
##########################################################################
def _mol2_atom(line, current_residue_index, current_residue, ignore_atom_type, temp, current_molecule,
               atom_residue_map):
    """

    :param line:
    :param current_residue_index:
    :param current_residue:
    :param ignore_atom_type:
    :param temp:
    :param current_molecule:
    :param atom_residue_map:
    :return:
    """
    words = line.split()
    if current_residue_index is None or int(words[6]) != current_residue_index:
        current_residue_index = int(words[6])
        if words[7] not in ResidueType.get_all_types():
            set_real_global_variable(words[7], ResidueType(name=words[7]))
            temp = True
        else:
            temp = False
        if current_residue:
            current_molecule.Add_Residue(current_residue)
        current_residue = Residue(ResidueType.get_type(words[7]))
    if ignore_atom_type:
        temp_atom_type = AtomType.get_type("UNKNOWN")
    else:
        temp_atom_type = AtomType.get_type(words[5])

    if temp:
        current_residue.type.Add_Atom(words[1], temp_atom_type, *words[2:5])
        current_residue.type.atoms[-1].Update(**{"charge[e]": float(words[8])})
    current_residue.Add_Atom(words[1], temp_atom_type, *words[2:5])
    current_residue.atoms[-1].Update(**{"charge[e]": float(words[8])})
    atom_residue_map[words[0]] = [words[1], current_residue, current_residue_index, temp, current_residue.atoms[-1],
                                  current_residue.type.atoms[-1]]
    return current_residue_index, current_residue, temp


def _mol2_bond(line, current_molecule, atom_residue_map):
    """

    :param line:
    :param current_molecule:
    :param atom_residue_map:
    :return:
    """
    words = line.split()
    if atom_residue_map[words[1]][1] == atom_residue_map[words[2]][1]:
        if atom_residue_map[words[1]][3]:
            atom_residue_map[words[1]][1].type.Add_Connectivity(atom_residue_map[words[1]][0],
                                                                atom_residue_map[words[2]][0])
        atom_residue_map[words[1]][1].Add_Connectivity(atom_residue_map[words[1]][0], atom_residue_map[words[2]][0])
    else:
        current_molecule.Add_Residue_Link(atom_residue_map[words[1]][4], atom_residue_map[words[2]][4])
        index_diff = atom_residue_map[words[1]][2] - atom_residue_map[words[2]][2]
        if abs(index_diff) == 1:
            if atom_residue_map[words[1]][3]:
                if index_diff < 0:
                    atom_residue_map[words[1]][1].type.tail = atom_residue_map[words[1]][0]
                else:
                    atom_residue_map[words[1]][1].type.head = atom_residue_map[words[1]][0]
            if atom_residue_map[words[2]][3]:
                if index_diff < 0:
                    atom_residue_map[words[2]][1].type.head = atom_residue_map[words[2]][0]
                else:
                    atom_residue_map[words[2]][1].type.tail = atom_residue_map[words[2]][0]


def load_mol2(filename, ignore_atom_type=False):
    """
    This **function** is used to load a mol2 file

    :param filename: the name of the file to load
    :param ignore_atom_type: ignore the atom types in the mol2 file
    :return: a Molecule instance
    """
    with open(filename) as f:
        # 存储读的时候的临时信息，key是编号
        # value是list：原子名(0)、residue(1)、residue编号(2)、是否是新的residue type(3)、该原子(4)、residue type的最新原子(5)
        atom_residue_map = {}
        flag = None
        nline = 0
        current_molecule = None
        current_residue = None
        current_residue_index = None
        temp = None
        for line in f:
            if line.strip():
                nline += 1
            else:
                continue

            if line.startswith("@<TRIPOS>"):
                if flag == "ATOM":
                    current_molecule.Add_Residue(current_residue)
                flag = line[9:].strip()
                nline = 0
            elif flag == "MOLECULE":
                if nline == 1:
                    current_molecule = Molecule(line.strip())
            elif flag == "ATOM":
                current_residue_index, current_residue, temp = _mol2_atom(line, current_residue_index, current_residue,
                                                                          ignore_atom_type, temp, current_molecule,
                                                                          atom_residue_map)
            elif flag == "BOND":
                _mol2_bond(line, current_molecule, atom_residue_map)
    return current_molecule


def _pdb_ssbond_before(chain, residue_type_map, ssbonds):
    """

    :param chain:
    :param residue_type_map:
    :param ssbonds:
    :return:
    """
    for ssbond in ssbonds:
        res_a_index = chain[ssbond[15]][int(ssbond[17:21])]
        residue_type_map[res_a_index] = "CYX"
        res_b_index = chain[ssbond[29]][int(ssbond[31:35])]
        residue_type_map[res_b_index] = "CYX"


def _pdb_ssbond_after(chain, ssbonds, molecule):
    """

    :param chain:
    :param ssbonds:
    :param molecule:
    :return:
    """
    for ssbond in ssbonds:
        res_a_index = chain[ssbond[15]][int(ssbond[17:21])]
        res_b_index = chain[ssbond[29]][int(ssbond[31:35])]
        if res_a_index > res_b_index:
            res_a_index, res_b_index = (res_b_index, res_a_index)
        res_a = molecule.residues[res_a_index]
        res_b = molecule.residues[res_b_index]
        molecule.Add_Residue_Link(res_a.name2atom(res_a.type.connect_atoms["ssbond"]),
                                  res_b.name2atom(res_b.type.connect_atoms["ssbond"]))


def _pdb_add_residue(filename, molecule, position_need, residue_type_map, ignore_hydrogen, ignore_unknown_name):
    """

    :param filename:
    :param molecule:
    :param position_need:
    :param residue_type_map:
    :param ignore_hydrogen:
    :param ignore_unknown_name:
    :return:
    """
    current_residue_count = -1
    current_residue_index = None
    current_residue = None
    current_resname = None
    links = []
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                extra = line[16]
                resindex = int(line[22:26])
                resname = line[17:20].strip()
                atomname = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if current_residue_index is None or current_residue_index != resindex or \
                        current_resname != resname:
                    current_residue_count += 1
                    if current_residue:
                        molecule.Add_Residue(current_residue)
                        if current_residue_index is not None and current_residue.type.tail and ResidueType.get_type(
                                residue_type_map[current_residue_count]).head:
                            links.append(len(molecule.residues))
                    current_residue = Residue(ResidueType.get_type(residue_type_map[current_residue_count]))
                    current_residue_index = resindex
                    current_resname = resname
                if extra not in (" ", position_need):
                    continue
                if ignore_hydrogen and atomname.startswith("H"):
                    continue
                try:
                    current_residue.Add_Atom(atomname, x=x, y=y, z=z)
                except KeyError as ke:
                    if ignore_unknown_name:
                        continue
                    raise ke
            elif line.startswith("TER"):
                current_residue_index = None
                current_resname = None

    if current_residue:
        molecule.Add_Residue(current_residue)
    for count in links:
        res_a = molecule.residues[count]
        res_b = molecule.residues[count - 1]
        molecule.Add_Residue_Link(res_a.name2atom(res_a.type.head), res_b.name2atom(res_b.type.tail))


def _pdb_judge_histone(judge_histone, residue_type_map, current_histone_information):
    """

    :param judge_histone:
    :param residue_type_map:
    :param current_histone_information:
    :return:
    """
    if judge_histone and residue_type_map and residue_type_map[-1] in GlobalSetting.HISMap["HIS"].keys():
        if current_histone_information["DeltaH"]:
            if current_histone_information["EpsilonH"]:
                residue_type_map[-1] = GlobalSetting.HISMap["HIS"][residue_type_map[-1]]["HIP"]
            else:
                residue_type_map[-1] = GlobalSetting.HISMap["HIS"][residue_type_map[-1]]["HID"]
        else:
            residue_type_map[-1] = GlobalSetting.HISMap["HIS"][residue_type_map[-1]]["HIE"]
        current_histone_information = {"DeltaH": False, "EpsilonH": False}


def load_pdb(filename, judge_histone=True, position_need="A", ignore_hydrogen=False, ignore_unknown_name=False):
    """
    This **function** is used to load a pdb file

    :param filename: the name of the file to load
    :param judge_histone: judge the protonized state of the histone residues
    :param position_need: the position character to read
    :param ignore_hydrogen: do not read the atom with a name beginning with "H"
    :param ignore_unknown_name: do not read the atom with an unknown name **New From 1.2.6.4**
    :return: a Molecule instance
    """
    molecule = Molecule(os.path.splitext(os.path.basename(filename))[0])
    chain = Xdict()
    ssbonds = []
    residue_type_map = []
    current_residue_count = -1
    current_residue_index = None
    current_resname = None
    current_histone_information = {"DeltaH": False, "EpsilonH": False}
    with open(filename) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                resindex = int(line[22:26])
                resname = line[17:20].strip()
                atomname = line[12:16].strip()
                if current_residue_index is None:
                    _pdb_judge_histone(judge_histone, residue_type_map, current_histone_information)
                    current_residue_count += 1
                    current_resname = resname
                    residue_type_map.append(resname)
                    current_residue_index = resindex
                    chain[chr(ord("A") + len(chain.keys()))] = Xdict()
                    chain[chr(ord("A") + len(chain.keys())-1)][resindex] = current_residue_count
                    if resname in GlobalSetting.PDBResidueNameMap["head"].keys():
                        resname = GlobalSetting.PDBResidueNameMap["head"][resname]
                elif current_residue_index != resindex or current_resname != resname:
                    _pdb_judge_histone(judge_histone, residue_type_map, current_histone_information)
                    current_residue_count += 1
                    current_resname = resname
                    current_residue_index = resindex
                    chain[chr(ord("A") + len(chain.keys()) - 1)][resindex] = current_residue_count

                    residue_type_map.append(resname)

                if judge_histone and resname in GlobalSetting.HISMap["HIS"].keys():
                    if atomname == GlobalSetting.HISMap["DeltaH"]:
                        current_histone_information["DeltaH"] = True
                    elif atomname == GlobalSetting.HISMap["EpsilonH"]:
                        current_histone_information["EpsilonH"] = True
            elif line.startswith("TER"):
                current_residue_index = None
                current_resname = None
                if residue_type_map[-1] in GlobalSetting.PDBResidueNameMap["tail"].keys():
                    residue_type_map[-1] = GlobalSetting.PDBResidueNameMap["tail"][residue_type_map[-1]]
                _pdb_judge_histone(judge_histone, residue_type_map, current_histone_information)

            elif line.startswith("SSBOND"):
                ssbonds.append(line)

    current_residue_index = None
    if residue_type_map[-1] in GlobalSetting.PDBResidueNameMap["tail"].keys():
        residue_type_map[-1] = GlobalSetting.PDBResidueNameMap["tail"][residue_type_map[-1]]

    _pdb_ssbond_before(chain, residue_type_map, ssbonds)
    _pdb_add_residue(filename, molecule, position_need, residue_type_map, ignore_hydrogen, ignore_unknown_name)
    _pdb_ssbond_after(chain, ssbonds, molecule)

    return molecule


##########################################################################
# SPONGE Format
##########################################################################
def load_coordinate(filename, mol=None):
    """
    This **function** is used to read the SPONGE coordinate in file

    :param filename: the coordinate file to load
    :param mol: the molecule or residue to load the coordinate into
    :return: two numpy arrays, representing the coordinates and the box information respectively
    """
    with Xopen(filename, "r") as f:
        atom_numbers = int(f.readline().split()[0])
        crd = np.zeros((atom_numbers, 3), dtype=np.float32)
        box = np.zeros(6, dtype=np.float32)
        for i in range(atom_numbers):
            crd[i][:] = np.array([float(x) for x in f.readline().split()])
        box = np.array(f.readline().split(), dtype=np.float32)
    if mol:
        for i, atom in enumerate(mol.atoms):
            atom.x = crd[i][0]
            atom.y = crd[i][1]
            atom.z = crd[i][2]
        if isinstance(mol, Molecule):
            mol.box_length = box
    return crd, box


##########################################################################
# amber Format
##########################################################################
def _frcmod_cmap(line, cmap, temp_cmp, cmap_flag):
    """

    :param line:
    :param cmap:
    :param temp_cmp:
    :param cmap_flag:
    :return:
    """
    if line.startswith("%FLAG"):
        if "CMAP_COUNT" in line:
            if temp_cmp:
                for res in temp_cmp["residues"]:
                    cmap[res] = temp_cmp["info"]
            temp_cmp = {"residues": [], "info": {"resolution": 24, "count": int(line.split()[-1]), "parameters": []}}
            cmap_flag = "CMAP_COUNT"
        elif "CMAP_RESOLUTION" in line:
            temp_cmp["info"]["resolution"] = int(line.split()[-1])
            cmap_flag = "CMAP_RESOLUTION"
        elif "CMAP_RESLIST" in line:
            cmap_flag = "CMAP_RESLIST"
        elif "CMAP_TITLE" in line:
            cmap_flag = "CMAP_TITLE"
        elif "CMAP_PARAMETER" in line:
            cmap_flag = "CMAP_PARAMETER"
    elif cmap_flag == "CMAP_RESLIST":
        temp_cmp["residues"].extend(line.split())
    elif cmap_flag == "CMAP_PARAMETER":
        temp_cmp["info"]["parameters"].extend([float(x) for x in line.split()])
    return temp_cmp, cmap_flag


def _frcmod_atoms_words(line, n):
    """

    :param line:
    :param n:
    :return:
    """
    return [word.strip() for word in line[:n].split("-")], line[n:].split()


def load_frcmod(filename, nbtype="RE"):
    """
    This **function** is used to load a frcmod file

    :param filename: the name of the file to load
    :param nbtype: the non-bonded interaction recording type in the frcmod file.
    :return: a list of strings, including atoms, bonds, angles, propers, impropers, ljs, cmap information respectively
    """
    with open(filename) as f:
        f.readline()
        flag = None
        atom_types = {}  # 元素符号和质量
        bonds = "name  k[kcal/mol·A^-2]    b[A]\n"
        angles = "name  k[kcal/mol·rad^-2]    b[degree]\n"
        propers = "name  k[kcal/mol]    phi0[degree]    periodicity    reset\n"
        reset = 1
        impropers = "name  k[kcal/mol]    phi0[degree]    periodicity\n"
        cmap = {}
        cmap_flag = None
        temp_cmp = {"residues": []}
        if nbtype == "SK":
            raise NotImplementedError
        if nbtype == "AC":
            ljs = "name A[kcal/mol·A^-12]   B[kcal/mol·A^-6]\n"
        elif nbtype == "RE":
            ljs = "name rmin[A]   epsilon[kcal/mol]\n"

        for line in f:
            if not line.strip():
                continue
            words = line.split()
            if flag != "CMAP" and len(words) == 1:
                flag = line.strip()
            elif flag[:4] == "MASS":
                atom_types[words[0]] = words[1]
            elif flag[:4] == "BOND":
                atoms, words = _frcmod_atoms_words(line, 5)
                bonds += "-".join(atoms) + "\t" + words[0] + "\t" + words[1] + "\n"
            elif flag[:4] == "ANGL":
                atoms, words = _frcmod_atoms_words(line, 8)
                angles += "-".join(atoms) + "\t" + words[0] + "\t" + words[1] + "\n"
            elif flag[:4] == "DIHE":
                atoms, words = _frcmod_atoms_words(line, 11)
                propers += "-".join(atoms) + "\t" + str(float(words[1]) / int(words[0])) + "\t" + words[2] + "\t" + str(
                    abs(int(float(words[3])))) + "\t" + str(reset) + "\n"
                if int(float(words[3])) < 0:
                    reset = 0
                else:
                    reset = 1
            elif flag[:4] == "IMPR":
                atoms, words = _frcmod_atoms_words(line, 11)
                impropers += "-".join(atoms) + "\t" + words[0] + "\t" + words[1] + "\t" + str(
                    int(float(words[2]))) + "\n"
            elif flag[:4] == "NONB":
                words = line.split()
                ljs += words[0] + "-" + words[0] + "\t" + words[1] + "\t" + words[2] + "\n"
            elif flag[:4] == "CMAP":
                temp_cmp, cmap_flag = _frcmod_cmap(line, cmap, temp_cmp, cmap_flag)

    for res in temp_cmp["residues"]:
        cmap[res] = temp_cmp["info"]
    atoms = "name  mass  LJtype\n"
    for atom, mass in atom_types.items():
        atoms += atom + "\t" + mass + "\t" + atom + "\n"
    toret = [atoms, bonds, angles, propers, impropers, ljs, cmap]
    return toret


def _parmdat_read_harmonic_bonds(f, bonds, n):
    """

    :param f:
    :param bonds:
    :param n:
    :return:
    """
    for line in f:
        if not line.strip():
            break
        atoms, words = _frcmod_atoms_words(line, n)
        bonds += "-".join(atoms) + "\t" + words[0] + "\t" + words[1] + "\n"
    return bonds


def load_parmdat(filename):
    """
    This **function** is used to load a parmdat file

    :param filename: the name of the file to load
    :return: a list of strings, including atoms, bonds, angles, propers, impropers, ljs information respectively
    """
    with open(filename) as f:
        f.readline()
        # 读原子
        atom_types = {}  # 元素符号和质量
        lj_types = Xdict()  # 元素符号和LJ类型
        for line in f:
            if not line.strip():
                break

            words = line.split()
            atom_types[words[0]] = words[1]
            lj_types[words[0]] = words[0]
        f.readline()
        # 读键长
        bonds = "name  k[kcal/mol·A^-2]    b[A]\n"
        bonds = _parmdat_read_harmonic_bonds(f, bonds, 5)

        # 读键角
        angles = "name  k[kcal/mol·rad^-2]    b[degree]\n"
        angles = _parmdat_read_harmonic_bonds(f, angles, 8)

        # 读恰当二面角
        reset = 1
        propers = "name  k[kcal/mol]    phi0[degree]    periodicity    reset\n"
        for line in f:
            if not line.strip():
                break
            atoms, words = _frcmod_atoms_words(line, 11)
            propers += "-".join(atoms) + "\t" + str(float(words[1]) / int(words[0])) + "\t" + words[2] + "\t" + str(
                abs(int(float(words[3])))) + "\t" + str(reset) + "\n"
            if int(float(words[3])) < 0:
                reset = 0
            else:
                reset = 1
        # 读非恰当二面角
        impropers = "name  k[kcal/mol]    phi0[degree]    periodicity\n"
        for line in f:
            if not line.strip():
                break
            atoms, words = _frcmod_atoms_words(line, 11)
            impropers += "-".join(atoms) + "\t" + words[0] + "\t" + words[1] + "\t" + str(
                int(float(words[2]))) + "\n"

        # 跳过水的信息
        f.readline()
        f.readline()

        # 读LJ种类
        for line in f:
            if not line.strip():
                break
            atoms = line.split()
            atom0 = atoms.pop(0)
            for atom in atoms:
                lj_types[atom] = atom0

        # 读LJ信息
        word = f.readline().split()[1]
        if word == "SK":
            raise NotImplementedError
        if word == "AC":
            ljs = "name A[kcal/mol·A^-12]   B[kcal/mol·A^-6]\n"
        elif word == "RE":
            ljs = "name rmin[A]   epsilon[kcal/mol]\n"

        for line in f:
            if not line.strip():
                break
            words = line.split()
            ljs += words[0] + "-" + words[0] + "\t" + words[1] + "\t" + words[2] + "\n"

    atoms = "name  mass  LJtype\n"
    for atom, mass in atom_types.items():
        atoms += atom + "\t" + mass + "\t" + lj_types[atom] + "\n"
    toret = [atoms, bonds, angles, propers, impropers, ljs]
    return toret


def load_rst7(filename, mol=None):
    """
    This **function** is used to load a rst7 file

    :param filename: the name of the file to load
    :param mol: the molecule to load the coordinates
    :return: a tuple including coordinates and box information
    """
    crds = []
    with open(filename) as f:
        f.readline()
        words = f.readline().split()
        atom_numbers = int(words[0])
        line = ""
        for line in f:
            words = line.split()
            while words and len(crds) < atom_numbers * 3:
                crds.append(float(words.pop(0)))
        box = [float(i) for i in line.split()]
    crds = np.array(crds).reshape((-1, 3))
    mol.box_length = box[:3]
    count = -1
    for residue in mol.residues:
        for atom in residue.atoms:
            count += 1
            atom.x = crds[count][0]
            atom.y = crds[count][1]
            atom.z = crds[count][2]

    return crds, box


##########################################################################
# GROMACS Format
##########################################################################
class GromacsTopologyIterator():
    """
    This **class** is used to read a GROMACS topology file

    usage example::

        f = GromacsTopologyIterator("example.itp")
        for line in f:
            print(line)

    :param filename: the name of the file to read
    :param macros: the macros used to read the Gromacs topology file

    """

    def __init__(self, filename=None, macros=None):
        self.files = []
        self.filenames = []

        self.flag = ""
        self.macro_define_stat = []

        if macros:
            self.defined_macros = macros
        else:
            self.defined_macros = {}
        if filename:
            self._add_iterator_file(filename)

    def __iter__(self):
        self.flag = ""
        self.macro_define_stat = []
        return self

    def __next__(self):
        while self.files:
            f = self.files[-1]
            line = f.readline()
            if line:
                line = self._line_preprocess(line)
                if line[0] == "#":
                    line = self._line_define(line)
                elif self.macro_define_stat and not self.macro_define_stat[-1]:
                    line = next(self)
                elif "[" in line and "]" in line:
                    self.flag = line[1:-1].strip()
                    line = next(self)
                for macro, tobecome in self.defined_macros.items():
                    line = line.replace(macro, tobecome)
                return line

            f.close()
            self.files.pop()
            self.filenames.pop()

        raise StopIteration

    def _add_iterator_file(self, filename):
        """

        :param filename:
        :return:
        """
        if self.files:
            filename = os.path.abspath(os.path.join(os.path.dirname(self.filenames[-1]), filename.replace('"', '')))
        else:
            filename = os.path.abspath(filename.replace('"', ''))

        f = Xopen(filename, "r")
        self.files.append(f)
        self.filenames.append(filename)

    def _line_preprocess(self, line):
        """

        :param line:
        :return:
        """
        line = line.strip()
        comment = line.find(";")
        if comment >= 0:
            line = line[:comment]
        while line and line[-1] == "\\":
            line = line[:-1] + " " + next(self).strip()
        if not line:
            line = next(self)
        return line

    def _line_define(self, line):
        """

        :param line:
        :return:
        """
        words = line.split()
        if words[0] == "#ifdef":
            macro = words[1]
            if self.macro_define_stat and not self.macro_define_stat[-1]:
                self.macro_define_stat.append(False)
            elif macro in self.defined_macros.keys():
                self.macro_define_stat.append(True)
            else:
                self.macro_define_stat.append(False)
        elif words[0] == "#else":
            if len(self.macro_define_stat) <= 1 or self.macro_define_stat[-2]:
                self.macro_define_stat[-1] = not self.macro_define_stat[-1]
        elif words[0] == "#endif":
            self.macro_define_stat.pop()
        elif self.macro_define_stat and not self.macro_define_stat[-1]:
            next(self)
        elif words[0] == "#define":
            if len(words) > 2:
                self.defined_macros[words[1]] = line[line.find(words[1]) + len(words[1]):].strip()
            else:
                self.defined_macros[words[1]] = ""
        elif words[0] == "#include":
            self._add_iterator_file(words[1])
        elif words[0] == "#undef":
            self.defined_macros.pop(words[1])
        elif words[0] == "#error":
            raise Exception(line)
        line = next(self)
        return line


def _ffitp_dihedrals(line, output):
    """

    :param line:
    :param output:
    :return:
    """
    words = line.split()
    func = words[4]
    if func == "1":
        output["dihedrals"] += "-".join(words[:4]) + " " + " ".join(words[5:]) + " 0\n"
    elif func == "2":
        temp1 = [words[1], words[2], words[0], words[3]]
        temp2 = [words[1], words[2], words[3], words[0]]
        if words[0][0].upper() in ("C", "N", "S"):
            output["impropers"] += "-".join(temp1) + " {b} {k}".format(b=float(words[5]), k=float(words[6]) / 2) + "\n"
        else:
            output["impropers"] += "-".join(temp2) + " {b} {k}".format(b=float(words[5]), k=float(words[6]) / 2) + "\n"
    elif func == "9":
        for i in range(5, len(words), 20):
            output["dihedrals"] += "-".join(words[:4]) + " " + " ".join(words[i:i + 3]) + " 0\n"
    else:
        raise NotImplementedError


def load_ffitp(filename, macros=None):
    """
    This **function** is used to load a fftip file

    .. ATTENTION::

        This is used to read a force field itp (ffitp) file instead of a simple itp file for a molecule.

    :param filename: the name of the file to load
    :param macros: the macros used to read the Gromacs topology file
    :return: a dict, which stores the name of the forcefield term - the corresponding information mapping
    """
    iterator = GromacsTopologyIterator(filename, macros)
    output = Xdict()
    output["nb14"] = "name  kLJ  kee\n"
    output["atomtypes"] = "name mass charge[e] LJtype\n"
    output["bonds"] = "name b[nm] k[kJ/mol·nm^-2]\n"
    output["angles"] = "name b[degree] k[kJ/mol·rad^-2]\n"
    output["Urey-Bradley"] = "name b[degree] k[kJ/mol·rad^-2] r13[nm] kUB[kJ/mol·nm^-2]\n"
    output["dihedrals"] = "name phi0[degree] k[kJ/mol] periodicity  reset\n"
    output["impropers"] = "name phi0[degree] k[kJ/mol·rad^-2]\n"
    output["cmaps"] = Xdict()
    for line in iterator:
        if iterator.flag == "":
            continue
        if iterator.flag == "defaults":
            words = line.split()
            assert int(words[0]) == 1, "SPONGE Only supports Lennard-Jones now"
            if int(words[1]) == 1:
                output["LJ"] = "name A[kJ/mol·nm^6] B[kJ/mol·nm^12]\n"
                output["nb14_extra"] = "name A[kJ/mol·nm^6] B[kJ/mol·nm^12] kee\n"
            else:
                output["LJ"] = "name sigma[nm] epsilon[kJ/mol] \n"
                output["nb14_extra"] = "name sigma[nm] epsilon[kJ/mol] kee\n"
            fudge_lj = float(words[3])
            fudge_qq = float(words[4])
            if words[2] == "yes":
                output["nb14"] += "X-X {fudgeLJ} {fudgeQQ}\n".format(fudgeLJ=fudge_lj, fudgeQQ=fudge_qq)

        elif iterator.flag == "atomtypes":
            words = line.split()
            output["atomtypes"] += "{type} {mass} {charge} {type}\n".format(type=words[0], mass=float(words[2]),
                                                                            charge=float(words[3]))
            output["LJ"] += "{type}-{type} {V} {W}\n".format(type=words[0], V=float(words[5]), W=float(words[6]))
        elif iterator.flag == "pairtypes":
            words = line.split()
            if len(words) <= 3:
                output["nb14"] += "{atom1}-{atom2} {kLJ} {kee}\n".format(atom1=words[0], atom2=words[1], kLJ=fudge_lj,
                                                                         kee=fudge_qq)
            elif words[2] == "1":
                output["nb14_extra"] += "{atom1}-{atom2} {V} {W} {kee}\n".format(atom1=words[0], atom2=words[1],
                                                                                 V=float(words[3]), W=float(words[4]),
                                                                                 kee=fudge_qq)
                output["nb14"] += "{atom1}-{atom2} 0 0\n".format(atom1=words[0], atom2=words[1])
            elif words[2] == "2":
                raise NotImplementedError
        elif iterator.flag == "bondtypes":
            words = line.split()
            func = words[2]
            if func == "1":
                output["bonds"] += "{atom1}-{atom2} {b} {k}\n".format(atom1=words[0], atom2=words[1], b=float(words[3]),
                                                                      k=float(words[4]) / 2)
            else:
                raise NotImplementedError
        elif iterator.flag == "angletypes":
            words = line.split()
            func = words[3]
            if func == "1":
                output["angles"] += "-".join(words[:3]) + " {b} {k}".format(b=float(words[4]),
                                                                            k=float(words[5]) / 2) + "\n"
            elif func == "5":
                output["Urey-Bradley"] += "-".join(words[:3]) + " {b} {k} {b2} {k2}".format(b=float(words[4]),
                                                                                            k=float(words[5]) / 2,
                                                                                            b2=float(words[6]),
                                                                                            k2=float(
                                                                                                words[7]) / 2) + "\n"
            else:
                raise NotImplementedError
        elif iterator.flag == "dihedraltypes":
            _ffitp_dihedrals(line, output)
        elif iterator.flag == "cmaptypes":
            words = line.split()
            output["cmaps"]["-".join(words[:5])] = {"resolution": int(words[7]),
                                                    "parameters": list(map(lambda x: float(x) / 4.184, words[8:]))}
        elif iterator.flag == "nonbond_params":
            words = line.split()
            output["LJ"] += "{type1}-{type2} {V} {W}\n".format(type1=words[0], type2=words[1], V=float(words[3]),
                                                               W=float(words[4]))
    return output

set_global_alternative_names()
