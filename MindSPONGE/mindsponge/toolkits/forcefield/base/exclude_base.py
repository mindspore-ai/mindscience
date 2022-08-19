"""
This **module** is the basic setting for the force field non bonded exclusion
"""
from functools import partial
from ... import Molecule
from ...helper import set_attribute_alternative_name, set_global_alternative_names, Xdict


class Exclude:
    """
    This **class** is used to set non bonded exclusion generally
    """

    #: The current effective Exclude class
    current = None


    def __init__(self, *args, **kwargs):
        n = 4
        if len(args) == 1:
            n = args[0]
        self.n = kwargs.get("n", n)
        Exclude.current = self
        set_attribute_alternative_name(self, self.get_excluded_atoms)

        def write_exclude(mol):
            exclude_numbers = 0
            excludes = []
            for atom in mol.atoms:
                temp = atom.extra_excluded_atoms.copy()
                atom_self_index = mol.atom_index[atom]
                filter_func = partial(lambda x, y: mol.atom_index[x] > y, y=atom_self_index)
                excludes.append(
                    list(map(lambda x: mol.atom_index[x], filter(filter_func, temp))))
                exclude_numbers += len(excludes[-1])
                for i in range(2, n + 1):
                    for aton in atom.linked_atoms.get(i, []):
                        if mol.atom_index[aton] > atom_self_index and aton not in temp:
                            temp.add(aton)
                            exclude_numbers += 1
                            excludes[-1].append(mol.atom_index[aton])
                for aton in atom.linked_atoms.get("v", []):
                    if mol.atom_index[aton] > atom_self_index and aton not in temp:
                        temp.add(aton)
                        exclude_numbers += 1
                        excludes[-1].append(mol.atom_index[aton])
                excludes[-1].sort()
            towrite = "%d %d\n" % (len(mol.atoms), exclude_numbers)
            for exclude in excludes:
                exclude.sort()
                towrite += "%d %s\n" % (len(exclude), " ".join([str(atom_index) for atom_index in exclude]))

            return towrite

        Molecule.Set_Save_SPONGE_Input("exclude")(write_exclude)

        # pylint: disable=unused-argument
        def todo(mol, sys_kwarg, ene_kwarg, use_pbc):
            if "exclude" not in ene_kwarg:
                ene_kwarg["exclude"] = []
            temp_dict = self.get_excluded_atoms(mol)
            temp_dict = {atom: list(set_) for atom, set_ in temp_dict.items()}
            exclude_max = max(len(atoms) for atoms in temp_dict.values())

            ene_kwarg["exclude"].append([[mol.atom_index[temp_dict[atom][j]]
                                          if j < len(temp_dict[atom]) else len(mol.atoms)
                                          for j in range(exclude_max)] for i, atom in enumerate(mol.atoms)])
            for li in ene_kwarg["exclude"][-1]:
                li.sort()


        Molecule.Set_MindSponge_Todo("exclude")(todo)

    def get_excluded_atoms(self, molecule):
        """
        This **function** gives the excluded atoms of a molecule

        :param molecule: a Molecule instance
        :return: a dict, which stores the atom - excluded atoms mapping
        """
        temp_dict = Xdict()
        for atom in molecule.atoms:
            temp_dict[atom] = atom.extra_excluded_atoms.copy()
            for i in range(2, self.n + 1):
                for aton in atom.linked_atoms.get(i, []):
                    temp_dict[atom].add(aton)
            for aton in atom.linked_atoms.get("v", []):
                temp_dict[atom].add(aton)
        return temp_dict

set_global_alternative_names()
