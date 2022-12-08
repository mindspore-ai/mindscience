residue_atom_renaming_swaps
===========================

Four of the 20 amino acids have partial atoms that are symmetrical. This dict records the mapping information of atomic pairs with symmetric invariance in each amino acid. Key is the type of amino acid and value is the pair of uncertain atoms.

+-----------------------+-------------------------------+
| key                   | value                         |
+=======================+===============================+
| ASP                   | {'OD1': 'OD2'}                |
+-----------------------+-------------------------------+
| GLU                   | {'OE1': 'OE2'}                |
+-----------------------+-------------------------------+
| PHE                   | {'CD1': 'CD2', 'CE1': 'CE2'}  |
+-----------------------+-------------------------------+
| TYR                   | {'CD1': 'CD2', 'CE1': 'CE2'}  |
+-----------------------+-------------------------------+