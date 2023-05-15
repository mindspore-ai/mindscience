# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"res_constants"
import numpy as np

with open('./commons/nmr_hydrogen_equivariance.txt', 'r') as f:
    lines = f.readlines()
EQ_GROUPS = []
for line in lines:
    lsp = line.split()
    eqg = [lsp[0], lsp[1], set(lsp[2].split(',')), set(lsp[3].split(','))]
    EQ_GROUPS.append(eqg)
EQ_GROUPS = np.array(EQ_GROUPS)

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

AA_3TO1 = {val: key for key, val in restype_1to3.items()}

atom_types = []
EQUI_VARIANCE = {}
for line in lines:
    words = line.split()
    if len(words) < 4 or words[0] == "##":
        continue
    aatype, hetero_atype, hetero_aname, equivariance = words

    equivariance = equivariance.split(",")
    aatype = restype_1to3.get(aatype)

    atom_types.extend(equivariance)

    if aatype not in EQUI_VARIANCE.keys():
        EQUI_VARIANCE[aatype] = {}

    for hname in equivariance:
        EQUI_VARIANCE.get(aatype)[hname] = {
            "equivariance": equivariance,
            "hetero_info": [hetero_atype, hetero_aname]
        }

atom_template = {'name': 'HA',
                 'res': 'MET10',
                 'atom_id': 169,
                 'segid': '    ',
                 'hetero_name': 'CA',
                 'type': 'H'}

peak_template = {'distance': None,
                 'peak_id': 2271,
                 'upper_bound': None,
                 'lower_bond': None,
                 'weight': 1.0,
                 'active': 1,
                 'merged': 0,
                 'ref_peak': {'volume': [21180.0, 0.0], 'intensity': [21180.0, 0.0], 'number': 2, 'ref_id': None,
                              'proton2ppm': [4.391, None],
                              'hetero2ppm': [None, None],
                              'proton1ppm': [8.875, None],
                              'hetero1ppm': [119.917, None],
                              'reliable': False,
                              'proton1assignments': [{'type': 'automatic', 'atoms': [
                                  {'name': 'H', 'res': 'MET10', 'atom_id': 168, 'segid': '    ', 'hetero_name': 'N',
                                   'type': 'H'}]}],
                              'hetero1assignments': [{'type': 'automatic', 'atoms': [
                                  {'name': 'N', 'res': 'MET10', 'atom_id': 177, 'segid': '    ', 'hetero_name': None,
                                   'type': 'N'}]}],
                              'hetero2assignments': [],
                              'proton2assignments': [
                                  {'type': 'automatic', 'atoms': [
                                      {'name': 'HA', 'res': 'MET10', 'atom_id': 169, 'segid': '    ',
                                       'hetero_name': 'CA', 'type': 'H'}]},
                                  {'type': 'automatic', 'atoms': [
                                      {'name': 'HB', 'res': 'THR18', 'atom_id': 308, 'segid': '    ',
                                       'hetero_name': 'CB', 'type': 'H'}]},
                                  {'type': 'automatic', 'atoms': [
                                      {'name': 'HA', 'res': 'VAL37', 'atom_id': 628, 'segid': '    ',
                                       'hetero_name': 'CA', 'type': 'H'}]},
                                  {'type': 'automatic', 'atoms': [
                                      {'name': 'HA', 'res': 'ASP57', 'atom_id': 940, 'segid': '    ',
                                       'hetero_name': 'CA', 'type': 'H'}]},
                                  {'type': 'automatic', 'atoms': [
                                      {'name': 'HA', 'res': 'ASP83', 'atom_id': 1373, 'segid': '    ',
                                       'hetero_name': 'CA', 'type': 'H'}]}],
                              },

                 'analysis': {'average_distance': [None, None],
                              'lower_bound_violation': [None, None],
                              'is_violated': None,
                              'degree_of_violation': None,
                              'figure_of_merit': [None, None],
                              'model_peak_size': [None, None],
                              'upper_bound_violation': [None, None],
                              'contributions': [{'figure_of_merit': None,
                                                 'weight': None,
                                                 'average_distance': [None, None],
                                                 'contribution_id': 17174,
                                                 'spin_pairs': [{'id': 19893,
                                                                 'Atom2': {'name': 'HA', 'res': 'MET10', 'atom_id': 169,
                                                                           'segid': '    ', 'hetero_name': 'CA',
                                                                           'type': 'H'},
                                                                 'Atom1': {'name': 'H', 'res': 'MET10', 'atom_id': 168,
                                                                           'segid': '    ', 'hetero_name': 'N',
                                                                           'type': 'H'}}],
                                                 'type': 'fast_exchange'},
                                                {'figure_of_merit': None,
                                                 'weight': None,
                                                 'average_distance': [None, None],
                                                 'contribution_id': 17175,
                                                 'spin_pairs': [{'id': 19894,
                                                                 'Atom2': {'name': 'HB', 'res': 'THR18', 'atom_id': 308,
                                                                           'segid': '    ', 'hetero_name': 'CB',
                                                                           'type': 'H'},
                                                                 'Atom1': {'name': 'H', 'res': 'MET10', 'atom_id': 168,
                                                                           'segid': '    ', 'hetero_name': 'N',
                                                                           'type': 'H'}}],
                                                 'type': 'fast_exchange'},
                                                {'figure_of_merit': None, 'weight': None,
                                                 'average_distance': [None, None], 'contribution_id': 17176,
                                                 'spin_pairs': [{'id': 19895,
                                                                 'Atom2': {'name': 'HA', 'res': 'VAL37', 'atom_id': 628,
                                                                           'segid': '    ', 'hetero_name': 'CA',
                                                                           'type': 'H'},
                                                                 'Atom1': {'name': 'H', 'res': 'MET10', 'atom_id': 168,
                                                                           'segid': '    ', 'hetero_name': 'N',
                                                                           'type': 'H'}}], 'type': 'fast_exchange'},
                                                {'figure_of_merit': None, 'weight': None,
                                                 'average_distance': [None, None], 'contribution_id': 17177,
                                                 'spin_pairs': [{'id': 19896,
                                                                 'Atom2': {'name': 'HA', 'res': 'ASP57', 'atom_id': 940,
                                                                           'segid': '    ', 'hetero_name': 'CA',
                                                                           'type': 'H'},
                                                                 'Atom1': {'name': 'H', 'res': 'MET10', 'atom_id': 168,
                                                                           'segid': '    ', 'hetero_name': 'N',
                                                                           'type': 'H'}}], 'type': 'fast_exchange'},
                                                {'figure_of_merit': None, 'weight': None,
                                                 'average_distance': [None, None], 'contribution_id': 17178,
                                                 'spin_pairs': [{'id': 19897, 'Atom2': {'name': 'HA', 'res': 'ASP83',
                                                                                        'atom_id': 1373,
                                                                                        'segid': '    ',
                                                                                        'hetero_name': 'CA',
                                                                                        'type': 'H'},
                                                                 'Atom1': {'name': 'H', 'res': 'MET10', 'atom_id': 168,
                                                                           'segid': '    ', 'hetero_name': 'N',
                                                                           'type': 'H'}}], 'type': 'fast_exchange'}],
                              },

                 }
