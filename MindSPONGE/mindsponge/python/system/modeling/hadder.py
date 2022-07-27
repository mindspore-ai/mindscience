# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#

# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
H-Adder Module.
"""
import sys
import numpy as np
from .add_missing_atoms import add_h
from .pdb_generator import gen_pdb
from .pdb_parser import read_pdb

hnames = {'ACE': {'CH3': ['H1', 'H2', 'H3']},
          'ALA': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB1', 'HB2', 'HB3']},
          'ARG': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'CD': ['HD2', 'HD3'],
                  'NE': ['HE'], 'NH1': ['HH11', 'HH12'], 'NH2': ['HH21', 'HH22']},
          'ASN': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'ND2': ['HD21', 'HD22']},
          'ASP': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3']},
          'CALA': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB1', 'HB2', 'HB3'], 'C': ['OXT']},
          'CARG': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'CD': ['HD2', 'HD3'],
                   'NE': ['HE'], 'NH1': ['HH11', 'HH12'], 'NH2': ['HH21', 'HH22'], 'C': ['OXT']},
          'CASN': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'ND2': ['HD21', 'HD22'], 'C': ['OXT']},
          'CASP': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'C': ['OXT']},
          'CCYS': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'SG': ['HG'], 'C': ['OXT']},
          'CGLN': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'NE2': ['HE21', 'HE22'],
                   'C': ['OXT']},
          'CGLU': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'C': ['OXT']},
          'CGLY': {'N': ['H'], 'CA': ['HA2', 'HA3'], 'C': ['OXT']},
          'CHID': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'ND1': ['HD1'], 'CE1': ['HE1'], 'CD2': ['HD2'],
                   'C': ['OXT']},
          'CHIS': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CE1': ['HE1'], 'NE2': ['HE2'], 'CD2': ['HD2'],
                   'C': ['OXT']},
          'CILE': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB'], 'CG2': ['HG21', 'HG22', 'HG23'], 'CG1': ['HG12', 'HG13'],
                   'CD1': ['HD11', 'HD12', 'HD13'], 'C': ['OXT']},
          'CLEU': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG'], 'CD1': ['HD11', 'HD12', 'HD13'],
                   'CD2': ['HD21', 'HD22', 'HD23'], 'C': ['OXT']},
          'CLYS': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'CD': ['HD2', 'HD3'],
                   'CE': ['HE2', 'HE3'], 'NZ': ['HZ1', 'HZ2', 'HZ3'], 'C': ['OXT']},
          'CMET': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'CE': ['HE1', 'HE2', 'HE3'],
                   'C': ['OXT']},
          'CPHE': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'CE1': ['HE1'], 'CZ': ['HZ'],
                   'CE2': ['HE2'], 'CD2': ['HD2'], 'C': ['OXT']},
          'CPRO': {'CD': ['HD2', 'HD3'], 'CG': ['HG2', 'HG3'], 'CB': ['HB2', 'HB3'], 'CA': ['HA'], 'C': ['OXT']},
          'CSER': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'OG': ['HG'], 'C': ['OXT']},
          'CTHR': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB'], 'CG2': ['HG21', 'HG22', 'HG23'], 'OG1': ['HG1'],
                   'C': ['OXT']},
          'CTRP': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'NE1': ['HE1'], 'CZ2': ['HZ2'],
                   'CH2': ['HH2'], 'CZ3': ['HZ3'], 'CE3': ['HE3'], 'C': ['OXT']},
          'CTYR': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'CE1': ['HE1'], 'OH': ['HH'],
                   'CE2': ['HE2'], 'CD2': ['HD2'], 'C': ['OXT']},
          'CVAL': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB'], 'CG1': ['HG11', 'HG12', 'HG13'],
                   'CG2': ['HG21', 'HG22', 'HG23'], 'C': ['OXT']},
          'CYS': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'SG': ['HG']},
          'GLN': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'NE2': ['HE21', 'HE22']},
          'GLU': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3']},
          'GLY': {'N': ['H'], 'CA': ['HA2', 'HA3']},
          'HID': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'ND1': ['HD1'], 'CE1': ['HE1'], 'CD2': ['HD2']},
          'HIS': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CE1': ['HE1'], 'NE2': ['HE2'], 'CD2': ['HD2']},
          'ILE': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB'], 'CG2': ['HG21', 'HG22', 'HG23'], 'CG1': ['HG12', 'HG13'],
                  'CD1': ['HD11', 'HD12', 'HD13']},
          'LEU': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG'], 'CD1': ['HD11', 'HD12', 'HD13'],
                  'CD2': ['HD21', 'HD22', 'HD23']},
          'LYS': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'CD': ['HD2', 'HD3'],
                  'CE': ['HE2', 'HE3'], 'NZ': ['HZ1', 'HZ2', 'HZ3']},
          'MET': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'], 'CE': ['HE1', 'HE2', 'HE3']},
          'NALA': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB1', 'HB2', 'HB3']},
          'NARG': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'],
                   'CD': ['HD2', 'HD3'], 'NE': ['HE'], 'NH1': ['HH11', 'HH12'], 'NH2': ['HH21', 'HH22']},
          'NASN': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'ND2': ['HD21', 'HD22']},
          'NASP': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3']},
          'NCYS': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'SG': ['HG']},
          'NGLN': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'],
                   'NE2': ['HE21', 'HE22']},
          'NGLU': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3']},
          'NGLY': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA2', 'HA3']},
          'NHID': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'ND1': ['HD1'], 'CE1': ['HE1'],
                   'CD2': ['HD2']},
          'NHIS': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CE1': ['HE1'], 'NE2': ['HE2'],
                   'CD2': ['HD2']},
          'NILE': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB'], 'CG2': ['HG21', 'HG22', 'HG23'],
                   'CG1': ['HG12', 'HG13'], 'CD1': ['HD11', 'HD12', 'HD13']},
          'NLEU': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG'],
                   'CD1': ['HD11', 'HD12', 'HD13'], 'CD2': ['HD21', 'HD22', 'HD23']},
          'NLYS': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'],
                   'CD': ['HD2', 'HD3'], 'CE': ['HE2', 'HE3'], 'NZ': ['HZ1', 'HZ2', 'HZ3']},
          'NME': {'N': ['H'], 'CH3': ['HH31', 'HH32', 'HH33']},
          'NMET': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CG': ['HG2', 'HG3'],
                   'CE': ['HE1', 'HE2', 'HE3']},
          'NPHE': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'CE1': ['HE1'],
                   'CZ': ['HZ'], 'CE2': ['HE2'], 'CD2': ['HD2']},
          'NPRO': {'N': ['H2', 'H3'], 'CD': ['HD2', 'HD3'], 'CG': ['HG2', 'HG3'], 'CB': ['HB2', 'HB3'], 'CA': ['HA']},
          'NSER': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'OG': ['HG']},
          'NTHR': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB'], 'CG2': ['HG21', 'HG22', 'HG23'],
                   'OG1': ['HG1']},
          'NTRP': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'NE1': ['HE1'],
                   'CZ2': ['HZ2'], 'CH2': ['HH2'], 'CZ3': ['HZ3'], 'CE3': ['HE3']},
          'NTYR': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'CE1': ['HE1'],
                   'OH': ['HH'], 'CE2': ['HE2'], 'CD2': ['HD2']},
          'NVAL': {'N': ['H1', 'H2', 'H3'], 'CA': ['HA'], 'CB': ['HB'], 'CG1': ['HG11', 'HG12', 'HG13'],
                   'CG2': ['HG21', 'HG22', 'HG23']},
          'PHE': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'CE1': ['HE1'], 'CZ': ['HZ'],
                  'CE2': ['HE2'], 'CD2': ['HD2']},
          'PRO': {'CD': ['HD2', 'HD3'], 'CG': ['HG2', 'HG3'], 'CB': ['HB2', 'HB3'], 'CA': ['HA']},
          'SER': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'OG': ['HG']},
          'THR': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB'], 'CG2': ['HG21', 'HG22', 'HG23'], 'OG1': ['HG1']},
          'TRP': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'NE1': ['HE1'], 'CZ2': ['HZ2'],
                  'CH2': ['HH2'], 'CZ3': ['HZ3'], 'CE3': ['HE3']},
          'TYR': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB2', 'HB3'], 'CD1': ['HD1'], 'CE1': ['HE1'], 'OH': ['HH'],
                  'CE2': ['HE2'], 'CD2': ['HD2']},
          'VAL': {'N': ['H'], 'CA': ['HA'], 'CB': ['HB'], 'CG1': ['HG11', 'HG12', 'HG13'],
                  'CG2': ['HG21', 'HG22', 'HG23']},
          }

hbond_type = {
    'ACE': {
        'CH3': np.array(['ch3', 'C', 'O'])
    },
    'ALA': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'CB']),
        'CB': np.array(['ch3', 'CA', 'C'])
    },
    'ARG': {
        'N': np.array(['dihedral', 'CA', 'CB']),
        'CA': np.array(['cc3', 'CB', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'NE']),
        'NE': np.array(['c6', 'CD', 'CZ']),
        'NH1': np.array([['dihedral', 'CZ', 'NH2'],
                         ['dihedral', 'CZ', 'NE']]),
        'NH2': np.array([['dihedral', 'CZ', 'NH1'],
                         ['dihedral', 'CZ', 'NE']])
    },
    'ASN': {
        'ND2': np.array(['c2h4', 'CG', 'OD1']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'CB'])
    },
    'ASP': {
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'CB'])
    },
    'CALA': {
        'N': np.array(['dihedral', 'CA', 'CB']),
        'CA': np.array(['cc3', 'CB', 'C']),
        'CB': np.array(['ch3', 'CA', 'C']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CARG': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'NE']),
        'NE': np.array(['dihedral', 'CZ', 'NH1']),
        'NH1': np.array([['dihedral', 'CZ', 'NH2'],
                         ['dihedral', 'CZ', 'NE']]),
        'NH2': np.array([['dihedral', 'CZ', 'NH1'],
                         ['dihedral', 'CZ', 'NE']]),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CASN': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'ND2': np.array(['c2h4', 'CG', 'OD1']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CASP': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CCYS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'SG']),
        'SG': np.array(['dihedral', 'CB', 'CA']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CGLN': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'NE2': np.array(['c2h4', 'CD', 'OE1']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CGLU': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CGLY': {
        'CA': np.array(['c2h2', 'N', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CHID': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD2': np.array(['c6', 'CG', 'NE2']),
        'ND1': np.array(['c6', 'CG', 'CE1']),
        'CE1': np.array(['c6', 'ND1', 'NE2']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CHIS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD2': np.array(['c6', 'CG', 'NE2']),
        'NE2': np.array(['c6', 'CD2', 'CE1']),
        'CE1': np.array(['c6', 'ND1', 'NE2']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CILE': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG1', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA']),
        'CG1': np.array(['c2h2', 'CB', 'CD1']),
        'CD1': np.array(['ch3', 'CG1', 'CB']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CLEU': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['cc3', 'CD2', 'CB']),
        'CD2': np.array(['ch3', 'CG', 'CD1']),
        'CD1': np.array(['ch3', 'CG', 'CB']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CLYS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'CE']),
        'CE': np.array(['c2h2', 'CD', 'NZ']),
        'NZ': np.array(['ch3', 'CE', 'CD']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CMET': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'SD']),
        'CE': np.array(['ch3', 'SD', 'CG']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CPHE': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'CE1']),
        'CD2': np.array(['c6', 'CG', 'CE2']),
        'CE1': np.array(['c6', 'CD1', 'CZ']),
        'CE2': np.array(['c6', 'CD2', 'CZ']),
        'CZ': np.array(['c6', 'CE2', 'CE1']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CPRO': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'N']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CSER': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'OG']),
        'OG': np.array(['dihedral', 'CB', 'CA']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CTHR': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG2', 'OG1']),
        'OG1': np.array(['dihedral', 'CB', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CTRP': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'NE1']),
        'NE1': np.array(['c6', 'CD1', 'CE2']),
        'CZ2': np.array(['c6', 'CE2', 'CH2']),
        'CH2': np.array(['c6', 'CZ2', 'CZ3']),
        'CZ3': np.array(['c6', 'CH2', 'CE3']),
        'CE3': np.array(['c6', 'CD2', 'CZ3']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CTYR': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'CE1']),
        'CE1': np.array(['c6', 'CD1', 'CZ']),
        'CD2': np.array(['c6', 'CG', 'CE2']),
        'CE2': np.array(['c6', 'CD2', 'CZ']),
        'OH': np.array(['dihedral', 'CZ', 'CE2']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CVAL': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG2', 'CA']),
        'CG1': np.array(['ch3', 'CB', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA']),
        'C': np.array(['dihedral', 'CA', 'N'])
    },
    'CYS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'SG']),
        'SG': np.array(['dihedral', 'CB', 'CA'])
    },
    'GLN': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'NE2': np.array(['c2h4', 'CD', 'OE1'])
    },
    'GLU': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD'])
    },
    'GLY': {
        'CA': np.array(['c2h2', 'N', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
    },
    'HID': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD2': np.array(['c6', 'CG', 'NE2']),
        'ND1': np.array(['c6', 'CG', 'CE1']),
        'CE1': np.array(['c6', 'ND1', 'NE2'])
    },
    'HIS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD2': np.array(['c6', 'CG', 'NE2']),
        'NE2': np.array(['c6', 'CD2', 'CE1']),
        'CE1': np.array(['c6', 'ND1', 'NE2'])
    },
    'ILE': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG1', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA']),
        'CG1': np.array(['c2h2', 'CB', 'CD1']),
        'CD1': np.array(['ch3', 'CG1', 'CB'])
    },
    'LEU': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['cc3', 'CD2', 'CB']),
        'CD2': np.array(['ch3', 'CG', 'CD1']),
        'CD1': np.array(['ch3', 'CG', 'CB'])
    },
    'LYS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'CE']),
        'CE': np.array(['c2h2', 'CD', 'NZ']),
        'NZ': np.array(['ch3', 'CE', 'CD'])
    },
    'MET': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'SD']),
        'CE': np.array(['ch3', 'SD', 'CG'])
    },
    'NALA': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'CB']),
        'CB': np.array(['ch3', 'CA', 'C'])
    },
    'NARG': {
        'N': np.array(['ch3', 'CA', 'C']),
        'CA': np.array(['cc3', 'CB', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'NE']),
        'NE': np.array(['dihedral', 'CZ', 'NH1']),
        'NH1': np.array([['dihedral', 'CZ', 'NH2'],
                         ['dihedral', 'CZ', 'NE']]),
        'NH2': np.array([['dihedral', 'CZ', 'NH1'],
                         ['dihedral', 'CZ', 'NE']])
    },
    'NASN': {
        'ND2': np.array(['c2h4', 'CG', 'OD1']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C'])
    },
    'NASP': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG'])
    },
    'NCYS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'SG']),
        'SG': np.array(['dihedral', 'CB', 'CA'])
    },
    'NGLN': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'NE2': np.array(['c2h4', 'CD', 'OE1'])
    },
    'NGLU': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD'])
    },
    'NGLY': {
        'CA': np.array(['c2h2', 'N', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
    },
    'NHID': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD2': np.array(['c6', 'CG', 'NE2']),
        'ND1': np.array(['c6', 'CG', 'CE1']),
        'CE1': np.array(['c6', 'ND1', 'NE2'])
    },
    'NHIS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD2': np.array(['c6', 'CG', 'NE2']),
        'NE2': np.array(['c6', 'CD2', 'CE1']),
        'CE1': np.array(['c6', 'ND1', 'NE2'])
    },
    'NILE': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG1', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA']),
        'CG1': np.array(['c2h2', 'CB', 'CD1']),
        'CD1': np.array(['ch3', 'CG1', 'CB'])
    },
    'NLEU': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['cc3', 'CD2', 'CB']),
        'CD2': np.array(['ch3', 'CG', 'CD1']),
        'CD1': np.array(['ch3', 'CG', 'CB'])
    },
    'NLYS': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'CE']),
        'CE': np.array(['c2h2', 'CD', 'NZ']),
        'NZ': np.array(['ch3', 'CE', 'CD'])
    },
    'NMET': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'SD']),
        'CE': np.array(['ch3', 'SD', 'CG'])
    },
    'NPHE': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'CE1']),
        'CD2': np.array(['c6', 'CG', 'CE2']),
        'CE1': np.array(['c6', 'CD1', 'CZ']),
        'CE2': np.array(['c6', 'CD2', 'CZ']),
        'CZ': np.array(['c6', 'CE2', 'CE1']),
    },
    'NPRO': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'N']),
        'N': np.array(['c2h2', 'CA', 'CD'])
    },
    'NSER': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'OG']),
        'OG': np.array(['dihedral', 'CB', 'CA'])
    },
    'NTHR': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG2', 'OG1']),
        'OG1': np.array(['dihedral', 'CB', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA'])
    },
    'NTRP': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'NE1']),
        'NE1': np.array(['c6', 'CD1', 'CE2']),
        'CZ2': np.array(['c6', 'CE2', 'CH2']),
        'CH2': np.array(['c6', 'CZ2', 'CZ3']),
        'CZ3': np.array(['c6', 'CH2', 'CE3']),
        'CE3': np.array(['c6', 'CD2', 'CZ3'])
    },
    'NTYR': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'CE1']),
        'CE1': np.array(['c6', 'CD1', 'CZ']),
        'CD2': np.array(['c6', 'CG', 'CE2']),
        'CE2': np.array(['c6', 'CD2', 'CZ']),
        'OH': np.array(['dihedral', 'CZ', 'CE2'])
    },
    'NVAL': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['ch3', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG2', 'CA']),
        'CG1': np.array(['ch3', 'CB', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA'])
    },
    'PHE': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'CE1']),
        'CD2': np.array(['c6', 'CG', 'CE2']),
        'CE1': np.array(['c6', 'CD1', 'CZ']),
        'CE2': np.array(['c6', 'CD2', 'CZ']),
        'CZ': np.array(['c6', 'CE2', 'CE1']),
    },
    'PRO': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CG': np.array(['c2h2', 'CB', 'CD']),
        'CD': np.array(['c2h2', 'CG', 'N']),
    },
    'SER': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'OG']),
        'OG': np.array(['dihedral', 'CB', 'CA'])
    },
    'THR': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG2', 'OG1']),
        'OG1': np.array(['dihedral', 'CB', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA'])
    },
    'TRP': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'NE1']),
        'NE1': np.array(['c6', 'CD1', 'CE2']),
        'CZ2': np.array(['c6', 'CE2', 'CH2']),
        'CH2': np.array(['c6', 'CZ2', 'CZ3']),
        'CZ3': np.array(['c6', 'CH2', 'CE3']),
        'CE3': np.array(['c6', 'CD2', 'CZ3'])
    },
    'TYR': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['c2h2', 'CA', 'CG']),
        'CD1': np.array(['c6', 'CG', 'CE1']),
        'CE1': np.array(['c6', 'CD1', 'CZ']),
        'CD2': np.array(['c6', 'CG', 'CE2']),
        'CE2': np.array(['c6', 'CD2', 'CZ']),
        'OH': np.array(['dihedral', 'CZ', 'CE2'])
    },
    'VAL': {
        'CA': np.array(['cc3', 'CB', 'C']),
        'N': np.array(['dihedral', 'CA', 'C']),
        'CB': np.array(['cc3', 'CG2', 'CA']),
        'CG1': np.array(['ch3', 'CB', 'CA']),
        'CG2': np.array(['ch3', 'CB', 'CA'])
    },
    'NME': None
}

addhs = {'c6': 1,
         'dihedral': 1,
         'c2h4': 2,
         'ch3': 3,
         'cc3': 1,
         'c2h2': 2}

sys.path.append('../')


def AddHydrogen(pdb_in, pdb_out):
    """ The API function for adding Hydrogen.
    Args:
        pdb_in(str): The input pdb file name, absolute file path is suggested.
        pdb_out(str): The output pdb file name, absolute file path is suggested.
    """
    pdb_name = pdb_in
    new_pdb_name = pdb_out
    atom_names, res_names, _, crds, _, _, _, _, _, \
        _, _, _, _, _ = read_pdb(pdb_name, ignoreh=True)

    for i, res in enumerate(res_names):
        res = 'N' * (i == 0) + res
        res = 'C' * (i == (len(res_names) - 1)) + res
        h_names = []
        crds[i] = np.array(crds[i])
        for atom in atom_names[i]:
            if atom == 'C' and len(res) == 4 and res.startswith(
                    'C') and np.isin(atom_names[i], 'OXT').sum() == 1:
                continue
            if atom in hbond_type[res].keys() and len(
                    hbond_type[res][atom].shape) == 1:
                addh_type = hbond_type[res][atom][0]
                for name in hnames[res][atom]:
                    h_names.append(name)
                try:
                    m = np.where(np.array(atom_names[i]) == [atom])[0][0]
                    n = np.where(
                        np.array(
                            atom_names[i]) == hbond_type[res][atom][1])[0][0]
                    o = np.where(
                        np.array(
                            atom_names[i]) == hbond_type[res][atom][2])[0][0]
                except IndexError as e:
                    raise ValueError(
                        'Some heavy atoms are missing in given pdb file.') from e
                new_crd = add_h(np.array(crds[i]),
                                atype=addh_type,
                                i=m,
                                j=n,
                                k=o)
                crds[i] = np.append(crds[i], new_crd, axis=0)
            elif atom in hbond_type[res].keys():
                for j, hbond in enumerate(hbond_type[res][atom]):
                    addh_type = hbond[0]
                    h_names.append(hnames[res][atom][j])
                    try:
                        m = np.where(np.array(atom_names[i]) == [atom])[0][0]
                        n = np.where(np.array(atom_names[i]) == hbond[1])[0][0]
                        o = np.where(np.array(atom_names[i]) == hbond[2])[0][0]
                    except IndexError as e:
                        raise ValueError(
                            'Some heavy atoms are missing in given pdb file.') from e
                    new_crd = add_h(np.array(crds[i]),
                                    atype=addh_type,
                                    i=m,
                                    j=n,
                                    k=o)
                    crds[i] = np.append(crds[i], new_crd, axis=0)
            else:
                continue
        for name in h_names:
            atom_names[i].append(name)

    new_crds = crds[0]
    for crd in crds[1:]:
        new_crds = np.append(new_crds, crd, axis=0)

    new_atom_names = np.array(atom_names[0])
    for name in atom_names[1:]:
        new_atom_names = np.append(new_atom_names, name)

    new_res_names = []
    new_res_ids = []
    for i, crd in enumerate(crds):
        for _ in range(crd.shape[0]):
            new_res_names.append(res_names[i])
            new_res_ids.append(i + 1)

    gen_pdb(new_crds[None, :], new_atom_names,
            new_res_names, new_res_ids, new_pdb_name)
    print('1 H-Adding task complete.')

def ReadPdbByMindsponge(pdb_name, addh):
    if addh:
        t_name = pdb_name.replace('.pdb', '_addH_by_mindsponge.pdb')
        AddHydrogen(pdb_name, t_name)
        return read_pdb(t_name)

    return read_pdb(pdb_name)
