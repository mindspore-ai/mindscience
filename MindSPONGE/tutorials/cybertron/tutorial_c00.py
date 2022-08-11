# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
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
Cybertron tutorial 00: The preprocessing of dataset
                       The NPZ file can be download from:
                       dataset_qm9.npz: http://gofile.me/6Utp7/tJ5hoDIAo
                       ethanol_dft.npz: http://gofile.me/6Utp7/hbQBofAFM
"""

import sys
import numpy as np

if __name__ == '__main__':

    sys.path.append('..')

    from cybertron.dataset import DatasetProcessor

    qm9_file = 'dataset_qm9.npz'
    qm9_data = np.load(qm9_file)

    for k, v in qm9_data.items():
        print(k, v.shape)

    atom_ref = qm9_data['atom_ref']
    ref = np.zeros((atom_ref.shape[0], 6))
    atom_ref = np.concatenate((ref, atom_ref), axis=-1)

    ds_qm9 = DatasetProcessor(
        name='qm9',
        atom_types=qm9_data['Z'],
        position=qm9_data['R'],
        label=qm9_data['properties'][:, 3:],
        length_unit='A',
        energy_unit='Ha',
        type_ref=atom_ref,
    )

    exc_idx = np.concatenate(
        (qm9_data['excluded'], qm9_data['uncharacterized']), -1) - 1
    ds_qm9.exclude_data(exc_idx)

    ds_qm9.convert_units('nm', 'kj/mol', [2, 3, 4, 6, 7, 8, 9, 10])

    ds_qm9.shuffle_dataset()

    mode = ['graph', 'graph', 'graph', 'graph', 'graph', 'graph',
            'graph', 'atomwise', 'atomwise', 'atomwise', 'atomwise', 'graph']
    ds_qm9.save_dataset('dataset_qm9', 1024, 128, 1024,
                        norm_train=True, norm_valid=True, norm_test=True, mode=mode)
    ds_qm9.save_dataset('dataset_qm9', 1024, 128, 1024, norm_train=False,
                        norm_valid=False, norm_test=False, mode=mode)

    md17_data = np.load('ethanol_dft.npz')

    for k, v in md17_data.items():
        print(k, v.shape)

    ds_md17 = DatasetProcessor(
        name='md17',
        atom_types=md17_data['z'],
        position=md17_data['R'],
        label=md17_data['E'],
        force=md17_data['F'],
        length_unit='A',
        energy_unit='kcal/mol',
    )

    ds_md17.convert_units('nm', 'kj/mol')

    ds_md17.shuffle_dataset()

    ds_md17.save_dataset('dataset_ethanol', 1024, 128, 1024, mode='atomwise')
