# Copyright 2023 Huawei Technologies Co., Ltd
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
"extract_restraints"
import argparse
import os
import stat
import numpy as np
from mindsponge.common import residue_constants
from mindsponge.common.protein import from_pdb_string

parser = argparse.ArgumentParser(description='extract_restraints.py')
parser.add_argument('--pdb_path', type=str, help='Location of training pdb file.')
parser.add_argument('--output_file', type=str, help='output file')

arguments = parser.parse_args()


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    is_gly = np.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = np.where(np.tile(is_gly[..., None].astype("int32"), \
                                   [1,] * len(is_gly.shape) + [3,]).astype("bool"), \
                           all_atom_positions[..., ca_idx, :], \
                           all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def preprocess_contact_info(gt_features):
    '''preprocess_contact_info'''
    remote_residue_threshold = 6

    # contact_mask
    pseudo_beta_mask = gt_features["pseudo_beta_mask"]
    contact_mask = pseudo_beta_mask[:, None] * pseudo_beta_mask[None]

    seq_len = pseudo_beta_mask.shape[0]
    if seq_len > remote_residue_threshold + 1:
        diagonal_mask = np.eye(seq_len)
        for i in range(1, remote_residue_threshold + 1):
            diagonal_mask += np.eye(seq_len, seq_len, i)
            diagonal_mask += np.eye(seq_len, seq_len, -i)
        diagonal_mask = diagonal_mask < 0.5
        contact_mask *= diagonal_mask
    else:
        contact_mask *= 0

    gt_features["contact_mask"] = contact_mask

    return gt_features


def generate_gaussian_filter(kernel_size, sigma=1, muu=0):
    '''generate_gaussian_filter'''
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    normal = 1 / (2.0 * np.pi * sigma ** 2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal
    gauss /= np.max(gauss)
    return gauss


def smoothing(x, margin_size=2):
    '''smoothing'''
    kernel_size = 2 * margin_size + 1
    x_pad = np.pad(x, ((margin_size, margin_size), (margin_size, margin_size)), constant_values=0)
    gaussian_filter = generate_gaussian_filter(kernel_size, sigma=1, muu=0)
    index = np.where(x_pad > 0)
    for i, j in np.array(index).transpose():
        x_pad[i - margin_size:i + margin_size + 1, j - margin_size:j + margin_size + 1] = \
            np.maximum(gaussian_filter, x_pad[i - margin_size:i + margin_size + 1, \
                                        j - margin_size:j + margin_size + 1])
    return x_pad[margin_size: - margin_size, margin_size: -margin_size]


def generate_contact_info(gt_features):
    '''generate_contact_info'''
    true_pseudo_beta = gt_features["pseudo_beta"]
    sequence_length = true_pseudo_beta.shape[0]
    contact_mask_input = np.zeros((sequence_length, sequence_length)).astype(np.float32)

    np.random.seed(0)
    try:
        constraints_num = 200
        print("num", constraints_num)

        good_constraints_num_ratio = 1.0

        constraints_num1 = int(constraints_num * good_constraints_num_ratio)

        contact_mask = gt_features["contact_mask"]

        true_cb_distance = np.sqrt((np.square(true_pseudo_beta[None] - true_pseudo_beta[:, None])).sum(-1) + 1e-8)

        # positive sample
        probs = (1.0 - 1 / (1 + np.exp(-400000.0 * (true_cb_distance - 8))))
        randoms = np.random.random(probs.shape)
        selected_index = np.where((probs > randoms) * contact_mask > 0.5)

        selected_index = np.array(selected_index).transpose()
        np.random.shuffle(selected_index)
        final_selected_index = selected_index[:constraints_num1]
        result = ""
        for str1 in final_selected_index:
            temp = f"{str1[0]} {str1[1]}"
            result += temp
            result += "\n"

        os_flags = os.O_RDWR | os.O_CREAT
        os_modes = stat.S_IRWXU
        with os.fdopen(os.open(arguments.output_file, os_flags, os_modes), 'w', encoding='utf-8') as fout:
            fout.write(result)
        real_good_constraints_num = final_selected_index.shape[0]

        for i in range(final_selected_index.shape[0]):
            contact_mask_input[final_selected_index[i][0], final_selected_index[i][1]] = 1

        bad_constraints_num = int(
            real_good_constraints_num * (1 - good_constraints_num_ratio) / good_constraints_num_ratio)
        probs = 1 / (1 + np.exp(-400000.0 * (true_cb_distance - 12)))
        randoms = np.random.random(probs.shape)
        selected_index = np.where((probs > randoms) * contact_mask > 0.5)

        selected_index = np.array(selected_index).transpose()
        np.random.shuffle(selected_index)
        final_selected_index = selected_index[:bad_constraints_num]

        print("constraints_num, good_constraints_num_ratio, real_good_constraints_num, bad_constraints_num",
              constraints_num, good_constraints_num_ratio, real_good_constraints_num, bad_constraints_num)

        for i in range(final_selected_index.shape[0]):
            print('hello world', i)
            contact_mask_input[final_selected_index[i][0], final_selected_index[i][1]] = 1



    except Exception as e:
        print("error while generating contact info", e)

    np.random.seed()


def select_contacts(pdb_file_path):
    '''select_contacts'''
    with open(pdb_file_path, 'r') as f:
        prot_pdb = from_pdb_string(f.read())
    aatype = prot_pdb.aatype
    atom37_positions = prot_pdb.atom_positions.astype(np.float32)
    atom37_mask = prot_pdb.atom_mask.astype(np.float32)

    # get pseudo_beta, pseudo_beta_mask
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(aatype, atom37_positions, atom37_mask)

    # combine all gt features
    gt_features = {'pseudo_beta': pseudo_beta, 'pseudo_beta_mask': pseudo_beta_mask}

    gt_features = preprocess_contact_info(gt_features)

    generate_contact_info(gt_features)


if __name__ == "__main__":
    select_contacts(arguments.pdb_path)
