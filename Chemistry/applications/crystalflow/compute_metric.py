# Copyright 2024 Huawei Technologies Co., Ltd
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
"""compute metric file"""
import itertools
import json
import os
import pickle
from collections import Counter
from pathlib import Path
import argparse
import yaml

import numpy as np
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
import smact
from smact.screening import pauling_test
from tqdm import trange

from models.infer_utils import chemical_symbols

matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
crystalnn_fp = CrystalNNFingerprint.from_preset("ops")
comp_fp = ElementProperty.from_preset('magpie')


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    """Smact validity. See details in the paper Crystal Diffution Variational Autoencoder and
       its codebase.
    """
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, _ = smact.neutral_ratios(ox_states,
                                       stoichs=stoichs,
                                       threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_ok = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_ok = True
            else:
                electroneg_ok = True
            if electroneg_ok:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    """Structure validity. See details in the paper Crystal Diffution Variational Autoencoder and
       its codebase.
    """
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1 or max(
            crystal.lattice.abc) > 40:
        return False

    return True

class Crystal:
    """Strict crystal validity. See details in the paper CDVAE `Crystal
    Diffution Variational Autoencoder` and
    its codebase. We adopt the same evaluation metric criteria as CDVAE.
    """

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = np.argmax(self.atom_types, axis=-1) + 1
            self.atom_types = np.argmax(self.atom_types, axis=-1) + 1

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        """get_structure
        """
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(
                self.angles).any() or np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
        else:
            try:
                self.structure = Structure(lattice=Lattice.from_parameters(
                    *(self.lengths.tolist() + self.angles.tolist())),
                                           species=self.atom_types,
                                           coords=self.frac_coords,
                                           coords_are_cartesian=False)
                self.constructed = True
            # pylint: disable=W0703
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        """get_composition
        """
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        """get_validity
        """
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        """get_fingerprints
        """
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = comp_fp.featurize(comp)
        try:
            site_fps = [
                crystalnn_fp.featurize(self.structure, i)
                for i in range(len(self.structure))
            ]
        # pylint: disable=W0703
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def get_rms(pred_struc_list, gt_struc: Structure, num_eval, np_list):
    """Calculate the rms distance between the ground truth and predicted crystal structures.

    Args:
        pred_struc_list (List[Structure]): The crystals generated by diffution model
            in the form of Structure.
        gt_struc (Structure): The ground truth crystal.
        num_eval (int): Specify that the first N items in the predicted List of crystal structures
            participate in the evaluationo.
        np_list (List[Dict]): The crystals generated by diffution model in the form of Dict.
    """

    def process_one(pred_struc: Structure):
        try:
            if not pred_struc.is_valid():
                return None
            rms_dist = matcher.get_rms_dist(pred_struc, gt_struc)
            rms_dist = None if rms_dist is None else rms_dist[0]
            tune_rms = rms_dist
        # pylint: disable=W0703
        except Exception:
            tune_rms = None
        return tune_rms

    min_rms = None
    min_struc = None
    for i, struct in enumerate(pred_struc_list):
        if i == num_eval:
            break
        rms = process_one(struct)
        if rms is not  None and (min_rms is None or min_rms > rms):
            min_rms = rms
            min_struc = np_list[i]
    return min_rms, min_struc


def get_struc_from_np_list(np_list):
    """convert the crystal in the form of Dict to pymatgen.Structure
    """
    result = []
    for cry_array in np_list:
        try:
            struct = Structure(lattice=Lattice.from_parameters(
                *(cry_array['lengths'].tolist() +
                  cry_array['angles'].tolist())),
                               species=cry_array['atom_types'],
                               coords=cry_array['frac_coords'],
                               coords_are_cartesian=False)
        # pylint: disable=W0703
        except Exception:
            print('Warning: One anomalous crystal structure has captured and removed. ')
            struct = None

        result.append(struct)
    return result

def main(args):
    """main
    """
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    eval_file = config['test']['eval_save_path']
    num_eval = config['test']['num_eval']
    output_path = config['test']['metric_dir']

    with open(eval_file, 'rb') as f:
        eval_dict = pickle.load(f)

    pred_list = eval_dict['pred']
    gt_list = eval_dict['gt']
    gt_list = get_struc_from_np_list(gt_list)
    rms = []

    # calculate rmsd
    for i in trange(len(gt_list)):
        pred_struc = get_struc_from_np_list(pred_list[i])
        gt_struc = gt_list[i]
        rms_single, struc_single = get_rms(pred_struc, gt_struc, num_eval,
                                           pred_list[i])
        rms.append((rms_single, struc_single))

    rms, struc_list = zip(*rms)

    # Remove the ones with RMSD as None, and store the valid structures in the list valid_crys.
    rms_np = []
    valid_crys = []
    for i, rms_per in enumerate(rms):
        if rms_per is not None:
            rms_np.append(rms_per)
            valid_crys.append(struc_list[i])

    # Conduct rigorous structural verification, specifically through verification using the Crystal class.
    print('Using the Crystal class for validity checks')
    valid_list = p_map(lambda x: Crystal(x).valid, valid_crys)
    rms_np_strict = []
    for i, is_valid in enumerate(valid_list):
        if is_valid:
            rms_np_strict.append(rms_np[i])

    rms_np = np.array(rms_np_strict)
    rms_valid_index = np.array([x is not None for x in rms_np_strict])

    match_rate = rms_valid_index.sum() / len(gt_list)
    rms = rms_np[rms_valid_index].mean()

    print('match_rate: ', match_rate)
    print('rms: ', rms)

    all_metrics = {'match_rate': match_rate, 'rms_dist': rms}

    if Path(output_path).exists():
        metrics_out_file = f'eval_metrics_{num_eval}.json'
        metrics_out_file = os.path.join(output_path, metrics_out_file)

        # only overwrite metrics computed in the new run.
        if Path(metrics_out_file).exists():
            with open(metrics_out_file, 'r') as f:
                written_metrics = json.load(f)
                if isinstance(written_metrics, dict):
                    written_metrics.update(all_metrics)
                else:
                    with open(metrics_out_file, 'w') as f:
                        json.dump(all_metrics, f)
            if isinstance(written_metrics, dict):
                with open(metrics_out_file, 'w') as f:
                    json.dump(written_metrics, f)
        else:
            with open(metrics_out_file, 'w') as f:
                json.dump(all_metrics, f)
    else:
        print('Warning: The metric result file path is not specified')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    main_args = parser.parse_args()
    main(main_args)
