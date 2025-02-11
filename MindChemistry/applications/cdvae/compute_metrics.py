# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Compute metrics
"""
from collections import Counter
import logging
import argparse
import os
import json

import numpy as np
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
from mindchemistry.cell.gemnet.data_utils import StandardScaler
from src.metrics_utils import (
    smact_validity, structure_validity, get_fp_pdist,
    get_crystals_list, compute_cov)

CRYSTALNNFP = CrystalNNFingerprint.from_preset("ops")
COMPFP = ElementProperty.from_preset("magpie")

COV_CUTOFFS = {
    "mp_20": {"struct": 0.4, "comp": 10.},
    "carbon_24": {"struct": 0.2, "comp": 4.},
    "perov_5": {"struct": 0.2, "comp": 4},
}
# threshold for coverage metrics, olny struct distance and comp distance
# smaller than the threshold will be counted as covered.


class Crystal():
    """get crystal structures"""

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict["frac_coords"]
        self.atom_types = crys_array_dict["atom_types"]
        self.lengths = crys_array_dict["lengths"]
        self.angles = crys_array_dict["angles"]
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        """get structure"""
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except (ValueError, AttributeError, TypeError):
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = "unrealistically_small_lattice"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        """get fingerprints"""
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = COMPFP.featurize(comp)
        try:
            site_fps = [CRYSTALNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except (ValueError, AttributeError, TypeError):
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval():
    """reconstruction evaluation result"""

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        """get match rate and rms, match rate shows how much rate of the prediction has
        the same structure as the ground truth."""
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except (ValueError, AttributeError, TypeError):
                return None
        validity = [c.valid for c in self.preds]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(x is not None for x in rms_dists) / len(self.preds)
        mean_rms_dist = np.array(
            [x for x in rms_dists if x is not None]).mean()
        return {"match_rate": match_rate,
                "rms_dist": mean_rms_dist}

    def get_metrics(self):
        return self.get_match_rate_and_rms()


class GenEval():
    """Generation Evaluation result"""

    def __init__(self, pred_crys, gt_crys, comp_scaler, n_samples=10, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name
        self.comp_scaler = comp_scaler

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f"not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}")

    def get_validity(self):
        """
        Compute Validity, which means whether the structure is reasonable and phyically stable
        in both composition and structure.
        """
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {"comp_valid": comp_valid,
                "struct_valid": struct_valid,
                "valid": valid}

    def get_comp_diversity(self):
        """the earth mover’s distance (EMD) between the property distribution of
        generated materials and test materials.
        """
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = self.comp_scaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {"comp_div": comp_div}

    def get_struct_diversity(self):
        return {"struct_div": get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {"wdist_density": wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {"wdist_num_elems": wdist_num_elems}

    def get_coverage(self):
        """measure the similarity between ensembles of generated materials
        and ground truth materials. COV-R measures the percentage of
        ground truth materials being correctly predicted.
        """
        cutoff_dict = COV_CUTOFFS[self.eval_model_name]
        (cov_metrics_dict, _) = compute_cov(
            self.crys, self.gt_crys, self.comp_scaler,
            struc_cutoff=cutoff_dict["struct"],
            comp_cutoff=cutoff_dict["comp"])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        print(f'evaluation metrics:{metrics}')
        metrics.update(self.get_coverage())
        return metrics


def get_crystal_array_list(data, gt_data=None, ground_truth=False):
    """get crystal array list"""
    crys_array_list = get_crystals_list(
        np.concatenate(data["frac_coords"], axis=1).squeeze(0),
        np.concatenate(data["atom_types"], axis=1).squeeze(0),
        np.concatenate(data["lengths"], axis=1).squeeze(0),
        np.concatenate(data["angles"], axis=1).squeeze(0),
        np.concatenate(data["num_atoms"], axis=1).squeeze(0))

    # if "input_data_batch" in data:
    if ground_truth:
        true_crystal_array_list = get_crystals_list(
            np.concatenate(gt_data["frac_coords"], axis=0).squeeze(),
            np.concatenate(gt_data["atom_types"], axis=0).squeeze(),
            np.concatenate(gt_data["lengths"],
                           axis=0).squeeze().reshape(-1, 3),
            np.concatenate(gt_data["angles"], axis=0).squeeze().reshape(-1, 3),
            np.concatenate(gt_data["num_atoms"], axis=0).squeeze())
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def main(args):
    all_metrics = {}
    eval_model_name = args.dataset

    if "recon" in args.tasks:
        out_data = np.load(args.eval_path+"/eval_recon.npy",
                           allow_pickle=True).item()
        gt_data = np.load(args.eval_path+"/gt_recon.npy",
                          allow_pickle=True).item()
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            out_data, gt_data, ground_truth=True)
        pred_crys = p_map(Crystal, crys_array_list)
        gt_crys = p_map(Crystal, true_crystal_array_list)

        rec_evaluator = RecEval(pred_crys, gt_crys)
        recon_metrics = rec_evaluator.get_metrics()
        all_metrics.update(recon_metrics)

    if "gen" in args.tasks:
        out_data = np.load(args.eval_path+"/eval_gen.npy",
                           allow_pickle=True).item()
        gt_data = np.load(args.eval_path+"/gt_recon.npy",
                          allow_pickle=True).item()
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            out_data, gt_data, ground_truth=True)

        gen_crys = p_map(Crystal, crys_array_list)
        gt_crys = p_map(Crystal, true_crystal_array_list)
        gt_comp_fps = [c.comp_fp for c in gt_crys]
        gt_fp_np = np.array(gt_comp_fps)
        comp_scaler = StandardScaler(replace_nan_token=0.)
        comp_scaler.fit(gt_fp_np)

        gen_evaluator = GenEval(
            gen_crys, gt_crys, comp_scaler, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    logging.info(all_metrics)

    if args.label == "":
        metrics_out_file = "eval_metrics.json"
    else:
        metrics_out_file = f"eval_metrics_{args.label}.json"
    metrics_out_file = os.path.join(args.eval_path, metrics_out_file)

    with open(metrics_out_file, "w") as f:
        json.dump(all_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="perov_5")
    parser.add_argument("--eval_path", default="./eval_result")
    parser.add_argument("--label", default="")
    parser.add_argument("--tasks", nargs="+", default=["recon"])
    main_args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    main(main_args)
