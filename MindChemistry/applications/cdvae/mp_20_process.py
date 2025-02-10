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
""" script used for generate mp_20 dataset from raw data"""
import os
import logging
import argparse
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter


def mp_20_process():
    """process the mp_20 dataset"""
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
        logging.info("%s has been created", args.data_path)

    # read json file and transfer to pandasframe
    df = pd.read_json(args.init_path)
    df = df[["id", "formation_energy_per_atom", "band_gap", "pretty_formula",
             "e_above_hull", "elements", "atoms", "spacegroup_number"]]
    struct_list = []
    element_list = []
    # generate Structure from its df["atoms"] for each samples
    for struct in df["atoms"]:
        lattice = Lattice(struct["lattice_mat"], (False, False, False))
        pos = struct["coords"]
        species = struct["elements"]
        structure = Structure(lattice, species, pos)
        # save cif from Structure
        cif = CifWriter(structure)
        struct_list.append(cif.__str__())
        element_list.append(struct["elements"])

    # add cif to df
    df.insert(7, "cif", struct_list)
    df = df.drop("atoms", axis=1)
    df["elements"] = element_list

    # save to csv file
    # solit the dataset to train:val:test = 6:2:2
    train_df = df.iloc[:int(0.6 * len(df))]
    val_df = df.iloc[int(0.6 * len(df)):int(0.8 * len(df))]
    test_df = df.iloc[int(0.8 * len(df)):]
    train_df.to_csv(args.data_path+"/train.csv", index=False)
    val_df.to_csv(args.data_path+"/val.csv", index=False)
    test_df.to_csv(args.data_path+"/test.csv", index=False)
    logging.info("Finished!")


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_path", default="./data/mp_20.json",
                        help="path to the initial dataset file")
    parser.add_argument("--data_path", default="./data/mp_20",
                        help="path to save the processed dataset")
    args = parser.parse_args()
    mp_20_process()
