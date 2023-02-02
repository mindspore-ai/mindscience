# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""predict"""
import os
import warnings
import argparse

import json
import mindspore as ms

from model.predictor import Predictor

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # Predict settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cm', '--cmap', type=str,
                        help="Protein contact map (in *npz file format) or protein PDB file to be annotated.")
    parser.add_argument('--npz_dir', type=str, help="Directory with *npz files of protein contact map.")
    parser.add_argument('--pdb_dir', type=str, help="Directory with PDB files of predicted Rosetta/DMPFold structures.")
    parser.add_argument('--save_path', type=str, default='./output',
                        help="Folder to save output files.")
    parser.add_argument('-ont', '--ontology', type=str, default=['mf'], nargs='+', required=True,
                        choices=['mf', 'bp', 'cc', 'ec'],
                        help="Gene Ontology/Enzyme Commission.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--evaluation_path', type=str,
                        help='calculate the precision and recall of different threshold on val dataset')
    parser.add_argument('-device', '--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: GPU)')
    parser.add_argument('-id', '--device_id', type=int, default=0,
                        help='device id where the data will be set in (default: 0)')
    args = parser.parse_args()

    ms.set_context(device_target=args.device_target, device_id=args.device_id, mode=ms.GRAPH_MODE)
    print("Device target : {}\nDevice id : {}\nMode : {}".
          format(args.device_target, args.device_id, ms.GRAPH_MODE))

    MODEL_CONFIG = './config/model_config.json'
    with open(MODEL_CONFIG) as json_file:
        params = json.load(json_file)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(os.path.join(args.save_path, "checkpoints"))
    else:
        if not os.path.exists(os.path.join(args.save_path, "checkpoints")):
            os.makedirs(os.path.join(args.save_path, "checkpoints"))
    print("Your outputs will be save in", args.save_path)

    params = params['gcn']
    gcn = params['gcn']
    layer_name = params['layer_name']
    models = params['models']
    configs = params['configs']

    for ont in args.ontology:
        print(models[ont])
        predictor = Predictor(models[ont], configs[ont], gcn=gcn)

        if args.cmap is not None:
            predictor.predict(args.cmap)
        if args.npz_dir is not None:
            predictor.predict_from_npz_dir(args.npz_dir)
        if args.pdb_dir is not None:
            predictor.predict_from_pdb_dir(args.pdb_dir)
        if args.evaluation_path is not None:
            predictor.compute_precision(args.evaluation_path + '*', ont)

        # save predictions
        if args.evaluation_path is None:
            predictor.export_csv(args.save_path + "/DeepFRI_" + ont.upper() + "_predictions.csv",
                                 args.verbose)
            predictor.save_predictions(
                args.save_path + "/DeepFRI_" + ont.upper() + "_pred_scores.json")
