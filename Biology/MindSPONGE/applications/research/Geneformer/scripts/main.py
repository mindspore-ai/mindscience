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
# pylint: disable=C0411
# pylint: disable=C0413

"""main script"""
import os
import pickle
import yaml
import argparse
import sys
import mindspore as ms
from mindspore.communication import init
sys.path.append("..")
from src.classifier import GeneClassifier


def parse_args():
    """args function"""
    parser = argparse.ArgumentParser(description="run geneformer_classification")
    parser.add_argument("--bert_config_path", type=str, required=True, help="bert config path")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--do_train", type=bool, required=False, default=True, help="do train mode")
    parser.add_argument("--data_parallel", type=bool, required=False, default=False, help="do data parallel")
    parser.add_argument("--max_ncells", type=int, required=False, default=10_000, help="max ncells")
    parser.add_argument("--freeze_layers", type=int, required=False, default=4, help="freeze_layers")
    parser.add_argument("--num_crossval_splits", type=int, required=False, default=5, help="num_crossval_splits")
    parser.add_argument("--forward_batch_size", type=int, required=False, default=200, help="forward_batch_size")
    parser.add_argument("--nproc", type=int, required=False, default=16, help="nproc")
    args = parser.parse_args()
    return args

def main(main_args):
    """main"""
    with open(main_args.dataset_path, 'r') as file:
        data_config = yaml.safe_load(file)
    gene_class_dict_path = data_config.get("gene_class_dict_path")
    dataset_path = data_config.get("dataset_path")
    output_prefix = data_config.get("output_prefix")
    output_dir = data_config.get("output_dir")
    data_output = data_config.get("data_output")
    model_output = data_config.get("model_output")

    # ensure not overwriting previously saved model
    ms_model = os.path.join(model_output, "geneformer_mindspore.ckpt")
    if os.path.isfile(ms_model) is False:
        raise FileNotFoundError(f"geneformer_mindspore.ckpt not found in {model_output}.")
    if not os.path.exists(data_output):
        os.makedirs(data_output)
    with open(gene_class_dict_path, "rb") as fp:
        gene_class_dict = pickle.load(fp)

    gc = GeneClassifier(gene_class_dict=gene_class_dict,
                        max_ncells=main_args.max_ncells,
                        freeze_layers=main_args.freeze_layers,
                        num_crossval_splits=main_args.num_crossval_splits,
                        forward_batch_size=main_args.forward_batch_size,
                        nproc=main_args.nproc,
                        config_path=main_args.bert_config_path,
                        do_train=main_args.do_train)

    gc.prepare_data(input_data_file=dataset_path,
                    output_directory=data_output,
                    output_prefix=output_prefix)

    all_metrics = gc.validate(model_directory=model_output,
                              prepared_input_data=f"{data_output}/{output_prefix}_labeled.dataset",
                              id_class_dict_file=f"{data_output}/{output_prefix}_id_class_dict.pkl",
                              output_directory=output_dir,
                              output_prefix=output_prefix)

    print(all_metrics)

if __name__ == '__main__':
    actual_args = parse_args()
    if actual_args.data_parallel:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=int(os.getenv('DEVICE_ID')))
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        init()
        ms.set_seed(1)
        main(actual_args)
    else:
        main(actual_args)
