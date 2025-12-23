# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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
Inference script for Protenix.
"""
import logging
import os
import time
import traceback
from os.path import join as opjoin
from typing import Any, Mapping

import mindspore as ms

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from protenix.config import parse_configs, parse_sys_args
from protenix.data.infer_data_pipeline import get_inference_dataloader
from protenix.model.feat_batch import Batch
from protenix.model.modules import featurization
from protenix.model.protenix import Protenix
from protenix.download.colab_request_parser import RequestParser
from protenix.utils.seed import seed_everything
from runner.dumper import DataDumper

logger = logging.getLogger(__name__)


class InferenceRunner:
    """
    Runner for inference.
    """
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        """Initialize environment."""
        RequestParser.download_data_cache()

    def init_basics(self) -> None:
        """Initialize basic settings."""
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        """Initialize model."""
        config = Protenix.Config()
        deafult_channel = 256
        feature_input = 449
        pair_channel = 128
        single_channel = 384
        out_channel = 128
        feat_shape = (deafult_channel, feature_input)
        act_shape = (deafult_channel, deafult_channel, pair_channel)
        pair_shape = (deafult_channel, deafult_channel, pair_channel)
        single_shape = (deafult_channel, single_channel)
        num_templates = None
        self.model = Protenix(config, self.configs, feat_shape, act_shape, pair_shape, single_shape,
                              out_channel, num_templates, is_train=False, dtype=ms.float32)

    def load_checkpoint(self) -> None:
        """Load checkpoint."""
        ckpt_file_name = self.configs.load_checkpoint_path
        param_dict = ms.load_checkpoint(ckpt_file_name)
        ms.load_param_into_net(
            self.model, param_dict)
        self.model.set_train(False)
        self.print("Finish loading checkpoint.")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ):
        """Initialize dumper."""
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, ms.Tensor]:
        """Run prediction."""
        for i in data["input_feature_dict"]:
            print(i)
        batch = Batch()
        batch.load_from_dict(data)
        max_relative_idx = 32
        max_relative_chain = 2
        batch.rel_features = featurization.create_relative_encoding(
            batch.token_features, max_relative_idx=max_relative_idx, max_relative_chain=max_relative_chain
        )
        start = time.time()
        prediction = self.model(
            data=batch,
            key=42
        )
        print('model time:', time.time()-start)
        return prediction

    def print(self, msg: str):
        """Log message."""
        logger.info(msg)

def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    """Run inference prediction."""
    # Data
    logger.info("Loading data from\n%s", configs.input_json_path)
    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.info(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a", encoding="utf-8") as f:
            f.write(error_message)
        return

    num_data = len(dataloader)
    for seed in configs.seeds:
        seed_everything(seed=seed)
        for batch in dataloader:
            data, atom_array, data_error_message = batch
            sample_name = data["sample_name"]

            if len(data_error_message) > 0:
                logger.info(data_error_message)
                with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a", encoding="utf-8") as f:
                    f.write(data_error_message)
                continue

            logger.info(
                "[(%s/%s)] %s: N_asym %s, N_token %s, N_atom %s, N_msa %s",
                data['sample_index'] + 1, num_data, sample_name,
                data['N_asym'].item(), data['N_token'].item(),
                data['N_atom'].item(), data['N_msa'].item()
            )
            t0 = time.time()
            prediction = runner.predict(data)
            t1 = time.time()
            print(f'Inference time: {t1-t0}s')

            runner.dumper.dump(
                dataset_name="",
                pdb_id=sample_name,
                seed=seed,
                pred_dict=prediction[1]['pred_dict'],
                atom_array=atom_array,
                entity_poly_type=data["entity_poly_type"],
            )

            logger.info(
                "%s succeeded.\nResults saved to %s",
                data['sample_name'], configs.dump_dir
            )


def main(configs: Any) -> None:
    """Main entry point."""
    # Runner
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


def run() -> None:
    """Run the script."""
    log_format = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )
    main(configs)


if __name__ == "__main__":
    run()
