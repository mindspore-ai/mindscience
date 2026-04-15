# ============================================================================
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
"""Evaluate."""

import argparse
import logging
import os
import pickle

import mindspore as ms
from mindspore import context

from finetune import build_loader
from src import pretrained, utils, trainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate(args):
    """Evaluate the model."""
    # set seed
    utils.seed_everything(args.random_seed)

    # load dataset
    val_loader = build_loader(
        dataset_path=args.val_data_path,
        num_workers=args.num_workers,
        batch_size=1000,
        target_config={"graph": ["energy", "stress"], "node": ["forces"]},
        augmentation=False, # do not apply random augment
        shuffle=False,
    )

    # load trained model
    if args.checkpoint_path is None:
        raise ValueError("Checkpoint path is not provided.")
    model = pretrained.orb_mptraj_only_v2(args.checkpoint_path)
    model_params = sum(p.size for p in model.trainable_params() if p.requires_grad)
    logging.info("Model has %d trainable parameters.", model_params)

    # begin evaluation
    model.set_train(False)
    val_iter = iter(val_loader)
    val_batch = next(val_iter)

    output = model(
        val_batch.edge_features,
        val_batch.node_features,
        val_batch.senders,
        val_batch.receivers,
        val_batch.n_node,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "predictions.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(output, f)

    loss_fn = trainer.OrbLoss(model)
    loss, logs = loss_fn.loss(val_batch)
    print(f"Validation loss: {loss}")
    for key, value in logs.items():
        print(f"    {key}: {value}")


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description="Evaluate orb model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/config_eval.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    args = utils.load_cfg(args.config)
    ms.set_context(
        mode=context.PYNATIVE_MODE,
        device_target=args.device_target,
        device_id=args.device_id,
        pynative_synchronize=True,
    )
    evaluate(args)


if __name__ == "__main__":
    main()
