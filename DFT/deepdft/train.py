# Copyright 2021 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""
import os
import sys
import json
import argparse
import math
import logging
import itertools
import timeit

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from   mindspore import nn, ops, context
from   mindspore.experimental import optim
from   mindspore.train import Model

import src.densitymodel as densitymodel
import src.dataset as dataset
from   src.utils import load_cfg


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config file")
    return load_cfg(parser.parse_args().config)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def split_data(dataset, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * 0.05))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }

        # Save split file
        with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
            json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = dataset.take(indices)
    return datasplits


def eval_model(model, dataloader):
    model.set_train(False)
    running_ae    = 0.
    running_se    = 0.
    running_count = 0.

    for batch_idx, batch in enumerate(dataloader.create_dict_iterator()):
        batch   = batch['validation']
        outputs = model(batch)
        targets = batch["probe_target"]

        running_ae    += ops.sum(ops.abs(targets - outputs)).asnumpy().item()
        running_se    += ops.sum(ops.square(targets - outputs)).asnumpy().item()
        running_count += ops.sum(batch["num_probes"]).asnumpy().item()

    mae  = (running_ae / running_count)
    rmse = (np.sqrt(running_se / running_count))
    return mae, rmse


def get_normalization(dataset, per_atom=True):
    try:
        num_targets = len(dataset.transformer.targets)
    except AttributeError:
        num_targets = 1
    x_sum = ops.zeros(num_targets)
    x_2 = ops.zeros(num_targets)
    num_objects = 0
    for sample in dataset:
        x = sample["targets"]
        if per_atom:
            x = x / sample["num_nodes"]
        x_sum += x
        x_2 += x ** 2.0
        num_objects += 1
    # Var(X) = E[X^2] - E[X]^2
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean ** 2.0

    return x_mean, ops.sqrt(x_var)


def count_parameters(model):
    return sum(p.size for p in model.trainable_params() if p.requires_grad)

def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    logging.debug("pid: %d", os.getpid())
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, 
                        device_id=args.device_id)
    logging.debug("Running in %s mode, using device %s", context.get_context("mode"), context.get_context("device_target"))

    if args.ignore_pbc and args.force_pbc:
        raise ValueError("ignore_pbc and force_pbc are mutually exclusive and can't both be set at the same time")
    elif args.ignore_pbc:
        set_pbc = False
    elif args.force_pbc:
        set_pbc = True
    else:
        set_pbc = None

    # Setup dataset and loader
    if args.dataset.endswith(".txt"):
    # Text file contains list of datafiles
        with open(args.dataset, "r") as datasetfiles:
            filelist = [os.path.join(os.path.dirname(args.dataset), line.strip('\n')) for line in datasetfiles]
    else:
        filelist = [args.dataset]

    logging.info("loading data %s", args.dataset)
    densitydata = None
    for path in filelist:
        if not densitydata:
            densitydata = dataset.DensityData(path)
        else:
            densitydata.concat(dataset.DensityData(path))

    # Split data into train and validation sets
    datasplits = split_data(densitydata, args)
    # datasplits["train"] = dataset.RotatingPoolData(datasplits["train"], 20)
    logging.info("train size: %d, val size: %d", len(datasplits["train"]), len(datasplits["validation"]))

    # Setup loaders
    batch_size = args.batch
    train_loader = ms.dataset.GeneratorDataset(
        source=dataset.BufferData(datasplits["train"], args, set_pbc, 1000),
        column_names=["train"],
        shuffle=True,
    )
    logging.info("Preloading training batch")
    train_loader = train_loader.batch(
        batch_size=batch_size, 
        per_batch_map=dataset.collate_list_of_dicts,
        num_parallel_workers=8,
    )

    val_loader = ms.dataset.GeneratorDataset(
        source=dataset.BufferData(datasplits["validation"], args, set_pbc, 5000),
        column_names=["validation"],
        shuffle=True,
    )
    logging.info("Preloading validation batch")
    val_loader = val_loader.batch(
        batch_size=batch_size,
        per_batch_map=dataset.collate_list_of_dicts,
        num_parallel_workers=8,
    )

    # Initialise model
    net = densitymodel.PainnDensityModel(args.num_interactions, args.node_size, args.cutoff,)
    logging.debug("model has %d parameters", count_parameters(net))

    # Setup optimizer
    optimizer = optim.Adam(net.trainable_params(), lr=0.0001)
    criterion = nn.MSELoss()
    scheduler_fn = lambda step: 0.96 ** (step / 100000)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, [scheduler_fn])

    def net_forward(input, output):
        logit = net(input)
        loss = criterion(logit, output)
        return loss

    net_backward = ms.value_and_grad(net_forward, None, optimizer.parameters)

    def train_step(input, output):
        loss, grads = net_backward(input, output)
        optimizer(grads)
        return loss

    log_interval = 5000
    running_loss = ms.Tensor(0.0, dtype=ms.float32)
    running_loss_count = ms.Tensor(0, dtype=ms.int32)
    best_val_mae = np.inf
    step = 0

    if args.load_model != "None":
        net_params = ms.load_checkpoint(os.path.join(args.load_model, "model.ckpt"))
        ms.load_param_into_net(net, net_params)
        optimizer_params = ms.load_checkpoint(os.path.join(args.load_model, "optimizer.ckpt"))
        ms.load_param_into_net(optimizer, optimizer_params)
        with open(os.path.join(args.load_model, "extra_info.json"), 'r') as f:
            runner_args = argparse.Namespace(**json.load(f))
        step = runner_args.step
        best_val_mae = runner_args.best_val_mae
        logging.info("start training")

    data_timer     = AverageMeter("data_timer")
    transfer_timer = AverageMeter("transfer_timer")
    train_timer    = AverageMeter("train_timer")
    eval_timer     = AverageMeter("eval_time")

    endtime = timeit.default_timer()
    for _ in itertools.count():
        for batch_idx, batch in enumerate(train_loader.create_dict_iterator()):
            net.set_train(True)
            data_timer.update(timeit.default_timer()-endtime)
            tstart = timeit.default_timer()

            batch = batch['train']
            transfer_timer.update(timeit.default_timer()-tstart)
            tstart = timeit.default_timer()

            loss = train_step(batch, batch["probe_target"])
            running_loss += loss * batch["probe_target"].shape[0] * batch["probe_target"].shape[1]
            running_loss_count += ops.sum(batch["num_probes"])
            train_timer.update(timeit.default_timer()-tstart)

            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                tstart = timeit.default_timer()
                train_loss = (running_loss / running_loss_count).item()
                running_loss = running_loss_count = 0
                val_mae, val_rmse = eval_model(net, val_loader)

                logging.info(
                    "step=%d, val_mae=%g, val_rmse=%g, sqrt(train_loss)=%g",
                    step,
                    val_mae,
                    val_rmse,
                    math.sqrt(train_loss),
                )

                # Save checkpoint
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    ms.save_checkpoint(net, os.path.join(args.output_dir, "model.ckpt"),)
                    ms.save_checkpoint(optimizer, os.path.join(args.output_dir, "optimizer.ckpt"),)
                    with open(os.path.join(args.output_dir, "extra_info.json"), 'w') as f:
                        json.dump({"step": step, "best_val_mae": best_val_mae}, f)

                eval_timer.update(timeit.default_timer()-tstart)
                logging.debug(
                    "%s %s %s %s" % (data_timer, transfer_timer, train_timer, eval_timer)
                )
            scheduler.step()

            step += 1
            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                sys.exit(0)

            endtime = timeit.default_timer()



if __name__ == "__main__":
    main()
