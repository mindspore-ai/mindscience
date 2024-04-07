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
r"""Train the model."""
import time
import argparse
import math
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import ops, nn, Tensor, context
from mindspore.communication import init, get_rank
from mindspore.amp import DynamicLossScaler, auto_mixed_precision

from src.data import load_dataset, split_data_tuple
from src.core.metric import calculate_l2_error, L2ErrorRecord
from src.core.losses import LossFunction
from src.core.lr_scheduler import get_lr
from src.core.optimizer import get_optimizer
from src.utils.load_yaml import load_config
from src.utils.record import init_record
from src.utils.tools import AllGather, postprocess_batch_data, postprocess_data, set_seed
from src.utils.visual import plot_2d
from src.cell import get_model


def parse_args():
    r"""Parse input args"""
    parser = argparse.ArgumentParser(description="pde foundation model")
    parser.add_argument("--mode", type=str, default="GRAPH",
                        choices=["GRAPH", "PYNATIVE"], help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False,
                        choices=[True, False], help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend", "CPU"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument('--no_distributed', action='store_true', help='unenable distributed training (data parallel)')
    parser.add_argument('--no_data_sink', action='store_true', help='unenable data sink during training')
    parser.add_argument("--config_file_path", type=str,
                        default="configs/config_base.yaml")

    input_args = parser.parse_args()
    input_args.distributed = not input_args.no_distributed
    input_args.data_sink = not input_args.no_data_sink

    return input_args


def test_loop(dataset_iter, dataset, img_name="result", plot_num=1):
    r"""Test loop for a single dataset"""
    l2_error = []
    worst_l2_error = 0.
    model.set_train(False)
    for batch_idx, data_tuple in enumerate(dataset_iter):
        input_tuple, label, data_idx = split_data_tuple(data_tuple)  # (tuple, tensor, tensor)

        pred = model(*input_tuple)  # [bsz, num_points, dim_out] or [bsz, dim_out, n_t_grid, n_x_grid]

        l2_error_tmp = calculate_l2_error(label, pred)  # [bsz]
        l2_error.extend(l2_error_tmp.tolist())

        # update the worst sample
        worst_idx = int(np.argmax(l2_error_tmp))
        if l2_error_tmp[worst_idx] > worst_l2_error:
            worst_l2_error = l2_error_tmp[worst_idx]
            worst_label = label[worst_idx]
            worst_pred = pred[worst_idx]
            worst_data_idx = data_idx[worst_idx]

        # plot label vs. pred, only plot the first component
        if batch_idx == 0:
            for plot_idx in range(plot_num):
                plot_data_idx = int(data_idx[plot_idx])
                pde_info = dataset.get_pde_info(plot_data_idx)

                label_plot, pred_plot, pde_latex = postprocess_batch_data(label, pred,
                                                                          pde_info, plot_idx, config.model_type)

                record.visual(plot_2d, label_plot, pred_plot, f"{img_name}_{plot_idx}.png",
                              title=pde_latex, save_dir=record.image2d_dir)

    l2_error = Tensor(l2_error).astype(mstype.float32)  # [datasize]

    # distributed training (data parallel)
    if use_ascend and args.distributed:
        # [num_devices, datasize] => [num_devices * datasize]
        all_l2_error = all_gather(l2_error)
        l2_error_np = all_l2_error.flatten().asnumpy()

        # select worst sample across devices
        worst_l2_errors = all_gather(Tensor([worst_l2_error]))  # [num_devices]
        worst_idx = int(np.argmax(worst_l2_errors.asnumpy()))
        worst_l2_error = worst_l2_errors[worst_idx]

        # [*] -> [1, *] -> [num_devices, *] -> [*]
        worst_label = all_gather(worst_label.expand_dims(0))[worst_idx]
        worst_pred = all_gather(worst_pred.expand_dims(0))[worst_idx]
        worst_data_idx = all_gather(worst_data_idx.expand_dims(0))[worst_idx]
    else:
        l2_error_np = l2_error.flatten().asnumpy()

    # plot the worst sample
    if plot_num > 0:
        worst_data_idx = int(worst_data_idx)
        pde_info = dataset.get_pde_info(worst_data_idx)

        label_plot, pred_plot, pde_latex = postprocess_data(worst_label, worst_pred, pde_info, config.model_type)

        record.visual(plot_2d, label_plot, pred_plot, f"{img_name}_worst-{worst_data_idx}.png",
                      title=pde_latex, save_dir=record.image2d_dir)

    return l2_error_np


def test_dataset_dict(epoch, dataset_dict, prefix="train"):
    r"""Test loop for multiple pde datasets"""
    l2_error_record = L2ErrorRecord()

    plot_num_per_cls = config.test.plot_num_per_cls
    for pde_type in dataset_dict:
        # make the plots distributed uniformly over all datasets
        num_datasets = len(dataset_dict[pde_type])
        if num_datasets == 0:
            continue
        plot_nums = [plot_num_per_cls // num_datasets] * num_datasets
        for i in range(plot_num_per_cls % num_datasets):
            plot_nums[i] += 1

        for datafile, (dataset_iter, dataset) in dataset_dict[pde_type].items():
            cur_plot_num = plot_nums.pop(0)
            img_name = f"{prefix}_epoch-{epoch}_{pde_type}_{datafile}"
            l2_error = test_loop(dataset_iter, dataset, img_name=img_name, plot_num=cur_plot_num)
            l2_error_dict = l2_error_record.append(pde_type, datafile, l2_error)
            record.print(f"Epoch {epoch}: {prefix} {pde_type} {datafile} " + l2_error_record.dict2str(l2_error_dict))
            record.add_dict(epoch, l2_error_dict, prefix=f"{prefix}_{pde_type}_{datafile}")

        if len(dataset_dict[pde_type]) != 0:  # pylint: disable=C1801
            l2_error_dict = l2_error_record.reduce(pde_type)
            record.print(f"Epoch {epoch}: {prefix} {pde_type} all " + l2_error_record.dict2str(l2_error_dict))
            record.add_dict(epoch, l2_error_dict, prefix=f"{prefix}_{pde_type}_all")

    l2_error_dict = l2_error_record.reduce("all")
    record.print(f"Epoch {epoch}: {prefix} all all " + l2_error_record.dict2str(l2_error_dict))
    record.add_dict(epoch, l2_error_dict, prefix=f"{prefix}_all_all")

    return l2_error_dict


def test(epoch):
    r"""Test the model with both train data and test data"""
    test_dataset_dict(epoch, train_iter_dict, prefix="train")
    l2_error_test = test_dataset_dict(epoch, test_iter_dict, prefix="test")

    return l2_error_test


def train():
    r"""Train the model."""
    # loss function
    loss_fn = LossFunction(config.train.loss.type.upper(),
                           normalize=config.train.loss.normalize,
                           reduce_mean=True,
                           normalize_eps=config.train.loss.normalize_eps)

    # optimizer
    steps_per_epoch = dataset_train.get_dataset_size()
    lr_var = get_lr(steps_per_epoch, config.train)
    optimizer = get_optimizer(lr_var, model, config)

    # gradient postprocess
    if use_ascend and args.distributed:
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters)
    else:
        def grad_reducer(x):
            return x
    grad_clip_value = config.train.get("grad_clip_value", -1)

    # auto mixed precision
    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

    # define forward function
    def forward_fn(input_tuple, label):
        pred = model(*input_tuple)
        loss = loss_fn(pred, label)

        # auto mixed precision
        if use_ascend:
            loss = loss_scaler.scale(loss)

        return loss, pred

    # define function of calculate gradient
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # train one step
    @ms.jit
    def train_step(*data_tuple):
        input_tuple, label, _ = split_data_tuple(data_tuple)  # (tuple, tensor, tensor)
        (loss, pred), grads = grad_fn(input_tuple, label)

        # auto mixed precision
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            grads = loss_scaler.unscale(grads)

        if grad_clip_value > 0:
            grads = ops.clip_by_global_norm(grads, clip_norm=grad_clip_value)
        grads = grad_reducer(grads)  # distributed training (data parallel)
        loss = ops.depend(loss, optimizer(grads))

        return loss, pred

    # data sink
    if args.data_sink:
        sink_process = ms.data_sink(train_step, dataset_train, 1)
    else:
        dataset_iter = dataset_train.create_tuple_iterator()
    dataset_train_size = dataset_train.get_dataset_size()

    # test before training
    l2_error_test = test(epoch=0)
    if 'l2_error_mean' not in l2_error_test:
        raise KeyError("'l2_error_mean' is not recorded in the test result. Please check the test function.")
    l2_error_best = l2_error_test['l2_error_mean']

    # training loop
    record.print('training...')
    print_interval = math.ceil(config.train.epochs / 2500)
    for epoch in range(1, 1 + config.train.epochs):
        model.set_train()
        loss_all = []
        if args.data_sink:
            # data sink
            for _ in range(dataset_train_size):
                loss, _ = sink_process()
                loss_all.append(loss.asnumpy())
        else:
            # not data sink
            for data_tuple in dataset_iter:
                loss, _ = train_step(*data_tuple)
                loss_all.append(loss.asnumpy())

        if (epoch - 1) % print_interval == 0:
            loss = np.mean(loss_all)  # [dataset_train_size] -> []
            record.print(f"Epoch {epoch}: loss {loss:>10f}")
            record.add_scalar("train/loss", loss, epoch)

        # test
        if epoch % config.test.interval == 0 or epoch == config.train.epochs:
            l2_error_test = test(epoch=epoch)
            if 'l2_error_mean' not in l2_error_test:
                raise KeyError("'l2_error_mean' is not recorded in the test result. Please check the test function.")

            # save last checkpoint
            record.save_ckpt(model, 'model_last.ckpt')

            # save best checkpoint
            if l2_error_best > l2_error_test['l2_error_mean']:
                l2_error_best = l2_error_test['l2_error_mean']
                record.save_ckpt(model, 'model_best.ckpt')

    record.print(f"best l2_error_mean: {l2_error_best:>7f}")
    record.print("training done!")


if __name__ == "__main__":
    # seed
    set_seed(123456)

    # args
    args = parse_args()

    # mindspore context
    context.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
        device_target=args.device_target)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    # compute_type
    compute_type = mstype.float16 if use_ascend else mstype.float32

    # distributed training (data parallel)
    rank_id = None
    if use_ascend:
        if args.distributed:
            init()  # enable HCCL
            context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
            rank_id = get_rank()
            all_gather = AllGather()  # nn.cell for ops.ALLGather()
        else:
            context.set_context(device_id=args.device_id)

    # load config file
    config, config_str = load_config(args.config_file_path)

    # init record
    record = init_record(use_ascend, rank_id, args, config, config_str)

    # dataset
    record.print(f"Loading {config.data.type} data...")
    (dataset_train, train_iter_dict, test_iter_dict, data_info) = load_dataset(config)

    # model
    model = get_model(config, record, compute_type)

    # train
    start_time = time.time()
    train()

    record.print(f"End-to-End total time: {time.time() - start_time} s")
    record.close()
