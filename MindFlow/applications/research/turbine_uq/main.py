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
"""main"""
import os
import time
import argparse

from omegaconf import OmegaConf
import numpy as np
from mindspore import dtype as mstype
from mindspore import jit, ops, context, set_seed, save_checkpoint
from mindspore.amp import DynamicLossScaler, auto_mixed_precision
from mindflow.loss import RelativeRMSELoss
from mindflow.utils import load_yaml_config, print_log

from src import load_dataset, get_model, get_optimizer, repeat_tensor,\
init_record, inference_loop, load_record,\
run_inference, run_visualization, run_optimization

set_seed(0)
np.random.seed(0)


def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(description="gvrb predict")
    parser.add_argument("--mode", type=str, default="PYNATIVE",
                        choices=["GRAPH", "PYNATIVE"], help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend", "CPU"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="configs/UNet_GVRB.yaml")
    input_args = parser.parse_args()
    return input_args


def train(record):
    '''train and evaluate the network'''
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    compute_type = mstype.float16 if use_ascend else mstype.float32
    print_log(f'Use Ascend: {use_ascend}')
    config = OmegaConf.create(load_yaml_config(record.config))
    model = get_model(config, compute_type)
    print_log("Loading data...")
    data_loader_train, data_loader_test = load_dataset(config)
    loss_fn = RelativeRMSELoss(reduction='mean')
    steps_per_epoch = data_loader_train.get_dataset_size()
    optimizer = get_optimizer(model, config, steps_per_epoch)
    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

    def forward_fn(inputs, outputs):
        inputs = inputs.astype(dtype=compute_type)
        outputs = outputs.astype(dtype=compute_type)
        pred = model(inputs)
        loss = loss_fn(pred, outputs)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss, pred

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    @jit
    def train_step(inputs, output):
        inputs = repeat_tensor(inputs)
        (loss, pred), grads = grad_fn(inputs, output)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            grads = loss_scaler.unscale(grads)
        loss = ops.depend(loss, optimizer(grads))
        return loss, pred

    loss_step = {'train': [], 'test': []}
    loss_step['train'] = test(model, data_loader_train, prefix="train", epoch=0)
    loss_step['test'] = test(model, data_loader_test, prefix="test", epoch=0)

    # train
    print_log('training...')
    start_time = time.time()
    for epoch in range(1, 1 + config.train.epochs):
        local_time_beg = time.time()
        model.set_train()
        loss_all = []
        for data_tuple in data_loader_train:
            loss, _ = train_step(*data_tuple)
            loss_all.append(loss.asnumpy())
        loss = np.mean(loss_all)
        print_log(f"Epoch {epoch}: loss {loss:>5.3f}")
        epoch_seconds = time.time() - local_time_beg
        step_seconds = epoch_seconds / steps_per_epoch
        print_log(f"Train epoch time: {epoch_seconds:>5.3f} s, per step time: {step_seconds* 1000:>5.1f} ms")
        if epoch % config.test.interval == 0 or epoch == config.train.epochs:
            loss_step['train'].append(test(model, data_loader_train, prefix="train", epoch=epoch))
            loss_step['test'].append(test(model, data_loader_test, prefix="test", epoch=epoch))
            save_checkpoint(model, os.path.join(record.ckpt_model))
    print_log("training done!")
    print_log(f"Total train time: {time.time() - start_time}s")

    # inference
    print_log("inferencing...")
    run_inference(config, record)
    print_log("inferencing done!")

    # visualization
    print_log("visualizing...")
    run_visualization(record)
    print_log("visualizing done!")


def test(model, data_loader, prefix="train", epoch=0):
    '''test and evaluate the network'''
    l2_error = inference_loop(model, data_loader)
    print_log(f"Epoch {epoch}: {prefix}: {l2_error}")
    return l2_error


if __name__ == "__main__":
    args = parse_args()
    context.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
        device_target=args.device_target, device_id=args.device_id)
    train_record = init_record(args.config_file_path, record_name='gvrb_train')
    train(train_record)

    opt_record = load_record('./exp/gvrb_train')
    run_optimization(opt_record, optimize_type='dtm')
    run_optimization(opt_record, optimize_type='uq')
