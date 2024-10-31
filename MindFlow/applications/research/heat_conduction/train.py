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
"""
train
"""
import os
import time

import numpy as np
from mindspore import nn, Tensor, context, ops, jit, set_seed, data_sink, save_checkpoint
from mindspore import dtype as mstype
from mindspore.nn import L1Loss
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.utils import load_yaml_config, print_log

from src.utils import Trainer, init_model, check_file_path, count_params, plot_image, plot_image_first
from src.dataset import init_dataset


def train():
    """train"""
    set_seed(0)
    np.random.seed(0)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target="Ascend",
                        device_id=0)
    use_ascend = context.get_context("device_target") == "Ascend"
    print(use_ascend)

    config = load_yaml_config("./configs/combined_methods.yaml")
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]

    train_dataset, test_dataset, means, stds = init_dataset(data_params)
    print('train_dataset', train_dataset)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, all_finite, auto_mixed_precision
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        compute_dtype = mstype.float16
        model = init_model("unet2d", data_params, model_params, compute_dtype=compute_dtype)
        auto_mixed_precision(model, optimizer_params["amp_level"]["unet2d"])
    else:
        context.set_context(enable_graph_kernel=False)
        loss_scaler = None
        compute_dtype = mstype.float32
        model = init_model("unet2d", data_params, model_params, compute_dtype=compute_dtype)

    loss_fn = L1Loss()
    summary_dir = os.path.join(summary_params["summary_dir"], "Exp_datadriven", "unet2d")
    ckpt_dir = os.path.join(summary_dir, "ckpt_dir")
    check_file_path(ckpt_dir)
    check_file_path(os.path.join(ckpt_dir, 'img'))
    print_log('model parameter count:', count_params(model.trainable_params()))
    print_log(
        f'learning rate: {optimizer_params["lr"]["unet2d"]}, '
        f'T_in: {data_params["T_in"]}, T_out: {data_params["T_out"]}')
    steps_per_epoch = train_dataset.get_dataset_size()

    lr = get_warmup_cosine_annealing_lr(optimizer_params["lr"]["unet2d"], steps_per_epoch,
                                        optimizer_params["epochs"], optimizer_params["warm_up_epochs"])
    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=Tensor(lr),
                                   weight_decay=optimizer_params["weight_decay"])

    trainer = Trainer(model, data_params, loss_fn, means, stds)

    def forward_fn(inputs, labels):
        loss, _, _, _, _, _, _ = trainer.get_loss(inputs, labels)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(inputs, labels):
        loss, grads = grad_fn(inputs, labels)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
        loss_new = ops.depend(loss, optimizer(grads))
        return loss_new, inputs, labels

    def test_step(inputs, labels):
        return trainer.get_loss(inputs, labels)

    train_size = train_dataset.get_dataset_size()
    test_size = test_dataset.get_dataset_size()
    train_sink = data_sink(train_step, train_dataset, sink_size=1)
    test_sink = data_sink(test_step, test_dataset, sink_size=1)
    test_interval = summary_params["test_interval"]
    save_ckpt_interval = summary_params["save_ckpt_interval"]

    for epoch in range(1, optimizer_params["epochs"] + 1):
        time_beg = time.time()
        train_l1 = 0.0
        model.set_train()
        for _ in range(1, train_size + 1):
            loss_train, inputs, labels = train_sink()
            train_l1 += loss_train.asnumpy()
        train_loss = train_l1 / train_size
        if epoch >= trainer.hatch_extent:
            _, loss1, loss2, _, _, _, _ = trainer.get_loss(inputs, labels)
            trainer.renew_loss_lists(loss1, loss2)
            trainer.adjust_hatchs()
        print_log(
            f"epoch: {epoch}, "
            f"step time: {(time.time() - time_beg) / steps_per_epoch:>7f}, "
            f"loss: {train_loss:>7f}")

        if epoch % test_interval == 0:
            model.set_train(False)
            test_l1 = 0.0
            for _ in range(test_size):
                loss_test, loss1, loss2, inputs, pred, labels, _ = test_sink()
                test_l1 += loss_test.asnumpy()
            test_loss = test_l1 / test_size
            print_log(
                f"epoch: {epoch}, "
                f"step time: {(time.time() - time_beg) / steps_per_epoch:>7f}, "
                f"loss: {test_loss:>7f}")

            plot_image(inputs, 0)
            plot_image_first(inputs, 0)
            plot_image(pred, 0)
            plot_image(labels, 0)

        if epoch % save_ckpt_interval == 0:
            save_checkpoint(model, ckpt_file_name=os.path.join(ckpt_dir, 'model_data.ckpt'))

    print("Training Finished!!")


if __name__ == "__main__":
    train()
