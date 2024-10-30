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
# ===========================================================================
"""Train process of every epoch"""
import os
import numpy as np
import mindspore as ms
from tqdm import tqdm


def fit_one_epoch(model_train, model, grad_fn, yolo_loss, loss_history, eval_callback,
                  optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, epoch_num, save_period, save_dir):
    """fit one epoch"""

    loss = 0
    val_loss = 0

    print('Start Train')
    pbar = tqdm(
        total=epoch_step,
        desc=f'Epoch {epoch + 1}/{epoch_num}',
        postfix=dict,
        mininterval=0.3)
    model_train.set_train()

    for iteration, batch in enumerate(
            gen.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if iteration >= epoch_step:
            break

        y_trues = [[], [], []]
        images, targets = batch["image"]["images"], batch["box"]["bboxes"]
        y_trues[0] = batch["y_true0"]["y_trues0"]
        y_trues[1] = batch["y_true1"]["y_trues1"]
        y_trues[2] = batch["y_true2"]["y_trues2"]
        images = ms.Tensor.from_numpy(np.array(images)).type(ms.float32)
        targets = [
            ms.Tensor.from_numpy(ann).type(
                ms.float32) for ann in targets]
        y_trues = [
            ms.Tensor.from_numpy(
                np.array(
                    ann, np.float32)).type(
                        ms.float32) for ann in y_trues]

        (loss_value, outputs), grads = grad_fn(images, targets, y_trues)
        optimizer(grads)
        loss += loss_value.item()
        pbar.set_postfix(**{'loss': loss / (iteration + 1),})
        pbar.update(1)

    pbar.close()
    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(
        total=epoch_step_val,
        desc=f'Epoch {epoch + 1}/{epoch_num}',
        postfix=dict,
        mininterval=0.3)

    model_train_eval = model_train.set_train(False)

    for iteration, batch in enumerate(
            gen_val.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if iteration >= epoch_step_val:
            break
        y_trues = [[], [], []]
        images, targets = batch["image"]["images"], batch["box"]["bboxes"]
        y_trues[0] = batch["y_true0"]["y_trues0"]
        y_trues[1] = batch["y_true1"]["y_trues1"]
        y_trues[2] = batch["y_true2"]["y_trues2"]
        images = ms.Tensor.from_numpy(np.array(images)).type(ms.float32)
        targets = [
            ms.Tensor.from_numpy(ann).type(
                ms.float32) for ann in targets]
        y_trues = [
            ms.Tensor.from_numpy(
                np.array(
                    ann, np.float32)).type(
                        ms.float32) for ann in y_trues]
        outputs = model_train_eval(images)

        loss_value_all = 0
        for l, _ in enumerate(outputs):
            loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
            loss_value_all += loss_item
        loss_value = loss_value_all

        val_loss += loss_value.item()
        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
        pbar.update(1)

    pbar.close()
    print('Finish Validation')
    loss_history.append_loss(
        loss / epoch_step,
        val_loss / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1, model_train_eval)
    print('Epoch:' + str(epoch + 1) + '/' + str(epoch_num))
    print(f'Total Loss: {loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f} ')

    if (epoch + 1) % save_period == 0 or epoch + 1 == epoch_num:
        ms.save_checkpoint(
            model, os.path.join(
                save_dir, f"ep{epoch + 1:03d}-loss{loss / epoch_step:.3f}"
                          f"-val_loss{val_loss / epoch_step_val:.3f}"))

    if len(loss_history.val_loss) <= 1 or (
            val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.ckpt')
        ms.save_checkpoint(model, os.path.join(save_dir, "best_epoch_weights"))

    ms.save_checkpoint(model, os.path.join(save_dir, "last_epoch_weights"))
