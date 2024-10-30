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
"""train code based Yolov5"""
import datetime
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset

from nets.yolo_v5 import YoloBody
from nets.yolo_training import YOLOLoss
from utils.callbacks_v5 import EvalCallback, LossHistory
from utils.dataloader_v5 import YoloDataset, yolo_dataset_collate
from utils.utils import (get_anchors, get_classes,
                         seed_everything, show_config)
from utils.utils_fit_v5 import fit_one_epoch

if __name__ == "__main__":
    seed = 11
    classes_path = 'model_data/voc_classes_elc.txt'  # category file path
    anchors_path = 'model_data/yolo_anchors.txt'  # anchors path
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model_path = ''  # the path of the trained model
    input_shape = [512, 512]  # the size of the input image
    phi = 's'
    label_smoothing = 0  # label smoothing, generally set to below 0.01

    init_epoch = 0  # The current epoch of the model
    freeze_epoch = 50  # The epoch of model freeze training
    freeze_batch_size = 16  # The size of batch when freeze
    unfreeze_epoch = 300  # The total epoch of training
    unfreeze_batch_size = 16  # The size of batch when unfreeze
    freeze_train = False  # When set to True, freeze the training of the backbone network

    init_lr = 1e-2  # Maximum learning rate of the model
    min_lr = init_lr * 0.01  # The minimum learning rate of the model
    momentum = 0.937  # The momentum parameter used internally by the optimizer
    weight_decay = 5e-4  # Weight decay can prevent overfitting
    save_period = 10  # Save weights every save_period epochs
    save_dir = 'logs_v5'  # The folder where weights and log files are saved
    eval_flag = True  # Whether to conduct evaluation during training
    eval_period = 10  # Evaluation every eval_period epochs
    num_workers = 4 # Number of threads occupied by reading data

    train_annotation_path = '2007_train.txt' # Train image paths and labels
    val_annotation_path = '2007_val.txt' # Verify image path and labels

    seed_everything(seed)
    ms.set_context(device_target="Ascend")

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    model = YoloBody(anchors_mask, num_classes, phi)
    if model_path != '':
        print(f'Load weights {model_path}.')
        pretrained_dict = ms.load_checkpoint(model_path)
        param_not_load, _ = ms.load_param_into_net(model, pretrained_dict)
        print(param_not_load)

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, label_smoothing)
    time_str = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir)

    model_train = model.set_train()

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        classes_path=classes_path, anchors_path=anchors_path, anchors_mask=anchors_mask, model_path=model_path,
        input_shape=input_shape, Init_Epoch=init_epoch, Freeze_Epoch=freeze_epoch, UnFreeze_Epoch=unfreeze_epoch,
        Freeze_batch_size=freeze_batch_size, Unfreeze_batch_size=unfreeze_batch_size, Freeze_Train=freeze_train,
        Init_lr=init_lr, Min_lr=min_lr, momentum=momentum, save_period=save_period, save_dir=save_dir,
        num_workers=num_workers, num_train=num_train, num_val=num_val
    )
    wanted_step = 5e4
    total_step = num_train // unfreeze_batch_size * unfreeze_epoch
    if total_step <= wanted_step:
        if num_train // unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // unfreeze_batch_size) + 1
        print(
            f"\n\033[1;33;44m[Warning] 使用sgd优化器时，建议将训练总步长设置到{wanted_step}以上。\033[0m")
        print(
            f"\033[1;33;44m[Warning] 本次运行的总训练数据量为{num_train}，Unfreeze_batch_size为{unfreeze_batch_size}，"
            f"共训练{unfreeze_epoch}个Epoch，计算出总训练步长为{total_step}。\033[0m")
        print(
            f"\033[1;33;44m[Warning] 由于总训练步长为{total_step}，小于建议总步长{wanted_step}，建议设置总世代为{wanted_epoch}。\033[0m")

    unfreeze_flag = False
    if freeze_train:
        for param in model.backbone.get_parameters():
            param.requires_grad = False

    batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size

    nbs = 64
    lr_limit_max = 5e-2
    lr_limit_min = 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr,
                          lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr,
                         lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    lr_scheduler_func = nn.cosine_decay_lr(
        min_lr_fit,
        init_lr_fit,
        total_step,
        num_train //
        unfreeze_batch_size,
        unfreeze_epoch)

    pg1 = list(
        filter(
            lambda x: 'weight' in x.name or 'gamma' in x.name,
            model.trainable_params()))
    pg2 = list(
        filter(
            lambda x: 'beta' in x.name,
            model.trainable_params()))
    group_params = [{'params': pg1, "weight_decay": weight_decay},
                    {'params': pg2},
                    {'order_params': model.trainable_params()}]
    optimizer = nn.SGD(
        group_params,
        lr_scheduler_func,
        momentum=momentum,
        nesterov=True)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    train_dataset = YoloDataset(
        train_lines,
        input_shape,
        num_classes,
        anchors,
        anchors_mask,
        train=True)
    val_dataset = YoloDataset(
        val_lines,
        input_shape,
        num_classes,
        anchors,
        anchors_mask,
        train=False)

    train_sampler = None
    val_sampler = None
    shuffle = True

    gen = GeneratorDataset(
        train_dataset,
        shuffle=shuffle,
        num_parallel_workers=num_workers,
        sampler=train_sampler,
        column_names=[
            "image",
            "box",
            "y_true0",
            "y_true1",
            "y_true2"])
    gen_val = GeneratorDataset(
        val_dataset,
        shuffle=shuffle,
        num_parallel_workers=num_workers,
        sampler=val_sampler,
        column_names=[
            "image",
            "box",
            "y_true0",
            "y_true1",
            "y_true2"])
    gen = gen.batch(
        batch_size,
        drop_remainder=True,
        per_batch_map=yolo_dataset_collate)
    gen_val = gen_val.batch(
        batch_size,
        drop_remainder=True,
        per_batch_map=yolo_dataset_collate)

    eval_callback = EvalCallback(
        model,
        input_shape,
        anchors,
        anchors_mask,
        class_names,
        num_classes,
        val_lines,
        log_dir,
        eval_flag=eval_flag,
        period=eval_period)

    for epoch in range(init_epoch, unfreeze_epoch):
        if epoch >= freeze_epoch and not unfreeze_flag and freeze_train:
            batch_size = unfreeze_batch_size

            nbs = 64
            lr_limit_max = 5e-2
            lr_limit_min = 5e-4
            init_lr_fit = min(
                max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
            min_lr_fit = min(max(batch_size / nbs * min_lr,
                                 lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            gen = gen.batch(
                batch_size,
                drop_remainder=True,
                per_batch_map=yolo_dataset_collate)
            gen_val = gen_val.batch(
                batch_size,
                drop_remainder=True,
                per_batch_map=yolo_dataset_collate)

            unfreeze_flag = True

        def forward_fn(images, targets, y_trues):
            """forward function"""
            outputs = model_train(images)
            loss_value_all = 0
            for l, _ in enumerate(outputs):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all += loss_item
            loss_value = loss_value_all

            return loss_value, outputs

        grad_fn = ms.value_and_grad(
            forward_fn, None, optimizer.parameters, has_aux=True)

        fit_one_epoch(model_train, model, grad_fn, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                      epoch_step_val, gen, gen_val, unfreeze_epoch, save_period, save_dir)

    loss_history.writer.close()
