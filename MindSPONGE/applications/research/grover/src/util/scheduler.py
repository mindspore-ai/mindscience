# Copyright 2022 Huawei Technologies Co., Ltd
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
The learning rate scheduler.
"""


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate."""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_step_lr(init_lr, max_lr, final_lr, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Warmup step learning rate.
    We use warmup step to optimize learning rate.
    """
    total_steps = int(total_epochs * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, max_lr, init_lr)
        else:
            exponential_gamma = float(final_lr / max_lr) ** float(1 / (total_steps - warmup_steps))
            lr = float(max_lr) * float(exponential_gamma ** (i + 1 - warmup_steps))
        lr_each_step.append(lr)

    return lr_each_step


def get_lr(args):
    """generate learning rate."""
    lr = warmup_step_lr(init_lr=args.init_lr,
                        max_lr=args.max_lr,
                        final_lr=args.final_lr,
                        warmup_epochs=args.warmup_epochs,
                        total_epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch)
    return lr
