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
"""trainer for training"""
from mindspore import jit, ops


class TrainerInfo:
    """trainer info"""

    def __init__(self, case_name, model, optimizer, problem,
                 use_ascend, loss_scaler, config,
                 if_constr_sigmoid, if_weighted):
        self.case_name = case_name
        self.model = model
        self.optimizer = optimizer
        self.problem = problem
        self.use_ascend = use_ascend
        self.loss_scaler = loss_scaler
        self.config = config
        self.if_constr_sigmoid = if_constr_sigmoid
        self.if_weighted = if_weighted


class Trainer:
    """trainer"""

    def __init__(self, case_name, model, optimizer, problem):
        """init"""
        self.case_name = case_name
        self.model = model
        self.optimizer = optimizer
        self.problem = problem


    def set_params(self, use_ascend, loss_scaler, if_constr_sigmoid, if_weighted):
        self.use_ascend = use_ascend
        self.loss_scaler = loss_scaler
        self.if_constr_sigmoid = if_constr_sigmoid
        self.if_weighted = if_weighted
        self.grad_fn = ops.value_and_grad(
            self.forward_fn, None, self.optimizer.parameters, has_aux=False)

    def forward_fn(self, *data):
        """forward function for training"""
        loss = self.problem.get_loss(
            *data, if_constr_sigmoid=self.if_constr_sigmoid, if_weighted=self.if_weighted)
        if self.use_ascend:
            loss = self.loss_scaler.scale(loss)
        return loss

    @jit
    def train_step(self, *data):
        """train_step for training"""
        loss, grads = self.grad_fn(*data)
        if self.use_ascend:
            loss = self.loss_scaler.unscale(loss)
            grads = self.loss_scaler.unscale(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss
