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

from src import mgda_step

ALL_LOSS = 5
PDE_LOSS = 1
PDE_IC_LOSS = 1
BC_LOSS = 3


class Trainer:
    """trainer"""

    def __init__(self, case_name, model, optimizer, problem, use_ascend, loss_scaler, config):
        self.case_name = case_name
        self.model = model
        self.optimizer = optimizer
        self.problem = problem
        self.use_ascend = use_ascend
        self.loss_scaler = loss_scaler
        self.config = config


class BurgersTrainer(Trainer):
    """trainer for burgers"""

    def __init__(self, case_name, model, optimizer, problem, use_ascend, loss_scaler, config):
        super().__init__(case_name, model, optimizer,
                         problem, use_ascend, loss_scaler, config)
        self.grad_fn = ops.value_and_grad(
            self.forward_fn, None, self.optimizer.parameters, has_aux=False)

    def forward_fn(self, pde_data, ic_data, bc_data, loss_type):
        """forward function"""
        loss = self.problem.get_loss(
            pde_data, ic_data, bc_data, loss_type)
        if self.use_ascend:
            loss = self.loss_scaler.scale(loss)
        return loss

    @jit
    def get_grads(self, pde_data, ic_data, bc_data, loss_type):
        """get grads"""
        loss, grads = self.grad_fn(
            pde_data, ic_data, bc_data, loss_type)
        if self.use_ascend:
            loss = self.loss_scaler.unscale(loss)
            grads = self.loss_scaler.unscale(grads)
        return grads, loss

    def train_step(self, pde_data, ic_data, bc_data):
        """train step"""
        pde_ic_grads, pde_ic_loss = self.get_grads(
            pde_data, ic_data, bc_data, PDE_IC_LOSS)
        bc_grads, bc_loss = self.get_grads(
            pde_data, ic_data, bc_data, BC_LOSS)
        losses = [pde_ic_loss, bc_loss]
        origin_grads = [pde_ic_grads, bc_grads]
        loss = mgda_step(losses, origin_grads, self.config, self.optimizer)
        return loss

    @jit
    def loss_step(self, pde_data, ic_data, bc_data):
        """loss step"""
        loss = self.problem.get_loss(
            pde_data, ic_data, bc_data, ALL_LOSS)
        return loss


class CylinderflowTrainer(Trainer):
    """cylinder flow trainer"""

    def __init__(self, case_name, model, optimizer, problem, use_ascend, loss_scaler, config):
        super().__init__(case_name, model, optimizer,
                         problem, use_ascend, loss_scaler, config)
        self.grad_fn = ops.value_and_grad(
            self.forward_fn, None, self.optimizer.parameters, has_aux=False)

    def forward_fn(self, pde_data, bc_data, bc_label, ic_data, ic_label, loss_type):
        """forward function"""
        loss = self.problem.get_loss(
            pde_data, bc_data, bc_label, ic_data, ic_label, loss_type)
        if self.use_ascend:
            loss = self.loss_scaler.scale(loss)
        return loss

    @jit
    def get_grads(self, pde_data, bc_data, bc_label, ic_data, ic_label, loss_type):
        """get grads"""
        loss, grads = self.grad_fn(
            pde_data, bc_data, bc_label, ic_data, ic_label, loss_type)
        if self.use_ascend:
            loss = self.loss_scaler.unscale(loss)
            grads = self.loss_scaler.unscale(grads)
        return grads, loss

    def train_step(self, pde_data, bc_data, bc_label, ic_data, ic_label):
        """train step"""
        pde_ic_grads, pde_ic_loss = self.get_grads(
            pde_data, bc_data, bc_label, ic_data, ic_label, PDE_IC_LOSS)
        bc_grads, bc_loss = self.get_grads(
            pde_data, bc_data, bc_label, ic_data, ic_label, BC_LOSS)
        losses = [pde_ic_loss, bc_loss]
        origin_grads = [pde_ic_grads, bc_grads]
        loss = mgda_step(losses, origin_grads, self.config, self.optimizer)
        return loss

    @jit
    def loss_step(self, pde_data, bc_data, bc_label, ic_data, ic_label):
        """loss step"""
        loss = self.problem.get_loss(
            pde_data, bc_data, bc_label, ic_data, ic_label, ALL_LOSS)
        return loss


class PeriodichillTrainer(Trainer):
    """periodic hill trainer"""

    def __init__(self, case_name, model, optimizer, problem, use_ascend, loss_scaler, config):
        super().__init__(case_name, model, optimizer,
                         problem, use_ascend, loss_scaler, config)
        self.grad_fn = ops.value_and_grad(
            self.forward_fn, None, self.optimizer.parameters, has_aux=False)

    def forward_fn(self, pde_data, data, label, loss_type):
        """forward function"""
        loss = self.problem.get_loss(
            pde_data, data, label, loss_type)
        if self.use_ascend:
            loss = self.loss_scaler.scale(loss)
        return loss

    @jit
    def get_grads(self, pde_data, data, label, loss_type):
        """get grads"""
        loss, grads = self.grad_fn(
            pde_data, data, label, loss_type)
        if self.use_ascend:
            loss = self.loss_scaler.unscale(loss)
            grads = self.loss_scaler.unscale(grads)
        return grads, loss

    def train_step(self, pde_data, data, label):
        """train step"""
        pde_grads, pde_loss = self.get_grads(
            pde_data, data, label, PDE_LOSS)
        bc_grads, bc_loss = self.get_grads(
            pde_data, data, label, BC_LOSS)
        losses = [pde_loss, bc_loss]
        origin_grads = [pde_grads, bc_grads]
        loss = mgda_step(losses, origin_grads, self.config, self.optimizer)
        return loss

    @jit
    def loss_step(self, pde_data, data, label):
        """loss step"""
        loss = self.problem.get_loss(
            pde_data, data, label, ALL_LOSS)
        return loss
