# Copyright 2021 Huawei Technologies Co., Ltd
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
"""inverse solver"""
from mindspore import nn, ops
from .metric import psnr, ssim


class TrainOneStepCell(nn.Cell):
    """
    One-step cell for problems using AD-FDTD network.

    Args:
        network (Cell): AD-FDTD network.
        optimizer (Cell): Gradient-based optimizer.
    """

    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        self.optimizer(grads)
        return loss


class EMInverseSolver:
    """
    AD-FDTD-Based Solver class for EM Inverse Problems.

    Args:
        network (Cell): AD-FDTD network.
        lossfn (Cell): Loss function.
        optimizer (Cell): Gradient-based optimizer.
    """

    def __init__(self, network, lossfn, optimizer):
        self.network = network
        self.lossfn = lossfn
        self.optimizer = optimizer

        self.net_with_loss = nn.WithLossCell(self.network, self.lossfn)
        self.one_train_step = TrainOneStepCell(
            self.net_with_loss, self.optimizer)

    def solve(self, epochs, waveform_t, field_labels):
        """solve process"""
        self.one_train_step.set_train()
        nbits = len(str(epochs))
        for epoch in range(epochs):
            loss = self.one_train_step(waveform_t, field_labels)
            print(
                f"Epoch: [{epoch:{nbits}d} / {epochs:{nbits}d}], loss: {loss.asnumpy():.6e}")

    def eval(self, epsr_labels=None, sigma_labels=None):
        """evaluation and measurement"""
        self.one_train_step.set_train(False)

        epsr, sigma = self.network.designer(self.network.rho)

        eval_result_str = '\n'

        if epsr_labels is not None:
            psnr_score = psnr(epsr.asnumpy(), epsr_labels.asnumpy())
            ssim_score = ssim(epsr.asnumpy(), epsr_labels.asnumpy())
            eval_result_str += f'[epsr] PSNR: {psnr_score:.6f} dB, SSIM: {ssim_score:.6f}\n'

        if sigma_labels is not None:
            psnr_score = psnr(sigma.asnumpy(), sigma_labels.asnumpy())
            ssim_score = ssim(sigma.asnumpy(), sigma_labels.asnumpy())
            eval_result_str += f'[sigma] PSNR: {psnr_score:.6f} dB, SSIM: {ssim_score:.6f}\n'

        print(eval_result_str)

        return epsr, sigma
