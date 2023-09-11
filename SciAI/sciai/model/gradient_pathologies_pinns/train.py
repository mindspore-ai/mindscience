
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

"""gppinns train"""
import timeit

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

from sciai.common import Sampler
from sciai.context import init_project
from sciai.utils import print_log, amp2datatype
from sciai.utils.python_utils import print_time
from src.network import Helmholtz2D, HelmholtzEqn, TrainOneStep
from src.plot import plot, plot_elements_namedtuple
from src.process import get_model_inputs, generate_test_data, prepare


def train(model, bcs_samplers, res_sampler, args, dtype):
    """Trains the model by minimizing the MSE loss"""

    start_time = timeit.default_timer()
    train_start_time = timeit.default_timer()

    loss_bcs_log = []
    loss_res_log = []

    # Adaptive constant
    beta = 0.9
    adaptive_constant_val = Tensor(1, dtype=dtype)
    adaptive_constant_log = []

    # learning rate & Adam Optimize
    exponential_decay_lr = nn.ExponentialDecayLR(args.lr, decay_rate=0.9, decay_steps=1000, is_stair=False)
    optimizer = nn.Adam(model.trainable_params(), exponential_decay_lr)
    train_net = TrainOneStep(model, optimizer)

    for it in range(args.epochs):

        inputs = get_model_inputs(bcs_samplers, res_sampler, model, args.batch_size, adaptive_constant_val)

        loss_res, loss_bcs, grad_res, grad_bcs = train_net(*inputs)
        loss = loss_res + loss_bcs

        model.update_grad_list(grad_res, grad_bcs)

        # Print
        if it % 10 == 0:
            elapsed = timeit.default_timer() - start_time
            total_time = timeit.default_timer() - train_start_time

            loss_bcs_log.append(loss_bcs / adaptive_constant_val)
            loss_res_log.append(loss_res)

            # Compute and Print adaptive weights during training
            if model.model in ['M2', 'M4']:
                adaptive_constant_new = model.max_grad_res / model.mean_grad_bcs
                adaptive_constant_val = adaptive_constant_new * (1.0 - beta) + beta * adaptive_constant_val

            adaptive_constant_log.append(adaptive_constant_val)

            print_log('step: %d, loss: %.3e, loss_bcs: %.3e, loss_res: %.3e, adaptive_constant: %.2f, interval: %.2f, '
                      'total: %.2f' % (it, loss, loss_bcs, loss_res, adaptive_constant_val, elapsed, total_time))

            start_time = timeit.default_timer()

        # Store gradients
        if it % 10000 == 0:
            model.save_gradients()
            print_log("Gradients information and ckpt stored ...")
            ms.save_checkpoint(model,
                               f'{args.save_ckpt_path}/Optim_{args.model_name}_{args.method}_{args.amp_level}.ckpt')

    return adaptive_constant_log, loss_res_log, loss_bcs_log


@print_time("train")
def main(args):
    # Define Helmholtz Equation
    helm = HelmholtzEqn(a1=1, a2=4, lam=1.0)

    # Domain boundaries
    bc1_coords = np.array([[-1.0, -1.0], [1.0, -1.0]])
    bc2_coords = np.array([[1.0, -1.0], [1.0, 1.0]])
    bc3_coords = np.array([[1.0, 1.0], [-1.0, 1.0]])
    bc4_coords = np.array([[-1.0, 1.0], [-1.0, -1.0]])
    dom_coords = np.array([[-1.0, -1.0], [1.0, 1.0]])

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, helm.u, name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, helm.u, name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, helm.u, name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, helm.u, name='Dirichlet BC4')
    bcs_sampler = [bc1, bc2, bc3, bc4]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, helm.f, name='Forcing')

    # build Helmholtz model
    dtype = amp2datatype(args.amp_level)
    model = Helmholtz2D(args.layers, res_sampler, 1.0, args.method, dtype)
    print_log(f"model {model.model} built successfully")

    if dtype == ms.float16:
        model.to_float(ms.float16)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, model)

    # Train model
    logs = train(model, bcs_sampler, res_sampler, args, dtype)
    if args.save_ckpt:
        ms.save_checkpoint(model, f'{args.save_ckpt_path}/{args.model_name}_{args.method}_amp_{args.amp_level}.ckpt')

    # Test data
    x1, x2, x_star = generate_test_data(dom_coords)

    # Evaluate
    u_pred, u_star = model.evaluate(helm, x_star)

    if args.save_fig:
        elements = plot_elements_namedtuple(args, model, data=(x_star, u_star, u_pred), test_data=(x1, x2), logs=logs)
        plot(elements)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
