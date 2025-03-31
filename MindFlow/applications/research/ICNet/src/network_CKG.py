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
# pylint: disable=C0103
"""Network definitions"""
import numpy as np

import mindspore as ms
from mindspore import nn, Tensor, ops, Parameter
from mindspore.common.api import jit
from mindspore.amp import DynamicLossScaler
from mindspore.ops.functional import norm
from src.derivatives import SecondOrderGrad, Grad


# the deep neural network
class NeuralNet(nn.Cell):
    """NeuralNetwork"""
    def __init__(self, layers, msfloat_type):
        super(NeuralNet, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up activation
        self.activation = nn.Tanh()

        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(nn.Dense(layers[i], layers[i + 1]).to_float(msfloat_type))
            layer_list.append(self.activation.to_float(msfloat_type))

        layer_list.append(nn.Dense(layers[-2], layers[-1]).to_float(msfloat_type))

        # deploy layers
        self.layers = nn.SequentialCell(layer_list).to_float(msfloat_type)

    @jit
    def construct(self, x):
        out = self.layers(x)
        return out


class InvarianceConstrainedNN:
    """InvarianceConstrainedNN"""
    # Initialize the class
    def __init__(self, X, u, v, X_f, layers, batchno, use_npu, msfloat_type):

        # The Size of Network
        self.layers = layers

        self.use_npu = use_npu

        self.msfloat_type = msfloat_type

        # Initialize trainable parameters
        self.lambda_u = Tensor(np.zeros([10, 1]), self.msfloat_type)
        self.lambda_v = Tensor(np.zeros([10, 1]), self.msfloat_type)

        self.lambda_uxx = Tensor([0.0], self.msfloat_type)
        self.lambda_uyy = Tensor([0.0], self.msfloat_type)
        self.lambda_vxx = Tensor([0.0], self.msfloat_type)
        self.lambda_vyy = Tensor([0.0], self.msfloat_type)

        self.lambda_u = Parameter(self.lambda_u, name="name_u", requires_grad=True)
        self.lambda_v = Parameter(self.lambda_v, name="name_v", requires_grad=True)

        self.lambda_uxx = Parameter(self.lambda_uxx, name="name_uxx", requires_grad=True)
        self.lambda_uyy = Parameter(self.lambda_uyy, name="name_uyy", requires_grad=True)
        self.lambda_vxx = Parameter(self.lambda_vxx, name="name_vxx", requires_grad=True)
        self.lambda_vyy = Parameter(self.lambda_vyy, name="name_vyy", requires_grad=True)

        # Training data
        self.x = Tensor(np.array(X[:, 0:1], np.float32), self.msfloat_type)
        self.y = Tensor(np.array(X[:, 1:2], np.float32), self.msfloat_type)
        self.t = Tensor(np.array(X[:, 2:3], np.float32), self.msfloat_type)
        self.u = Tensor(np.array(u, np.float32), self.msfloat_type)
        self.v = Tensor(np.array(v, np.float32), self.msfloat_type)

        # Collection data for physics
        self.x_f = Tensor(np.array(X_f[:, 0:1], np.float32), self.msfloat_type)
        self.y_f = Tensor(np.array(X_f[:, 1:2], np.float32), self.msfloat_type)
        self.t_f = Tensor(np.array(X_f[:, 2:3], np.float32), self.msfloat_type)

        self.batchno = batchno
        self.batchsize_f = np.floor(self.x_f.shape[0] / self.batchno)

        # Initialize NNs---deep neural networks
        self.dnn = NeuralNet(layers, self.msfloat_type)

        params = self.dnn.trainable_params()
        params.append(self.lambda_u)
        params.append(self.lambda_v)

        params.append(self.lambda_uxx)
        params.append(self.lambda_uyy)
        params.append(self.lambda_vxx)
        params.append(self.lambda_vyy)

        # The function of Auto-differentiation
        self.grad = Grad(self.dnn)
        self.hessian_u_tt = SecondOrderGrad(self.dnn, 2, 2, output_idx=0)
        self.hessian_u_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=0)
        self.hessian_u_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=0)

        self.hessian_v_tt = SecondOrderGrad(self.dnn, 2, 2, output_idx=1)
        self.hessian_v_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=1)
        self.hessian_v_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=1)

        # The loss function
        self.l2_loss = nn.MSELoss()
        self.reduce_mean = ops.ReduceMean()

        # optimizers: using the same settings
        self.loss_scaler = DynamicLossScaler(1024, 2, 100)

    def net_u(self, x, y, t):
        H = ops.concat([x, y, t], 1)
        Y = self.dnn(H)
        return Y

    def net_f(self, x_f, y_f, t_f, N_f):
        """ The minspore autograd version of calculating residual """
        U = self.net_u(x_f, y_f, t_f)
        u = U[:, 0:1]
        v = U[:, 1:2]

        data = ops.concat([x_f, y_f, t_f], 1)

        u_tt = self.hessian_u_tt(data)
        u_xx = self.hessian_u_xx(data)
        u_yy = self.hessian_u_yy(data)

        v_tt = self.hessian_v_tt(data)
        v_xx = self.hessian_v_xx(data)
        v_yy = self.hessian_v_yy(data)

        lib_fun = [
            Tensor(
                np.ones(
                    (int(N_f),
                     1),
                    dtype=np.float32),
                self.msfloat_type),
            u,
            u * u,
            u * u * u,
            u * v,
            u * v * v,
            v,
            v * v,
            v * v * v,
            v * u * u]

        f_u = u_tt + self.lambda_uxx * u_xx + self.lambda_uyy * u_yy
        f_v = v_tt + self.lambda_vxx * v_xx + self.lambda_vyy * v_yy

        for i in range(len(lib_fun)):
            f_u = f_u - lib_fun[i] * self.lambda_u[i:i + 1, 0:1]
            f_v = f_v - lib_fun[i] * self.lambda_v[i:i + 1, 0:1]

        return f_u, f_v

    def loss_fn(self, x, y, t, x_f, y_f, t_f, u, v):
        """loss_fn"""
        U = self.net_u(x, y, t)
        u_pred = U[:, 0:1]
        v_pred = U[:, 1:2]

        f_u_pred, f_v_pred = self.net_f(x_f, y_f, t_f, self.batchsize_f)

        loss_u = self.l2_loss(u_pred, u)
        loss_v = self.l2_loss(v_pred, v)

        loss_f_u = self.l2_loss(f_u_pred, ops.zeros_like(f_u_pred))
        loss_f_v = self.l2_loss(f_v_pred, ops.zeros_like(f_v_pred))

        lambda_u = self.lambda_u
        lambda_v = self.lambda_v
        loss_lambda_u = 1e-7 * norm(lambda_u, ord=1)
        loss_lambda_v = 1e-7 * norm(lambda_v, ord=1)

        loss = loss_u + loss_v + loss_f_u + loss_f_v + loss_lambda_u + loss_lambda_v

        if self.use_npu:
            loss = self.loss_scaler.scale(loss)
        return loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_lambda_u, loss_lambda_v


class InvarianceConstrainedNN_STRdige:
    """InvarianceConstrainedNN_STRdige"""
    # Initialize the class
    def __init__(self, X, u, v, X_f, layers, batchno, lambda_u, lambda_v, lambda_uxx,
                 lambda_uyy, lambda_vxx, lambda_vyy, load_params, second_path, msfloat_type):

        # The Size of Network
        self.layers = layers

        self.msfloat_type = msfloat_type
        self.npfloat_type = npfloat_type

        # Initialize trainable parameters
        self.lambda_u = Tensor(lambda_u, self.msfloat_type)
        self.lambda_v = Tensor(lambda_v, self.msfloat_type)
        self.lambda_uxx = Tensor(lambda_uxx, self.msfloat_type)
        self.lambda_uyy = Tensor(lambda_uyy, self.msfloat_type)
        self.lambda_vxx = Tensor(lambda_vxx, self.msfloat_type)
        self.lambda_vyy = Tensor(lambda_vyy, self.msfloat_type)

        self.lambda_u = Parameter(self.lambda_u, name="name_u", requires_grad=True)
        self.lambda_v = Parameter(self.lambda_v, name="name_v", requires_grad=True)
        self.lambda_uxx = Parameter(self.lambda_uxx, name="name_uxx", requires_grad=True)
        self.lambda_uyy = Parameter(self.lambda_uyy, name="name_uyy", requires_grad=True)
        self.lambda_vxx = Parameter(self.lambda_vxx, name="name_vxx", requires_grad=True)
        self.lambda_vyy = Parameter(self.lambda_vyy, name="name_vyy", requires_grad=True)

        # Training data
        self.x = Tensor(np.array(X[:, 0:1], np.float32), self.msfloat_type)
        self.y = Tensor(np.array(X[:, 1:2], np.float32), self.msfloat_type)
        self.t = Tensor(np.array(X[:, 2:3], np.float32), self.msfloat_type)
        self.u = Tensor(np.array(u, np.float32), self.msfloat_type)
        self.v = Tensor(np.array(v, np.float32), self.msfloat_type)

        # Collection data for physics
        self.x_f = Tensor(np.array(X_f[:, 0:1], np.float32), self.msfloat_type)
        self.y_f = Tensor(np.array(X_f[:, 1:2], np.float32), self.msfloat_type)
        self.t_f = Tensor(np.array(X_f[:, 2:3], np.float32), self.msfloat_type)

        self.batchno = batchno
        self.batchsize_f = np.floor(self.x_f.shape[0] / self.batchno)

        # Initialize NNs---deep neural networks
        self.dnn = NeuralNet(layers, self.msfloat_type)
        if load_params:
            params_dict = ms.load_checkpoint(f'model/{second_path}/model.ckpt')
            ms.load_param_into_net(self.dnn, params_dict)

        params = self.dnn.trainable_params()
        params.append(self.lambda_u)
        params.append(self.lambda_v)

        params.append(self.lambda_uxx)
        params.append(self.lambda_uyy)
        params.append(self.lambda_vxx)
        params.append(self.lambda_vyy)

        # The function of Auto-differentiation
        self.grad = Grad(self.dnn)
        self.hessian_u_tt = SecondOrderGrad(self.dnn, 2, 2, output_idx=0)
        self.hessian_u_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=0)
        self.hessian_u_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=0)

        self.hessian_v_tt = SecondOrderGrad(self.dnn, 2, 2, output_idx=1)
        self.hessian_v_xx = SecondOrderGrad(self.dnn, 0, 0, output_idx=1)
        self.hessian_v_yy = SecondOrderGrad(self.dnn, 1, 1, output_idx=1)

        # The loss function
        self.l2_loss = nn.MSELoss()
        self.reduce_mean = ops.ReduceMean()

        # optimizers: using the same settings
        self.optimizer_Adam = nn.Adam(params, learning_rate=1e-3)
        self.loss_scaler = DynamicLossScaler(1024, 2, 100)

    def net_u(self, x, y, t):
        H = ops.concat([x, y, t], 1)
        Y = self.dnn(H)
        return Y

    def net_f(self, x_f, y_f, t_f, N_f):
        """ The minspore autograd version of calculating residual """
        U = self.net_u(x_f, y_f, t_f)
        u = U[:, 0:1]
        v = U[:, 1:2]

        data = ops.concat([x_f, y_f, t_f], 1)

        u_tt = self.hessian_u_tt(data)
        u_xx = self.hessian_u_xx(data)
        u_yy = self.hessian_u_yy(data)

        v_tt = self.hessian_v_tt(data)
        v_xx = self.hessian_v_xx(data)
        v_yy = self.hessian_v_yy(data)

        f_u = u_tt + self.lambda_uxx * u_xx + self.lambda_uyy * u_yy
        f_v = v_tt + self.lambda_vxx * v_xx + self.lambda_vyy * v_yy

        Phi_u = ops.cat((Tensor(np.ones((int(N_f), 1), dtype=np.float32), self.msfloat_type),
                         u, u * u, u * u * u, u * v, u * v * v, v, v * v, v * v * v, v * u * u), 1)
        Phi_v = ops.cat((Tensor(np.ones((int(N_f), 1), dtype=np.float32), self.msfloat_type),
                         u, u * u, u * u * u, u * v, u * v * v, v, v * v, v * v * v, v * u * u), 1)

        return f_u, f_v, Phi_u, Phi_v

    def loss_fn(self, x, y, t, x_f, y_f, t_f, u, v):
        """loss_fn"""
        U = self.net_u(x, y, t)
        u_pred = U[:, 0:1]
        v_pred = U[:, 1:2]

        f_u_pred, f_v_pred, _, _ = self.net_f(x_f, y_f, t_f, self.batchsize_f)

        loss_u = self.l2_loss(u_pred, u)
        loss_v = self.l2_loss(v_pred, v)

        loss_f_u = self.l2_loss(f_u_pred, ops.zeros_like(f_u_pred))
        loss_f_v = self.l2_loss(f_v_pred, ops.zeros_like(f_v_pred))

        lambda_u = self.lambda_u
        lambda_v = self.lambda_v
        loss_lambda_u = 1e-7 * norm(lambda_u, ord=1)
        loss_lambda_v = 1e-7 * norm(lambda_v, ord=1)

        loss = loss_u + loss_v + loss_f_u + loss_f_v + loss_lambda_u + loss_lambda_v

        # loss = self.loss_scaler.scale(loss)
        return loss, loss_u, loss_v, loss_f_u, loss_f_v, loss_lambda_u, loss_lambda_v

    def call_trainstridge(self, lam, d_tol):
        """call_trainstridge"""
        U_tt_pred, V_tt_pred, Phi_u, Phi_v = self.net_f(
            self.x_f, self.y_f, self.t_f, self.batchsize_f)

        lambda_u2 = self.train_stridge(
            Phi_u.numpy(),
            U_tt_pred.numpy(),
            lam,
            d_tol,
            maxit=100,
            STR_iters=10,
            l0_penalty=None,
            normalize=2,
            split=0.8,
            print_best_tol=False,
            uv_flag=True)
        lambda_v2 = self.train_stridge(
            Phi_v.numpy(),
            V_tt_pred.numpy(),
            lam,
            d_tol,
            maxit=100,
            STR_iters=10,
            l0_penalty=None,
            normalize=2,
            split=0.8,
            print_best_tol=False,
            uv_flag=False)

        return lambda_u2, lambda_v2

    def train_stridge(self, R, Ut, lam, d_tol, maxit, STR_iters=10, l0_penalty=None, normalize=2,
                      split=0.8, print_best_tol=False, uv_flag=True):
        """
        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
        Science Advances 3.4 (2017): e1602614.
        """

        # Split data into 80% training and 20% test, then search for the best tolderance.
        np.random.seed(0)  # for consistency
        n, _ = R.shape
        train = np.random.choice(n, int(n * split), replace=False)
        test = [i for i in np.arange(n) if i not in train]
        TestR = R[test, :]
        TestY = Ut[test, :]

        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        tol = d_tol

        # Get the standard least squares estimator
        if uv_flag:
            w_best = self.lambda_u.numpy()
        else:
            w_best = self.lambda_v.numpy()

        err_f = np.mean((TestY - TestR.dot(w_best))**2)

        if uv_flag:
            self.l0_penalty_0_u = 10 * err_f
            l0_penalty = self.l0_penalty_0_u
        else:
            self.l0_penalty_0_v = 10 * err_f
            l0_penalty = self.l0_penalty_0_v

        err_lambda = l0_penalty * np.count_nonzero(w_best)
        err_best = err_f + err_lambda
        tol_best = 0

        # Now increase tolerance until test performance decreases
        for iteration in range(maxit):
            print(f"Iteration {iteration + 1}")

            # Get a set of coefficients and error
            w = self.stridge(R, Ut, lam, STR_iters, tol, normalize=normalize, uv_flag=uv_flag)
            err_f = np.mean((TestY - TestR.dot(w))**2)

            err_lambda = l0_penalty * np.count_nonzero(w)
            err = err_f + err_lambda

            # Has the accuracy improved?
            if err <= err_best:
                err_best = err
                w_best = w
                tol_best = tol
                tol = 1.2 * tol

            else:
                tol = 0.8 * tol

        if print_best_tol:
            print("Optimal tolerance:", tol_best)

        optimaltol_history = np.empty([0])
        optimaltol_history = np.append(optimaltol_history, tol_best)

        return np.real(w_best)

    def stridge(self, X0, y, lam, maxit, tol, normalize=2, uv_flag=True):
        """stridge"""
        n, d = X0.shape
        X = np.zeros((n, d), dtype=np.float32)
        # First normalize data
        if normalize != 0:
            Mreg = np.zeros((d, 1))
            for i in range(0, d):
                Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
                X[:, i] = Mreg[i] * X0[:, i]
        else:
            X = X0

        # Get the standard ridge estimate
        # Inherit w from previous training
        if uv_flag:
            w = self.lambda_u.numpy() / Mreg
        else:
            w = self.lambda_v.numpy() / Mreg

        num_relevant = d
        biginds = np.where(abs(w) > tol)[0]

        # Threshold and continue
        for j in range(maxit):

            # Figure out which items to cut out
            smallinds = np.where(abs(w) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds):
                break
            else:
                num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if not new_biginds:
                if j == 0:
                    if normalize != 0:
                        w = np.multiply(Mreg, w)
                        return w
                    return w
                break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0

            if lam != 0:
                w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(
                    X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]
            else:
                w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []:
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(
                X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]

        if normalize != 0:
            w = np.multiply(Mreg, w)
            return w
        return w
