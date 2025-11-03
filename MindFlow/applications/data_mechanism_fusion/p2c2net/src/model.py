'''
Neural network models.
'''

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn, ms, Tensor, value_and_grad, jit, ops
from mindspore.amp import DynamicLossScaler, all_finite, auto_mixed_precision
from mindscience.models import FNO2D

class RCNN(nn.Cell):
    '''
    Recurrent Neural Network
    
    This class implements a correction block, filter block and Runge-Kutta 4th order method
    for solving burgers equations.
    
    The network consists of:
    - Two FNO2D blocks for u and v component corrections
    - A filter block for computing spatial derivatives
    - RK4 time integration for temporal evolution
    
    Args:
        deno (float): Denoising parameter, defaults to 1/104.0*4
        time_steps (int): Number of time steps for simulation, defaults to 32
        viscosity (float): Viscosity coefficient, defaults to 1/200.0
        delta_t (float): Time step size, defaults to 0.001
        in_channels (int): Number of input channels, defaults to 1
        out_channels (int): Number of output channels, defaults to 1
        resolution (int): Spatial resolution of the grid, defaults to 26
        modes (int): Number of Fourier modes in FNO, defaults to 12
        hidden_channels (int): Number of hidden channels in FNO, defaults to 12
        projection_channels (int): Number of projection channels in FNO, defaults to 5
        depths (int): Number of layers in FNO, defaults to 4
        kernel_size (int): Kernel size for filter operations, defaults to 5
        use_ascend (bool): Whether to use Ascend, defaults to True
        compute_dtype (mindspore.dtype): Computation data type, defaults to mstype.float32
    '''
    def __init__(self,
                 deno=1/104.0*4,
                 time_steps=32,
                 viscosity=1/200.0,
                 delta_t=0.001,
                 in_channels=1,
                 out_channels=1,
                 resolution=26,
                 modes=12,
                 hidden_channels=12,
                 projection_channels=5,
                 depths=4,
                 kernel_size=5,
                 use_ascend=True,
                 compute_dtype=mstype.float32
        ):
        super().__init__()
        fno_dtype = compute_dtype
        if use_ascend:
            fno_dtype = mstype.float16

        self.u_cor = FNO2D(
            in_channels = in_channels,
            out_channels = out_channels,
            resolutions = resolution,
            n_modes = modes,
            hidden_channels = hidden_channels,
            n_layers = depths,
            projection_channels = projection_channels,
            data_format = "channels_first",
            positional_embedding=False,
            fno_compute_dtype = fno_dtype
        )
        self.v_cor = FNO2D(
            in_channels = in_channels,
            out_channels = out_channels,
            resolutions = resolution,
            n_modes = modes,
            hidden_channels = hidden_channels,
            n_layers = depths,
            projection_channels = projection_channels,
            data_format = "channels_first",
            positional_embedding=False,
            fno_compute_dtype = fno_dtype
        )

        self.filter = Filter(
            out_channels=out_channels,
            kernel_size=kernel_size,
            deno=deno,
            compute_dtype=compute_dtype
        )

        self.steps = time_steps
        self.vis = viscosity
        self.dt = delta_t

    def rk(self, h):
        '''
        Compute the right-hand side of the burgers equation using spatial derivatives.
        '''
        u_prev = h[:, 0:1, ...]
        v_prev = h[:, 1:2, ...]

        u_cor = self.u_cor(u_prev)
        v_cor = self.v_cor(v_prev)

        filter_u = self.filter(u_cor)  # (du/dx, du/dy, d2u/dx2, d2u/dy2)
        filter_v = self.filter(v_cor)  # (dv/dx, dv/dy, d2v/dx2, d2v/dy2)

        # vis * (d2u/dx2 + d2u/dy2) - (du/dx * u_prev + du/dy * v_prev)
        u_res = (self.vis * (filter_u[:, 2:3, :, :] + filter_u[:, 3:4, :, :]) - (
                u_prev * filter_u[:, 0:1, :, :] + v_prev * filter_u[:, 1:2, :, :]))

        v_res = (self.vis * (filter_v[:, 2:3, :, :] + filter_v[:, 3:4, :, :]) - (
                u_prev * filter_v[:, 0:1, :, :] + v_prev * filter_v[:, 1:2, :, :]))

        return u_res, v_res

    def call_cell(self, h):
        '''
        Perform one time step using 4th-order Runge-Kutta method.
        '''
        u_prev = h[:, 0:1]
        v_prev = h[:, 1:2]
        k1_u, k1_v = self.rk(h)
        u1 = u_prev + k1_u * self.dt / 2.0
        v1 = v_prev + k1_v * self.dt / 2.0

        k2_u, k2_v = self.rk(ops.concat((u1, v1), axis=1))
        u2 = u_prev + k2_u * self.dt / 2.0
        v2 = v_prev + k2_v * self.dt / 2.0

        k3_u, k3_v = self.rk(ops.concat((u2, v2), axis=1))
        u3 = u_prev + k3_u * self.dt
        v3 = v_prev + k3_v * self.dt

        k4_u, k4_v = self.rk(ops.concat((u3, v3), axis=1))
        u_next = u_prev + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6.0 * self.dt
        v_next = v_prev + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0 * self.dt
        ch = ops.concat((u_next, v_next), axis=1)
        return ch

    def construct(self, init_uv):
        '''
        forward pass through the model for multiple time steps.
        '''
        outputs_uv = []
        internal_uv = init_uv
        for _ in range(self.steps - 1):
            internal_uv = self.call_cell(internal_uv)
            if internal_uv.shape[0] > 1:
                outputs_uv.append(ops.expand_dims(internal_uv, 0))
            else:  # infer
                outputs_uv.append(internal_uv)

        outputs_uv = ops.concat(tuple(outputs_uv), axis=0)
        return outputs_uv


class P2N2Net(nn.Cell):
    """
    Training wrapper for the P2C2Net model.
    
    Args:
        model (nn.Cell): The underlying RCNN model to be trained
        learning_rate (float): Learning rate for the Adam optimizer
        weight_decay (float): Weight decay coefficient for L2 regularization
        use_ascend (bool): Whether to use Ascend
        compute_dtype (mindspore.dtype): Computation data type
    """
    def __init__(self, model, learning_rate, weight_decay, use_ascend):
        super().__init__()
        self.model = model

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = nn.Adam(
            self.model.trainable_params(),
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.grad_fn = value_and_grad(
            fn = self.forward,
            grad_position = None,
            weights = self.model.trainable_params(),
            has_aux = False
        )
        self.use_ascend = use_ascend

        if self.use_ascend:
            auto_mixed_precision(self.model, 'O1')
            self.loss_scaler = DynamicLossScaler(
                scale_value = 2**10,
                scale_factor = 2,
                scale_window = 1
            )

    def forward(self, inputs, targets):
        '''
        Forward pass with loss computation.
        '''
        output_uv = self.model(inputs)
        logits = ops.transpose(output_uv, (1, 0, 2, 3, 4))
        loss = self.mse_loss(logits, targets) * 1e5
        return loss

    @jit
    def construct(self, batch_data):
        '''
        Training step with gradient computation and optimization.
        '''
        if 'data' not in batch_data or 'labels' not in batch_data:
            raise ValueError("batch_data must contain 'data' and 'labels' keys")

        inputs = batch_data['data'].squeeze()
        targets = batch_data['labels']
        loss, grads = self.grad_fn(inputs, targets)
        if self.use_ascend:
            loss = self.loss_scaler.unscale(loss)
            is_finite = all_finite(grads)
            if is_finite:
                grads = self.loss_scaler.unscale(grads)
                loss = ops.depend(loss, self.optimizer(grads))
            self.loss_scaler.adjust(is_finite)
        else:
            loss = ops.depend(loss, self.optimizer(grads))
        return loss

class Filter(nn.Cell):
    '''
    Filter Block
    '''
    def __init__(self, out_channels, kernel_size, deno, compute_dtype):
        super().__init__()
        self.dx = Conv(
            out_channels=out_channels,
            kernel_size=kernel_size,
            order=1,
            deno=deno,
            compute_dtype=compute_dtype
        )
        self.dxx = Conv(
            out_channels=out_channels,
            kernel_size=kernel_size,
            order=2,
            deno=deno,
            compute_dtype=compute_dtype
        )

    def pad_method(self, x):
        x_pad = ops.concat((x[:, :, :, -2:], x, x[:, :, :, :2]), axis=3)
        x_pad = ops.concat((x_pad[:, :, -2:, :], x_pad, x_pad[:, :, :2, :]), axis=2)
        return x_pad

    def construct(self, x):
        x = self.padMethod(x)
        dx = self.dx(x)
        dy = ops.transpose(self.dx(ops.transpose(x, (0, 1, 3, 2))), (0, 1, 3, 2))
        dxx = self.dxx(x)
        dyy = ops.transpose(self.dxx(ops.transpose(x, (0, 1, 3, 2))), (0, 1, 3, 2))
        res = ops.concat((dx, dy, dxx, dyy), axis=1)
        return res


class Conv(nn.Cell):
    '''
    A convolution module for computing spatial derivatives.
    '''
    def __init__(self, out_channels, order, deno, compute_dtype, kernel_size=5):
        super().__init__()
        self.deno = deno
        self.order = order
        self.conv2d = ops.Conv2D(
            out_channel=out_channels,
            kernel_size=kernel_size,
            data_format="NCHW"
        )
        if self.order == 2:
            self.deno = self.deno ** 2

        self.matrix_3 = ms.Parameter(Tensor(np.random.randn(3, 3), dtype=compute_dtype),
                                            requires_grad=True)

    def get_kernel(self):
        '''
        Get the kernel matrix.
        '''
        matrix = ops.zeros((5, 5))
        matrix[0, 0] = self.matrix_3[0, 0]
        matrix[0, 1] = self.matrix_3[0, 1]
        matrix[1, 0] = self.matrix_3[1, 0]
        matrix[1, 1] = self.matrix_3[1, 1]
        matrix[0, 2] = self.matrix_3[0, 2]
        matrix[1, 2] = self.matrix_3[1, 2]
        matrix[2, 0] = self.matrix_3[2, 0]
        matrix[2, 1] = self.matrix_3[2, 1]

        if self.order == 1:
            # 1
            matrix[0, 3] = -matrix[0, 1]
            matrix[0, 4] = -matrix[0, 0]
            matrix[1, 3] = -matrix[1, 1]
            matrix[1, 4] = -matrix[1, 0]
            # 2
            matrix[3, 0] = -matrix[1, 0]
            matrix[4, 0] = -matrix[0, 0]
            matrix[3, 1] = -matrix[1, 1]
            matrix[4, 1] = -matrix[0, 1]
            # 3
            matrix[3, 3] = -matrix[1, 3]
            matrix[4, 3] = -matrix[0, 3]
            matrix[3, 4] = -matrix[1, 4]
            matrix[4, 4] = -matrix[0, 4]
            # middle
            matrix[3, 2], matrix[4, 2] = -matrix[1, 2], -matrix[0, 2]
            matrix[2, 3], matrix[2, 4] = -matrix[2, 1], -matrix[2, 0]
            # temp = matrix[2,2]
            # matrix[2,2] = -(sum(matrix) - temp)
            matrix[2, 2] = 0
        else:
            matrix[0, 3] = matrix[0, 1]
            matrix[0, 4] = matrix[0, 0]
            matrix[1, 3] = matrix[1, 1]
            matrix[1, 4] = matrix[1, 0]
            # 2
            matrix[3, 0] = matrix[1, 0]
            matrix[4, 0] = matrix[0, 0]
            matrix[3, 1] = matrix[1, 1]
            matrix[4, 1] = matrix[0, 1]
            # 3
            matrix[3, 3] = matrix[1, 3]
            matrix[4, 3] = matrix[0, 3]
            matrix[3, 4] = matrix[1, 4]
            matrix[4, 4] = matrix[0, 4]
            # middle
            matrix[3, 2], matrix[4, 2] = matrix[1, 2], matrix[0, 2]
            matrix[2, 3], matrix[2, 4] = matrix[2, 1], matrix[2, 0]
            matrix[2, 2] = -(
                    (matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1]) * 4 + (
                    matrix[0, 2] + matrix[1, 2] + matrix[2, 0] + matrix[2, 1]) * 2)

        return matrix

    def construct(self, x):
        # update matrix
        m = self.get_kernel()
        weight = ops.expand_dims(ops.expand_dims(m, 0), 0)
        central = self.conv2d(x, weight) / self.deno
        return central
