
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

"""maxwell net network"""
import math
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, ops, nn

from sciai.architecture import MSE


class LossNet(nn.Cell):
    """Loss Net"""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.mse = MSE()

    def construct(self, scat_pot_ms, ri_value_ms):
        """construct"""
        diff, _ = self.net(scat_pot_ms, ri_value_ms)
        loss = self.mse(diff)
        return loss


class MaxwellNet(nn.Cell):
    """Maxwell Net"""
    def __init__(self, args):
        super(MaxwellNet, self).__init__()
        self.mode = args.problem
        self.high_order = args.high_order
        self.model = UNet(args.in_channels, args.out_channels, args.depth, args.filter, args.norm, args.up_mode)

        # pixel size [um / pixel]
        delta = args.wavelength / args.dpl
        # wave-number [1 / um]
        k = 2 * math.pi / args.wavelength
        self.delta = ms.Tensor(delta, dtype=ms.float32, const_arg=True)
        self.k = ms.Tensor(k, dtype=ms.float32, const_arg=True)

        self.symmetry_x = args.symmetry_x
        self.pad = self.high_order

        self.padding_ref = nn.SequentialCell(
            [nn.ReflectionPad2d((0, 0, self.pad, 0)), nn.ZeroPad2d((self.pad, self.pad, 0, self.pad))])
        self.padding_zero = nn.SequentialCell([nn.ZeroPad2d((self.pad, self.pad, self.pad, self.pad))])

        nx, nz = args.nx, args.nz
        if self.symmetry_x:
            x = np.linspace(-self.pad, nx + self.pad - 1, nx + 2 * self.pad) * delta
        else:
            x = np.linspace(-nx // 2 - self.pad, nx // 2 + self.pad - 1, nx + 2 * self.pad) * delta
        z = np.linspace(-nz // 2 - self.pad, nz // 2 + self.pad - 1, nz + 2 * self.pad) * delta

        # Coordinate set-up
        zz, xx = np.meshgrid(z, x)
        self.nx = zz.shape[0]
        self.nz = zz.shape[1]

        # incident electric and magnetic fields definition on the Yee grid
        fast = np.exp(1j * (k * zz))
        fast_z = np.exp(1j * (k * (zz + delta / 2)))
        self.fast = ops.zeros((1, 2, fast.shape[0], fast.shape[1]), ms.float32)
        self.fast[0, 0, :, :] = Tensor.from_numpy(np.real(fast))
        self.fast[0, 1, :, :] = Tensor.from_numpy(np.imag(fast))

        self.fast_z = ops.zeros((1, 2, fast_z.shape[0], fast_z.shape[1]), ms.float32)
        self.fast_z[0, 0, :, :] = Tensor.from_numpy(np.real(fast_z))
        self.fast_z[0, 1, :, :] = Tensor.from_numpy(np.imag(fast_z))

        # perfectly-matched-layer set up
        m = 4
        const = 5
        rx_p = 1 + 1j * const * (xx - x[-1] + args.pml_thickness * delta) ** m
        rx_p[0:-args.pml_thickness, :] = 0
        rx_n = 1 + 1j * const * (xx - x[0] - args.pml_thickness * delta) ** m
        rx_n[args.pml_thickness::, :] = 0
        rx = rx_p + rx_n
        if self.symmetry_x:
            rx[0:-args.pml_thickness:, :] = 1
        else:
            rx[args.pml_thickness:-args.pml_thickness, :] = 1

        rz_p = 1 + 1j * const * (zz - z[-1] + args.pml_thickness * delta) ** m
        rz_p[:, 0:-args.pml_thickness] = 0
        rz_n = 1 + 1j * const * (zz - z[0] - args.pml_thickness * delta) ** m
        rz_n[:, args.pml_thickness::] = 0
        rz = rz_p + rz_n
        rz[:, args.pml_thickness:-args.pml_thickness] = 1

        rx_inverse = 1 / rx
        rz_inverse = 1 / rz

        self.rx_inverse = ops.zeros((1, 2, rx_inverse.shape[0], rx_inverse.shape[1]), ms.float32)
        self.rx_inverse[0, 0, :, :] = Tensor.from_numpy(np.real(rx_inverse))
        self.rx_inverse[0, 1, :, :] = Tensor.from_numpy(np.imag(rx_inverse))

        self.rz_inverse = ops.zeros((1, 2, rz_inverse.shape[0], rz_inverse.shape[1]), ms.float32)
        self.rz_inverse[0, 0, :, :] = Tensor.from_numpy(np.real(rz_inverse))
        self.rz_inverse[0, 1, :, :] = Tensor.from_numpy(np.imag(rz_inverse))

        # Gradient and laplacian kernels set up
        self.gradient_h_z = ops.zeros((2, 1, 1, 3), ms.float32)
        self.gradient_h_z[:, :, 0, :] = ms.Tensor([-1 / delta, +1 / delta, 0])
        self.gradient_h_x = ops.transpose(self.gradient_h_z, (0, 1, 3, 2))
        self.gradient_h_z_ho = ops.zeros((2, 1, 1, 5), ms.float32)
        self.gradient_h_z_ho[:, :, 0, :] = ms.Tensor(
            [1 / 24 / delta, -9 / 8 / delta, +9 / 8 / delta, -1 / 24 / delta, 0])
        self.gradient_h_x_ho = ops.transpose(self.gradient_h_z_ho, (0, 1, 3, 2))

        self.gradient_e_z = ops.zeros((2, 1, 1, 3), ms.float32)
        self.gradient_e_z[:, :, 0, :] = ms.Tensor([0, -1 / delta, +1 / delta])
        self.gradient_e_x = ops.transpose(self.gradient_e_z, (0, 1, 3, 2))
        self.gradient_e_z_ho = ops.zeros((2, 1, 1, 5), ms.float32)
        self.gradient_e_z_ho[:, :, 0, :] = ms.Tensor(
            [0, 1 / 24 / delta, -9 / 8 / delta, +9 / 8 / delta, -1 / 24 / delta])
        self.gradient_e_x_ho = ops.transpose(self.gradient_e_z_ho, (0, 1, 3, 2))

        self.dd_z_fast = self.dd_z(self.fast)[:, :, self.pad:-self.pad:, :]
        self.dd_z_ho_fast = self.dd_z_ho(self.fast)[:, :, self.pad:-self.pad:, :]

    def construct(self, scat_pot, ri_value):
        """construct"""
        diff_x, diff_z = 0, 0
        total, diff = 0, 0
        if self.mode == 'te':
            epsillon = scat_pot * ri_value ** 2
            epsillon = mnp.where(epsillon > 1.0, epsillon, ms.Tensor([1], dtype=ms.float32))
            x = self.model(scat_pot)
            total = ops.concat((x[:, 0:1, :, :] + 1, x[:, 1:2, :, :]), 1)

            ey = self.complex_multiplication(total[:, 0:2, :, :],
                                             self.fast[:, :, self.pad:-self.pad:, self.pad:-self.pad:])
            ey_i = self.fast
            ey_s = ey - ey_i[:, :, self.pad:-self.pad:, self.pad:-self.pad:]

            if self.symmetry_x:
                ey_s = self.padding_ref(ey_s)
            else:
                ey_s = self.padding_zero(ey_s)

            if self.high_order == 2:
                diff = self.dd_x_pml(ey_s)[:, :, :, self.pad:-self.pad] \
                       + self.dd_z_pml(ey_s)[:, :, self.pad:-self.pad, :] \
                       + self.dd_z_fast \
                       + self.k ** 2 * (epsillon * ey)

            elif self.high_order == 4:
                diff = self.dd_x_ho_pml(ey_s)[:, :, :, self.pad:-self.pad] \
                       + self.dd_z_ho_pml(ey_s)[:, :, self.pad:-self.pad, :] \
                       + self.dd_z_ho_fast \
                       + self.k ** 2 * (epsillon * ey)

        elif self.mode == 'tm':
            epsillon = scat_pot * ri_value ** 2
            epsillon_x = mnp.where(epsillon[:, 0:1, :, :] > 1.0, epsillon[:, 0:1, :, :],
                                   ms.Tensor([1], dtype=ms.float32))
            epsillon_z = mnp.where(epsillon[:, 1:2, :, :] > 1.0, epsillon[:, 1:2, :, :],
                                   ms.Tensor([1], dtype=ms.float32))

            x = self.model(scat_pot)
            total = ops.concat((x[:, 0:1, :, :] + 1, x[:, 1:4, :, :]), 1)

            ex = self.complex_multiplication(total[:, 0:2, :, :],
                                             self.fast[:, :, self.pad:-self.pad:, self.pad:-self.pad:])
            ex_i = self.fast
            ex_s = ex - ex_i[:, :, self.pad:-self.pad:, self.pad:-self.pad:]

            ez_s = self.complex_multiplication(
                total[:, 2:4, :, :], self.fast_z[:, :, self.pad:-self.pad:, self.pad:-self.pad:])

            if self.symmetry_x:
                ex_s = self.padding_zero(ex_s)
                ez_s = self.padding_ref(ez_s)
                ex_s[:, :, 0:self.pad, :] = mnp.flip(ex_s[:, :, self.pad:2 * self.pad, :], [2])
                ez_s[:, :, 0:self.pad, :] = -ez_s[:, :, 0:self.pad, :]
            else:
                ex_s = self.padding_zero(ex_s)
                ez_s = self.padding_zero(ez_s)

            if self.high_order == 2:
                diff_x = self.dd_z_pml(ex_s)[:, :, self.pad:-self.pad:, :] \
                         + self.dd_z_fast \
                         - self.dd_zx(ez_s)[:, :, self.pad // 2:-self.pad // 2:, self.pad // 2:-self.pad // 2] \
                         + self.k ** 2 * (epsillon_x * ex)

                diff_z = self.dd_x_pml(ez_s)[:, :, :, self.pad:-self.pad] \
                         - self.dd_xz(ex_s)[:, :, self.pad // 2:-self.pad // 2:, self.pad // 2:-self.pad // 2] \
                         + self.k ** 2 * (epsillon_z * ez_s)

            elif self.high_order == 4:
                diff_x = self.dd_z_ho_pml(ex_s)[:, :, self.pad:-self.pad:, :] \
                         + self.dd_z_ho_fast \
                         - self.dd_zx_ho_pml(ez_s)[:, :, self.pad // 2:-self.pad // 2:, self.pad // 2:-self.pad // 2] \
                         + self.k ** 2 * (epsillon_x * ex)

                diff_z = self.dd_x_ho_pml(ez_s)[:, :, :, self.pad:-self.pad] \
                         - self.dd_xz_ho_pml(ex_s)[:, :, self.pad // 2:-self.pad // 2:, self.pad // 2:-self.pad // 2] \
                         + self.k ** 2 * (epsillon_z * ez_s[:, :, self.pad:- self.pad:, self.pad:-self.pad:])

            diff = ops.concat((diff_x, diff_z), 1)

        return diff, total

    def complex_multiplication(self, a, b):
        r_p = ops.mul(a[:, 0:1, :, :], b[:, 0:1, :, :]) - \
              ops.mul(a[:, 1:2, :, :], b[:, 1:2, :, :])
        i_p = ops.mul(a[:, 0:1, :, :], b[:, 1:2, :, :]) + \
              ops.mul(a[:, 1:2, :, :], b[:, 0:1, :, :])
        return ops.concat((r_p, i_p), 1)

    def complex_conjugate(self, a):
        return ops.concat((-a[:, 1:2, :, :], a[:, 0:1, :, :]), 1)

    def d_e_x(self, x):
        return ops.conv2d(x, self.gradient_e_x, padding=0, groups=2)

    def d_e_x_ho(self, x):
        return ops.conv2d(x, self.gradient_e_x_ho, padding=0, groups=2)

    def d_h_x(self, x):
        return ops.conv2d(x, self.gradient_h_x, padding=0, groups=2)

    def d_h_x_ho(self, x):
        return ops.conv2d(x, self.gradient_h_x_ho, padding=0, groups=2)

    def d_e_z(self, x):
        return ops.conv2d(x, self.gradient_e_z, padding=0, groups=2)

    def d_e_z_ho(self, x):
        return ops.conv2d(x, self.gradient_e_z_ho, padding=0, groups=2)

    def d_h_z(self, x):
        return ops.conv2d(x, self.gradient_h_z, padding=0, groups=2)

    def d_h_z_ho(self, x):
        return ops.conv2d(x, self.gradient_h_z_ho, padding=0, groups=2)

    def dd_x(self, x):
        return self.d_h_x(self.d_e_x(x))

    def dd_x_ho(self, x):
        return self.d_h_x_ho(self.d_e_x_ho(x))

    def dd_x_pml(self, x):
        return self.complex_multiplication(self.rx_inverse[:, :, 2:-2, :], self.d_h_x(
            self.complex_multiplication(self.rx_inverse[:, :, 1:-1, :], self.d_e_x(x))))

    def dd_x_ho_pml(self, x):
        return self.complex_multiplication(self.rx_inverse[:, :, 4:-4, :], self.d_h_x_ho(
            self.complex_multiplication(self.rx_inverse[:, :, 2:-2, :], self.d_e_x_ho(x))))

    def dd_z(self, x):
        return self.d_h_z(self.d_e_z(x))

    def dd_z_ho(self, x):
        return self.d_h_z_ho(self.d_e_z_ho(x))

    def dd_z_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, :, 2:-2], self.d_h_z(
            self.complex_multiplication(self.rz_inverse[:, :, :, 1:-1], self.d_e_z(x))))

    def dd_z_ho_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, :, 4:-4], self.d_h_z_ho(
            self.complex_multiplication(self.rz_inverse[:, :, :, 2:-2], self.d_e_z_ho(x))))

    def dd_zx(self, x):
        return self.d_h_z(self.d_e_x(x))

    def dd_zx_ho(self, x):
        return self.d_h_z_ho(self.d_e_x_ho(x))

    def dd_zx_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, 1:-1, 1:-1], self.d_h_z(
            self.complex_multiplication(self.rx_inverse[:, :, 1:-1, :], self.d_e_x(x))))

    def dd_zx_ho_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, 2:-2, 2:-2], self.d_h_z_ho(
            self.complex_multiplication(self.rx_inverse[:, :, 2:-2, :], self.d_e_x_ho(x))))

    def dd_xz(self, x):
        return self.d_h_x(self.d_e_z(x))

    def dd_xz_ho(self, x):
        return self.d_h_x_ho(self.d_e_z_ho(x))

    def dd_xz_pml(self, x):
        return self.complex_multiplication(self.rx_inverse[:, :, 1:-1, 1:-1], self.d_h_x(
            self.complex_multiplication(self.rz_inverse[:, :, :, 1:-1], self.d_e_z(x))))

    def dd_xz_ho_pml(self, x):
        return self.complex_multiplication(self.rx_inverse[:, :, 2:-2, 2:-2], self.d_h_x_ho(
            self.complex_multiplication(self.rz_inverse[:, :, :, 2:-2], self.d_e_z_ho(x))))


class UNet(nn.Cell):
    """UNet"""
    def __init__(self, in_channels=1, out_channels=2, depth=5, wf=6, norm='weight', up_mode='upconv'):
        super(UNet, self).__init__()
        if up_mode not in ('upconv', 'upsample'):
            raise ValueError("illegal upmode")
        self.down_path = nn.CellList()
        self.up_path = nn.CellList()

        prev_channels = in_channels

        for i in range(depth):
            if i != depth - 1:
                self.down_path.append(UNetConvBlock(prev_channels, [wf * (2 ** i), wf * (2 ** i)], 3, 0, norm))
                prev_channels = int(wf * (2 ** i))
                self.down_path.append(nn.AvgPool2d(2, 2))
            else:
                self.down_path.append(UNetConvBlock(prev_channels, [wf * (2 ** i), wf * (2 ** (i - 1))], 3, 0, norm))
                prev_channels = int(wf * (2 ** (i - 1)))

        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, [wf * (2 ** i), int(wf * (2 ** (i - 1)))], up_mode, 3, 0, norm))
            prev_channels = int(wf * (2 ** (i - 1)))

        self.last_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=1, padding=0, has_bias=False,
                                   pad_mode='pad')

    def construct(self, scat_pot):
        """construct"""
        blocks = []
        x = scat_pot
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i % 2 == 0 and i != (len(self.down_path) - 1):
                blocks.append(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last_conv(x)
        return x


class UNetConvBlock(nn.Cell):
    """UNet Convolutional Block"""
    def __init__(self, in_size, out_size, kersize, padding, norm):
        super(UNetConvBlock, self).__init__()
        block = []
        if norm == 'weight':
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append(nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                   padding=int(0), has_bias=True, pad_mode='pad'))
            block.append(nn.CELU())
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))

            block.append(nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                   padding=int(0), has_bias=True, pad_mode='pad'))
        elif norm == 'batch':
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append(nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                   padding=int(padding), has_bias=True, pad_mode='pad'))
            block.append(nn.BatchNorm2d(out_size[0]))
            block.append(nn.CELU())

            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append(nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                   padding=int(padding), has_bias=True, pad_mode='pad'))
            block.append(nn.BatchNorm2d(out_size[1]))

        elif norm == 'no':
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append(nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                   padding=int(0), has_bias=True, pad_mode='pad'))
            block.append(nn.CELU())
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append(nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                   padding=int(0), has_bias=True, pad_mode='pad'))

        self.block = nn.SequentialCell(block)

    def construct(self, x):
        """construct"""
        out = self.block(x)
        return out


class UNetUpBlock(nn.Cell):
    """UNet Up Block"""
    def __init__(self, in_size, out_size, up_mode, kersize, padding, norm):
        super(UNetUpBlock, self).__init__()
        block = []
        if up_mode == 'upconv':
            block.append(nn.Conv2dTranspose(in_size, in_size, kernel_size=2, stride=2, has_bias=False))
        elif up_mode == 'upsample':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2))
            block.append(nn.Conv2d(in_size, in_size, kernel_size=1, has_bias=False))

        self.block = nn.SequentialCell(block)
        self.conv_block = UNetConvBlock(in_size * 2, out_size, kersize, padding, norm)

    def construct(self, x, bridge):
        """construct"""
        up = self.block(x)
        out = ops.concat([up, bridge], 1)
        out = self.conv_block(out)
        return out
