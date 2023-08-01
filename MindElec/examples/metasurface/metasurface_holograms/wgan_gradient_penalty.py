"""
gradient penalty module
"""
import os.path

import scipy.io as sci
from mindspore import Parameter, nn, ops, Tensor
from mindspore.common.initializer import initializer, HeUniform
from mindspore.ops import operations as P

from resnet34 import InstanceNorm2d, _resnet34_ln


def he_uniform(out_channels: int, in_channels: int = 1, is_bias: bool = False):
    distribution = initializer(HeUniform(mode="fan_in"), (out_channels, in_channels)).init_data()
    dim_fit = distribution
    return dim_fit[:, 0] if is_bias else dim_fit


class FixedParamDense(nn.Cell):
    def __init__(self, param):
        super(FixedParamDense, self).__init__()
        self.matmul = P.MatMul()
        self.weight = Parameter(param, name="w", requires_grad=False)  # (1600, 1600)

    def construct(self, x):
        out = self.matmul(x, self.weight)
        return out


class Generator(nn.Cell):
    """
    generator
    """
    def __init__(self, data_path):
        """
        generator
        """
        super().__init__()
        self.resnet34 = _resnet34_ln(img_channel=1, first_stride=1, logit=False)
        print(self.resnet34)
        path = os.path.join(data_path, 'Wr_matrix.mat')
        data = sci.loadmat(path)
        wr = data['W_r_real']
        wm = data['W_r_im']
        wr = wr.astype('float32')
        wm = wm.astype('float32')
        self.transpose = ops.Transpose()

        self.fc_real = FixedParamDense(self.transpose(Tensor(wr), (1, 0)))
        self.fc_im = FixedParamDense(self.transpose(Tensor(wm), (1, 0)))

        self.mul = P.Mul()
        self.sqrt = P.Sqrt()
        self.div = P.Div()
        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims()

    def construct(self, x):
        """
        generator network construction
        Args:
            x: input

        Returns:

        """
        x = self.resnet34(x)  # x in:[batch 1 40 40] out: [batch 1600]
        out_real = self.fc_real(x)  # [batch 1600]
        out_im = self.fc_im(x)  # [batch 1600]
        out = self.mul(out_real, out_real) + self.mul(out_im, out_im)
        out = self.sqrt(out)
        out = self.div(out, self.expand_dims(out.max(axis=1), 1))
        out_shape_0 = out.shape[0]
        out = self.reshape(out, (out_shape_0, 1, 40, 40))

        return out


class Discriminator(nn.Cell):
    """
    discriminator
    """
    def __init__(self, channels=2):
        super().__init__()
        self.main_module = nn.SequentialCell()
        # Image [batch 2 40 40]
        self.main_module.append(
            nn.Conv2d(channels, 256, 4, stride=2, pad_mode="pad", padding=1, has_bias=True, weight_init="HeUniform",
                      bias_init=he_uniform(256, is_bias=True)))  # [batch 256 20 20]
        self.main_module.append(InstanceNorm2d(256))  # IN只支持GPU？
        self.main_module.append(nn.LeakyReLU(alpha=0.2))

        # State (256x16x16)
        self.main_module.append(
            nn.Conv2d(256, 512, 4, stride=2, pad_mode="pad", padding=1, has_bias=True, weight_init="HeUniform",
                      bias_init=he_uniform(512, is_bias=True)))  # [batch 512 10 10]
        self.main_module.append(InstanceNorm2d(512))
        self.main_module.append(nn.LeakyReLU(alpha=0.2))

        # State (512x8x8)
        self.main_module.append(
            nn.Conv2d(512, 1024, 5, stride=2, pad_mode="pad", padding=1, has_bias=True, weight_init="HeUniform",
                      bias_init=he_uniform(1024, is_bias=True)))  # [batch 1024 4 4]
        self.main_module.append(InstanceNorm2d(1024))
        self.main_module.append(nn.LeakyReLU(alpha=0.2))

        self.output = nn.SequentialCell()
        self.output.append(
            nn.Conv2d(1024, 1, 4, stride=1, pad_mode="pad", padding=0, has_bias=True, weight_init="HeUniform",
                      bias_init="zeros"))  # [batch 1 1 1]
        # The output of D is no longer a probability, we do not apply sigmoid at the output of D.

        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.main_module(x)  # [batch 1024 4 4]
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        bs = x.shape[0]
        return self.reshape(x, (bs, 1024 * 4 * 4))  # [batch 1024*4*4]
