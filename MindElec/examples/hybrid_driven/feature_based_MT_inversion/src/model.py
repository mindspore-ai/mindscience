# copyright 2024 Huawei Technologies co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
# See the License for the specific language governing permissions and
# limitations under the license.
##==========================================================================
"""
    neural network structure definition
"""
import logging
from mindspore import ops, nn

# %%
class Swish(nn.Cell):
    """
    Swish
    """
    def construct(self, x):
        return x * ops.sigmoid(x)

class Sampling(nn.Cell):
    """
    Sampling
    """
    def construct(self, z_mean, z_log_var):
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = ops.normal(shape=(batch, dim), mean=0, stddev=1)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

## Latent space
class MeanModel(nn.Cell):
    """
    MeanModel
    """

    DEFINE1 = 1
    DEFINE2 = 2
    DEFINE3 = 3
    DEFINE16 = 16
    DEFINE32 = 32
    DEFINE64 = 64
    DEFINE256 = 256
    DEFINESAME = "same"

    def __init__(self):
        super().__init__()
        self.conv1d1 = nn.Conv1d(
            self.DEFINE1, self.DEFINE16, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn1 = nn.BatchNorm1d(self.DEFINE16)
        self.swish1 = Swish()

        self.conv1d2 = nn.Conv1d(
            self.DEFINE16, self.DEFINE16, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn2 = nn.BatchNorm1d(self.DEFINE16)
        self.swish2 = Swish()
        self.maxpool1d2 = nn.MaxPool1d(self.DEFINE2, self.DEFINE2)

        self.conv1d3 = nn.Conv1d(
            self.DEFINE16, self.DEFINE32, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn3 = nn.BatchNorm1d(self.DEFINE32)
        self.swish3 = Swish()

        self.conv1d4 = nn.Conv1d(
            self.DEFINE32, self.DEFINE32, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn4 = nn.BatchNorm1d(self.DEFINE32)
        self.swish4 = Swish()
        self.maxpool1d4 = nn.MaxPool1d(self.DEFINE2, self.DEFINE2)

        self.conv1d5 = nn.Conv1d(
            self.DEFINE32, self.DEFINE64, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn5 = nn.BatchNorm1d(self.DEFINE64)
        self.swish5 = Swish()

        self.conv1d6 = nn.Conv1d(
            self.DEFINE64, self.DEFINE64, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn6 = nn.BatchNorm1d(self.DEFINE64)
        self.swish6 = Swish()
        self.maxpool1d6 = nn.MaxPool1d(self.DEFINE2, self.DEFINE2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(self.DEFINE256, self.DEFINE16)  # todo
        self.dense2 = nn.Dense(self.DEFINE256, self.DEFINE16)

    def construct(self, x):
        """
        :param x: input
        :return: output
        """
        x = self.conv1d1(x)
        x = self.bn1(x)
        x = self.swish1(x)

        x = self.conv1d2(x)
        x = self.bn2(x)
        x = self.swish2(x)
        x = self.maxpool1d2(x)

        x = self.conv1d3(x)
        x = self.bn3(x)
        x = self.swish3(x)

        x = self.conv1d4(x)
        x = self.bn4(x)
        x = self.swish4(x)
        x = self.maxpool1d2(x)

        x = self.conv1d5(x)
        x = self.bn5(x)
        x = self.swish5(x)

        x = self.conv1d6(x)
        x = self.bn6(x)
        x = self.swish6(x)
        x = self.maxpool1d2(x)

        x = self.flatten(x)
        z_mean = self.dense1(x)
        z_log_var = self.dense2(x)
        return z_mean, z_log_var

class Encoder(nn.Cell):
    def __init__(self, mean_model):
        super().__init__()
        self.mean_model = mean_model
        self.sample = Sampling()

    def construct(self, inputs1):
        z_mean, z_log_var = self.mean_model(inputs1)
        encoder_output = self.sample(z_mean, z_log_var)
        return encoder_output

class Decoder(nn.Cell):
    """
    Decoder
    """

    DEFINE1 = 1
    DEFINE2 = 2
    DEFINE3 = 3
    DEFINE16 = 16
    DEFINE32 = 32
    DEFINE64 = 64
    DEFINE256 = 256
    DEFINESAME = "same"

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(self.DEFINE16, self.DEFINE256)
        self.swish1 = Swish()

        self.conv1d_trans2 = nn.Conv1dTranspose(
            self.DEFINE64,
            self.DEFINE32,
            self.DEFINE3,
            stride=self.DEFINE2,
            pad_mode=self.DEFINESAME,
        )  # 16 32
        self.conv1d2 = nn.Conv1d(
            self.DEFINE32, self.DEFINE32, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn2 = nn.BatchNorm1d(self.DEFINE32)
        self.swish2 = Swish()

        self.conv1d3 = nn.Conv1d(
            self.DEFINE32, self.DEFINE32, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn3 = nn.BatchNorm1d(self.DEFINE32)
        self.swish3 = Swish()

        self.conv1d_trans4 = nn.Conv1dTranspose(
            self.DEFINE32,
            self.DEFINE16,
            self.DEFINE3,
            stride=self.DEFINE2,
            pad_mode=self.DEFINESAME,
        )  # 32 16
        self.conv1d4 = nn.Conv1d(
            self.DEFINE16, self.DEFINE16, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn4 = nn.BatchNorm1d(self.DEFINE16)
        self.swish4 = Swish()

        self.conv1d5 = nn.Conv1d(
            self.DEFINE16, self.DEFINE16, self.DEFINE3, pad_mode=self.DEFINESAME
        )
        self.bn5 = nn.BatchNorm1d(self.DEFINE16)
        self.swish5 = Swish()

        self.conv1d_trans6 = nn.Conv1dTranspose(
            self.DEFINE16,
            self.DEFINE1,
            self.DEFINE3,
            stride=self.DEFINE2,
            pad_mode=self.DEFINESAME,
        )  # 64 1
        self.conv1d6 = nn.Conv1d(
            self.DEFINE1, self.DEFINE1, self.DEFINE3, pad_mode=self.DEFINESAME
        )

    def construct(self, decoder_inputs):
        """

        :param decoder_inputs: input
        :return: output
        """
        x = self.dense1(decoder_inputs)
        x = self.swish1(x)
        x = ops.reshape(x, (-1, self.DEFINE64, int(self.DEFINE32 / 8)))

        x = self.conv1d_trans2(x)  # 16 32
        x = self.conv1d2(x)
        x = self.bn2(x)
        x = self.swish2(x)

        x = self.conv1d3(x)
        x = self.bn3(x)
        x = self.swish3(x)

        x = self.conv1d_trans4(x)  # 32 16
        x = self.conv1d4(x)
        x = self.bn4(x)
        x = self.swish4(x)

        x = self.conv1d5(x)
        x = self.bn5(x)
        x = self.swish5(x)

        x = self.conv1d_trans6(x)  # 64 1
        x = self.conv1d6(x)
        return x

class Model(nn.Cell):
    """
    Model
    """

    def __init__(self):
        super().__init__()
        self.mean_model = MeanModel()
        self.sampling = Sampling()
        self.decoder = Decoder()

    def construct(self, inputs2):
        z_mean, z_log_var = self.mean_model(inputs2)
        encoder_output = self.sampling(z_mean, z_log_var)
        outputs2 = self.decoder(encoder_output)
        return outputs2, z_mean, z_log_var

class KLLossNet(nn.Cell):
    """
    KLLossNet
    """
    def __init__(self, net1):
        super().__init__()
        self.net = net1

    def construct(self, z_mean, z_log_var, kl_weight):
        kl_loss = (
            kl_weight
            * -0.5
            * ops.reduce_mean(
                z_log_var - ops.square(z_mean) - ops.exp(z_log_var) + 1
            )
        )
        return kl_loss

class SSIM(nn.Cell):
    """
    SSIM
    """
    def construct(self, evaluation, target):
        """

        :param evaluation:
        :param target:
        :return:
        """
        mu_x = ops.mean(evaluation)
        mu_y = ops.mean(target)
        sigma_x = ops.sqrt(ops.mean((evaluation - mu_x) ** 2))
        sigma_y = ops.sqrt(ops.mean((target - mu_y) ** 2))
        sigma = ops.mean((evaluation - mu_x) * (target - mu_y))

        data_range = ops.max(ops.abs(target))[0] - ops.min(ops.abs(target))[0]
        c1 = data_range * 1e-2
        c2 = data_range * 3e-2
        try:
            ssim_score = ((2 * mu_x * mu_y + c1) * (2.0 * sigma + c2)) / (
                (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
            )
        except ZeroDivisionError as es:
            logging.error("Error %s", es)
            raise
        return ssim_score

class LossFuncNet(nn.Cell):
    """
    LossFuncNet
    """
    def __init__(self, net2):
        super().__init__()
        self.net = net2
        self.ssim = SSIM()
        self.kl_loss_net = KLLossNet(net2)

    def construct(self, inputsf, y_truew, lw_tens, ssim_w, kl_weight):
        """

        :param self:
        :param inputsf:
        :param y_truew:
        :param lw_tens:
        :param ssim_w:
        :param kl_weight:
        :return:
        """
        outputs1, z_mean, z_log_var = self.net(inputsf)
        kl_loss = self.kl_loss_net(z_mean, z_log_var, kl_weight)
        loss1 = ops.reduce_mean(
            ops.multiply(lw_tens, ops.square(y_truew - outputs1))
        )  # , axis=-2)
        loss2 = 1 - ops.reduce_mean(self.ssim(outputs1, y_truew))
        return kl_loss + loss1 + ssim_w * loss2

class MSELoss(nn.Cell):
    """
    MSELoss
    """
    def construct(self, y_true_a, y_pred, lw_tens):
        """

        :param y_true_a:
        :param y_pred:
        :param lw_tens:
        :return:
        """
        return ops.reduce_mean(
            ops.multiply(lw_tens, ops.square(y_true_a - y_pred))
        )  # , axis=-2)
