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
# ==============================================================================
"""ResUnet with Attention Spatial Channel Generator"""
import mindspore.nn as nn
from mindspore import ops


class DoubleConvLayer(nn.Cell):
    """
    construction of a double convolutional layer using
    SAME padding
    RELU nonlinear activation function
    :param in_channels: in channel number
    :param filter_size: size of convolutional filter
    :param size: number of filters
    :param dropout: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :return: output of a double convolutional layer
    """
    def __init__(self, in_channels, filter_size, size, dropout):
        super(DoubleConvLayer, self).__init__()
        self.dropout = dropout
        self.double_conv_sequential = nn.SequentialCell(
            nn.Conv2d(in_channels, size, (filter_size, filter_size), has_bias=True),
            nn.BatchNorm2d(size, eps=1e-3, momentum=0.99),
            nn.ReLU(),
            nn.Conv2d(size, size, (filter_size, filter_size), has_bias=True),
            nn.BatchNorm2d(size, eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        self.conv_size1 = nn.Conv2d(in_channels, size, (1, 1), has_bias=True)
        self.batch_normal = nn.BatchNorm2d(size, eps=1e-3, momentum=0.99)
        if dropout > 0:
            self.drop = nn.Dropout(p=dropout)

    def construct(self, x):
        """
        :param x: input
        :return: output of a double convolutional layer
        """
        conv = self.double_conv_sequential(x)
        if self.dropout > 0:
            conv = self.drop(conv)
        shortcut = self.conv_size1(x)
        shortcut = self.batch_normal(shortcut)
        res_path = shortcut + conv
        return res_path


class SeBlock(nn.Cell):
    """
    self attention squeeze-excitation block, attention mechanism on channel dimension
    :param in_channels: in channel number
    :param out_dim: output channel number
    :param ratio: ratio
    :return: attention weighted on channel dimension feature map
    """
    def __init__(self, in_channels, out_dim, ratio):
        super(SeBlock, self).__init__()
        self.out_dim = out_dim
        self.se_sequentialcell = nn.SequentialCell(
            nn.BatchNorm1d(in_channels, eps=1e-3, momentum=0.99),
            nn.Dense(in_channels, out_dim // ratio, has_bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim // ratio, eps=1e-3, momentum=0.99),
            nn.Dense(out_dim // ratio, out_dim, has_bias=True),
            nn.Sigmoid()
        )

    def construct(self, x):
        """
        :param x: input feature map
        :return: attention weighted on channel dimension feature map
        """
        # Squeeze: global average pooling
        x_s = ops.mean(x, axis=(2, 3), keep_dims=False)
        # Excitation: bottom-up top-down FCs
        x_e = self.se_sequentialcell(x_s)
        x_e = ops.reshape(x_e, (x_e.shape[0], self.out_dim, 1, 1))
        result = ops.mul(x, x_e)
        return result


class GatingSignal(nn.Cell):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param in_channels: in channel number
    :param out_size: output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    def __init__(self, in_channels, out_size):
        super(GatingSignal, self).__init__()
        self.gating_signal_sequentialcell = nn.SequentialCell(
            nn.Conv2d(in_channels, out_size, (1, 1), has_bias=True),
            nn.BatchNorm2d(out_size, eps=1e-3, momentum=0.99),
            nn.ReLU()
        )

    def construct(self, x):
        """
        :param x: down-dim feature map
        :return: the gating feature map with the same dimension of the up layer feature map
        """
        x = self.gating_signal_sequentialcell(x)
        return x


class AttentionBlock(nn.Cell):
    """
    self gated attention, attention mechanism on spatial dimension
    :param in_channels: in channel number
    :param inter_shape: intermedium channel number
    :return: attention weighted on spatial dimension feature map
    """
    def __init__(self, in_channels, inter_shape):
        super(AttentionBlock, self).__init__()
        conv_imgs = 1024//in_channels
        gating_imgs = 512//in_channels
        self.conv_nobias = nn.Conv2d(in_channels, inter_shape, (1, 1))
        self.conv = nn.Conv2d(in_channels, inter_shape, (1, 1), has_bias=True)
        self.conv2dt = nn.Conv2dTranspose(inter_shape, inter_shape, (3, 3),
                                          stride=(conv_imgs//gating_imgs, conv_imgs//gating_imgs), has_bias=True)
        self.relu = nn.ReLU()
        self.conv_num1 = nn.Conv2d(inter_shape, 1, (1, 1), has_bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv_result = nn.Conv2d(in_channels, in_channels, (1, 1), has_bias=True)
        self.batch_normal = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.99)

    def construct(self, conv, gating):
        """
        :param conv: input feature map
        :param gating: gate signal, feature map from the lower layer
        :return: attention weighted on spatial dimension feature map
        """
        shape_x = conv.shape
        theta_x = self.conv_nobias(conv)
        phi_g = self.conv(gating)
        upsample_g = self.conv2dt(phi_g)
        concat_xg = upsample_g + theta_x
        act_xg = self.relu(concat_xg)
        psi = self.conv_num1(act_xg)
        sigmoid_xg = self.sigmoid(psi)
        upsample_psi = ops.tile(sigmoid_xg, (1, shape_x[1], 1, 1))
        y = ops.mul(upsample_psi, conv)
        result = self.conv_result(y)
        result_bn = self.batch_normal(result)
        return result_bn


class AttentionResUnet(nn.Cell):
    """
    Rsidual UNet construction, with attention gate
    convolution: 3*3 SAME padding
    pooling: 2*2 VALID padding
    upsampling: 3*3 VALID padding
    final convolution: 1*1
    :param input_channel: in channel number
    :param output_mask_channel: output channel number
    :param filter_num: filter number
    :param filter_size: filter size
    :param merge: bool of merge model
    :param latent_z: bool of latent_z
    :return: model
    """
    def __init__(self, input_channel, output_mask_channel, filter_num, filter_size, merge=False, latent_z=False):
        super(AttentionResUnet, self).__init__()
        self.axis = 1
        self.se_ratio = 16
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.merge = merge
        self.latent_z = latent_z

        self.double_conv_layer1 = DoubleConvLayer(input_channel, self.filter_size, self.filter_num, dropout=0.5)
        self.max_pooling1 = nn.MaxPool2d(2, 2)
        self.double_conv_layer2 = DoubleConvLayer(self.filter_num, self.filter_size, 2 * self.filter_num, dropout=0.5)
        self.max_pooling2 = nn.MaxPool2d(2, 2)
        self.double_conv_layer3 = DoubleConvLayer(2 * self.filter_num, self.filter_size, 4 * self.filter_num,
                                                  dropout=0.5)
        self.max_pooling3 = nn.MaxPool2d(2, 2)
        self.double_conv_layer4 = DoubleConvLayer(4 * self.filter_num, self.filter_size, 8 * self.filter_num,
                                                  dropout=0.5)
        self.max_pooling4 = nn.MaxPool2d(2, 2)
        self.double_conv_layer5 = DoubleConvLayer(8 * self.filter_num, self.filter_size, 16 * self.filter_num,
                                                  dropout=0.5)

        if self.latent_z:
            self.gating_signal1 = GatingSignal(16 * self.filter_num + 51, 8 * self.filter_num)
            self.conv2d_transpose1 = nn.Conv2dTranspose(16 * self.filter_num + 51, 16 * self.filter_num, (2, 2),
                                                        stride=(2, 2), has_bias=True)
        else:
            self.gating_signal1 = GatingSignal(16 * self.filter_num + 1, 8 * self.filter_num)
            self.conv2d_transpose1 = nn.Conv2dTranspose(16 * self.filter_num + 1, 16 * self.filter_num, (2, 2),
                                                        stride=(2, 2), has_bias=True)

        self.gating_signal2 = GatingSignal(8 * self.filter_num, 4 * self.filter_num)
        self.gating_signal3 = GatingSignal(4 * self.filter_num, 2 * self.filter_num)
        self.gating_signal4 = GatingSignal(2 * self.filter_num, self.filter_num)

        self.conv2d_transpose2 = nn.Conv2dTranspose(8 * self.filter_num, 8 * self.filter_num, (2, 2),
                                                    stride=(2, 2), has_bias=True)
        self.conv2d_transpose3 = nn.Conv2dTranspose(4 * self.filter_num, 4 * self.filter_num, (2, 2),
                                                    stride=(2, 2), has_bias=True)
        self.conv2d_transpose4 = nn.Conv2dTranspose(2 * self.filter_num, 2 * self.filter_num, (2, 2),
                                                    stride=(2, 2), has_bias=True)

        self.attention_block1 = AttentionBlock(8 * self.filter_num, 8 * self.filter_num)
        self.attention_block2 = AttentionBlock(4 * self.filter_num, 4 * self.filter_num)
        self.attention_block3 = AttentionBlock(2 * self.filter_num, 2 * self.filter_num)
        self.attention_block4 = AttentionBlock(self.filter_num, self.filter_num)

        self.se_conv_16_layer = SeBlock(
            16 * self.filter_num + 8 * self.filter_num, 16 * self.filter_num + 8 * self.filter_num, ratio=self.se_ratio)
        self.se_conv_32_layer = SeBlock(
            8 * self.filter_num + 4 * self.filter_num, 8 * self.filter_num + 4 * self.filter_num, ratio=self.se_ratio)
        self.se_conv_64_layer = SeBlock(
            4 * self.filter_num + 2 * self.filter_num, 4 * self.filter_num + 2 * self.filter_num, ratio=self.se_ratio)
        self.se_conv_128_layer = SeBlock(
            2 * self.filter_num + self.filter_num, 2 * self.filter_num + self.filter_num, ratio=self.se_ratio)

        self.up_conv_16_layer = DoubleConvLayer(
            16 * self.filter_num + 8 * self.filter_num, self.filter_size, 8 * self.filter_num, dropout=0.5)
        self.up_conv_32_layer = DoubleConvLayer(
            8 * self.filter_num + 4 * self.filter_num, self.filter_size, 4 * self.filter_num, dropout=0.5)
        self.up_conv_64_layer = DoubleConvLayer(
            4 * self.filter_num + 2 * self.filter_num, self.filter_size, 2 * self.filter_num, dropout=False)
        self.up_conv_128_layer = DoubleConvLayer(
            2 * self.filter_num + self.filter_num, self.filter_size, self.filter_num, dropout=False)

        self.conv_final_layer = nn.Conv2d(self.filter_num, output_mask_channel, kernel_size=(1, 1), pad_mode='valid',
                                          has_bias=True)
        self.batch_normal = nn.BatchNorm2d(output_mask_channel, eps=1e-3, momentum=0.99)
        self.tanh = nn.Tanh()
        self.concat = ops.Concat(axis=self.axis)

    def construct(self, inputs, merge_inputs=None, z_inputs=None):
        """Rsidual UNet Generator, with attention gate"""
        # Down-sampling layers
        # DownRes 1, double residual convolution + pooling   dropout, batch_norm=False
        conv_128 = self.double_conv_layer1(inputs)  # (8,128,128)  # comments are FILTER_NUM=8
        pool_64 = self.max_pooling1(conv_128)  # (8,64,64)
        # DownRes 2
        conv_64 = self.double_conv_layer2(pool_64)  # (16,64,64)
        pool_32 = self.max_pooling2(conv_64)  # (16,32,32)
        # DownRes 3
        conv_32 = self.double_conv_layer3(pool_32)  # (32,32,32)
        pool_16 = self.max_pooling3(conv_32)  # (32,16,16)
        # DownRes 4
        conv_16 = self.double_conv_layer4(pool_16)  # (64,16,16)
        pool_8 = self.max_pooling4(conv_16)  # (64,8,8)
        # DownRes 5, convolution only
        conv_8 = self.double_conv_layer5(pool_8)  # (128,8,8)

        # merge the other features, including Re, u_scaling and latent_z
        if self.merge:
            # merge_inputs # (8,8,1)
            conv_8 = self.concat([conv_8, merge_inputs])  # (129,8,8)
        if self.latent_z:
            # merge_inputs # (8,8,1)
            # z_inputs # (8,8,50)
            conv_8 = self.concat([conv_8, merge_inputs, z_inputs])  # (8,8,179)

        # Up-sampling layers
        gating_16 = self.gating_signal1(conv_8)
        up_16 = self.conv2d_transpose1(conv_8)  # (None, 128, 16, 16)

        # add channel attention block after spatial attention block
        att_16 = self.attention_block1(conv_16, gating_16)  # (None, 64, 16, 16)
        up_16 = self.concat([up_16, att_16])  # (None, 192, 16, 16)
        se_conv_16 = self.se_conv_16_layer(up_16)  # (None, 192, 16, 16)
        up_conv_16 = self.up_conv_16_layer(se_conv_16)  # (None, 64, 16, 16)

        # UpRes 7
        gating_32 = self.gating_signal2(up_conv_16)
        up_32 = self.conv2d_transpose2(up_conv_16)  # (None, 64, 32, 32)

        att_32 = self.attention_block2(conv_32, gating_32)  # (None, 32, 32, 32)
        up_32 = self.concat([up_32, att_32])  # (None, 96, 32, 32)
        se_conv_32 = self.se_conv_32_layer(up_32)  # (None, 96, 32, 32)
        up_conv_32 = self.up_conv_32_layer(se_conv_32)  # (None, 32, 32, 32)

        # UpRes 8
        gating_64 = self.gating_signal3(up_conv_32)
        up_64 = self.conv2d_transpose3(up_conv_32)  # (None, 32, 64, 64)

        att_64 = self.attention_block3(conv_64, gating_64)  # (None, 64, 64, 16)
        up_64 = self.concat([up_64, att_64])  # (None, 48, 64, 64)
        se_conv_64 = self.se_conv_64_layer(up_64)  # (None, 48, 64, 64)
        up_conv_64 = self.up_conv_64_layer(se_conv_64)  # (None, 16, 64, 64)

        # UpRes 9
        gating_128 = self.gating_signal4(up_conv_64)
        up_128 = self.conv2d_transpose4(up_conv_64)  # (None, 16, 128, 128)

        att_128 = self.attention_block4(conv_128, gating_128)  # (None, 8, 128, 128)
        up_128 = self.concat([up_128, att_128])  # (None, 24, 128, 128)
        se_conv_128 = self.se_conv_128_layer(up_128)  # (None, 24, 128, 128)
        up_conv_128 = self.up_conv_128_layer(se_conv_128)  # (None, 8, 128, 128)

        conv_final = self.conv_final_layer(up_conv_128)
        conv_final = self.batch_normal(conv_final)
        conv_final = self.tanh(conv_final)
        return conv_final
