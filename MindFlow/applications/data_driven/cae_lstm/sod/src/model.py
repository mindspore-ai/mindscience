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
"""
cae-lstm model
"""
from mindspore import nn, ops, float32


class CaeEncoder(nn.Cell):
    """
    encoder net
    """
    def __init__(self, conv_kernel_size, maxpool_kernel_size, maxpool_stride, channels_encoder):
        super(CaeEncoder, self).__init__()
        self.conv1 = nn.Conv1d(channels_encoder[0], channels_encoder[1], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv2 = nn.Conv1d(channels_encoder[1], channels_encoder[2], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv3 = nn.Conv1d(channels_encoder[2], channels_encoder[3], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv4 = nn.Conv1d(channels_encoder[3], channels_encoder[4], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv5 = nn.Conv1d(channels_encoder[4], channels_encoder[5], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv6 = nn.Conv1d(channels_encoder[5], channels_encoder[6], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')

        self.max_pool1d = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)

        self.relu = nn.ReLU()

    def construct(self, x):
        """
        encoder construct
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1d(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool1d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool1d(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool1d(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool1d(x)

        x = self.conv6(x)
        x = self.max_pool1d(x)
        return x


class CaeDecoder(nn.Cell):
    """
    decoder net
    """
    def __init__(self, data_dimension, conv_kernel_size, channels_decoder):
        super(CaeDecoder, self).__init__()
        self.conv1 = nn.Conv1d(channels_decoder[0], channels_decoder[1], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv2 = nn.Conv1d(channels_decoder[1], channels_decoder[2], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv3 = nn.Conv1d(channels_decoder[2], channels_decoder[3], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv4 = nn.Conv1d(channels_decoder[3], channels_decoder[4], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv5 = nn.Conv1d(channels_decoder[4], channels_decoder[5], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv6 = nn.Conv1d(channels_decoder[5], channels_decoder[6], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv7 = nn.Conv1d(channels_decoder[6], channels_decoder[7], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')

        self.relu = nn.ReLU()

        self.data_dimension = data_dimension

    def construct(self, x):
        """
        decoder construct
        """
        x = self.conv1(x)
        x = ops.expand_dims(x, -1)
        x = ops.ResizeNearestNeighbor((self.data_dimension[5], 1))(x)
        x = ops.squeeze(x, -1)

        x = self.conv2(x)
        x = self.relu(x)
        x = ops.expand_dims(x, -1)
        x = ops.ResizeNearestNeighbor((self.data_dimension[4], 1))(x)
        x = ops.squeeze(x, -1)

        x = self.conv3(x)
        x = self.relu(x)
        x = ops.expand_dims(x, -1)
        x = ops.ResizeNearestNeighbor((self.data_dimension[3], 1))(x)
        x = ops.squeeze(x, -1)

        x = self.conv4(x)
        x = self.relu(x)
        x = ops.expand_dims(x, -1)
        x = ops.ResizeNearestNeighbor((self.data_dimension[2], 1))(x)
        x = ops.squeeze(x, -1)

        x = self.conv5(x)
        x = self.relu(x)
        x = ops.expand_dims(x, -1)
        x = ops.ResizeNearestNeighbor((self.data_dimension[1], 1))(x)
        x = ops.squeeze(x, -1)

        x = self.conv6(x)
        x = self.relu(x)
        x = ops.expand_dims(x, -1)
        x = ops.ResizeNearestNeighbor((self.data_dimension[0], 1))(x)
        x = ops.squeeze(x, -1)

        x = self.conv7(x)
        return x


class CaeNet(nn.Cell):
    """
    cae net
    """
    def __init__(self, data_dimension, conv_kernel_size, maxpool_kernel_size, maxpool_stride,
                 channels_encoder, channels_decoder):
        super(CaeNet, self).__init__()
        self.encoder = CaeEncoder(conv_kernel_size, maxpool_kernel_size, maxpool_stride, channels_encoder)
        self.decoder = CaeDecoder(data_dimension, conv_kernel_size, channels_decoder)

    def construct(self, x):
        lattent = self.encoder(x)
        x = self.decoder(lattent)
        return x


class Lstm(nn.Cell):
    """
    lstm net
    """
    def __init__(self, latent_size, hidden_size, num_layers):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense = nn.Dense(hidden_size, latent_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def construct(self, x):
        h0 = ops.zeros((self.num_layers, ops.shape(x)[0], self.hidden_size), float32)
        c0 = ops.zeros((self.num_layers, ops.shape(x)[0], self.hidden_size), float32)
        x, _ = self.lstm(x, (h0, c0))
        x = self.dense(x)
        return x
