# ============================================================================
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
"""cae-Transformer model"""
from mindspore import nn, ops


class CaeEncoder(nn.Cell):
    """
    encoder net
    """
    def __init__(self, conv_kernel_size, maxpool_kernel_size, maxpool_stride, channels_encoder, channels_dense):
        super(CaeEncoder, self).__init__()
        self.conv1 = nn.Conv2d(channels_encoder[0], channels_encoder[1], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv2 = nn.Conv2d(channels_encoder[1], channels_encoder[2], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv3 = nn.Conv2d(channels_encoder[2], channels_encoder[3], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv4 = nn.Conv2d(channels_encoder[3], channels_encoder[4], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv5 = nn.Conv2d(channels_encoder[4], channels_encoder[5], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv6 = nn.Conv2d(channels_encoder[5], channels_encoder[6], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')

        self.max_pool2d = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)

        self.relu = nn.ReLU()

        self.flatten = ops.Flatten()

        self.dense1 = nn.Dense(channels_dense[0], channels_dense[1], weight_init='HeUniform', activation='relu')
        self.dense2 = nn.Dense(channels_dense[1], channels_dense[2], weight_init='HeUniform', activation='relu')
        self.dense3 = nn.Dense(channels_dense[2], channels_dense[3], weight_init='HeUniform')

        self.reshape = ops.Reshape()

    def construct(self, x):
        """
        encoder construct
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class CaeDecoder(nn.Cell):
    """
    decoder net
    """
    def __init__(self, data_dimension, conv_kernel_size, channels_decoder, channels_dense):
        super(CaeDecoder, self).__init__()
        self.dense1 = nn.Dense(channels_dense[3], channels_dense[2], weight_init='HeUniform', activation='relu')
        self.dense2 = nn.Dense(channels_dense[2], channels_dense[1], weight_init='HeUniform', activation='relu')
        self.dense3 = nn.Dense(channels_dense[1], channels_dense[0], weight_init='HeUniform', activation='relu')
        self.reshape = ops.Reshape()
        self.conv1 = nn.Conv2d(channels_decoder[0], channels_decoder[1], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv2 = nn.Conv2d(channels_decoder[1], channels_decoder[2], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv3 = nn.Conv2d(channels_decoder[2], channels_decoder[3], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv4 = nn.Conv2d(channels_decoder[3], channels_decoder[4], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv5 = nn.Conv2d(channels_decoder[4], channels_decoder[5], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv6 = nn.Conv2d(channels_decoder[5], channels_decoder[6], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')
        self.conv7 = nn.Conv2d(channels_decoder[6], channels_decoder[7], conv_kernel_size,
                               has_bias=True, weight_init='HeUniform')

        self.relu = nn.ReLU()

        self.data_dimension = data_dimension

        self.channels_decoder = channels_decoder

    def construct(self, x):
        """
        decoder construct
        """
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        x = self.reshape(x, (ops.shape(x)[0], self.channels_decoder[0],
                             round(pow(ops.shape(x)[-1]/self.channels_decoder[0], 0.5)), -1))

        x = self.conv1(x)
        x = self.relu(x)
        x = ops.ResizeNearestNeighbor((self.data_dimension[5], self.data_dimension[5]))(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = ops.ResizeNearestNeighbor((self.data_dimension[4], self.data_dimension[4]))(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = ops.ResizeNearestNeighbor((self.data_dimension[3], self.data_dimension[3]))(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = ops.ResizeNearestNeighbor((self.data_dimension[2], self.data_dimension[2]))(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = ops.ResizeNearestNeighbor((self.data_dimension[1], self.data_dimension[1]))(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = ops.ResizeNearestNeighbor((self.data_dimension[0], self.data_dimension[0]))(x)

        x = self.conv7(x)
        return x
