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
"""Encoder & Decoder"""
from mindspore import ops, nn


class ConvLayer(nn.Cell):
    """Convolutional layer with batch normalization and ELU activation"""
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1
        self.conv = nn.Conv1d(in_channels=c_in,
                              out_channels=c_in,
                              kernel_size=3,
                              padding=padding,
                              pad_mode='pad')
        self.norm = nn.BatchNorm2d(num_features=c_in, momentum=0.9)
        self.activation = nn.ELU()
        self.pad = nn.Pad(((0, 0), (0, 0), (1, 1)), "CONSTANT")
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

    def construct(self, x):
        """Applies a convolutional layer on the input."""
        x = self.conv(x.transpose(0, 2, 1))
        x = ops.Squeeze(-1)(self.norm(ops.expand_dims(x, -1)))
        x = self.activation(x)
        x = self.maxpool(self.pad(x))
        x = x.swapaxes(1, 2)
        return x


class EncoderLayer(nn.Cell):
    """Encoder layer of the transformer"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(keep_prob=1-dropout)
        self.activation = ops.ReLU() if activation == "relu" else ops.GeLU()

    def construct(self, x, attn_mask=None):
        """Applies a transformer encoder layer on the input."""
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.swapaxes(-1, 1))))
        y = self.dropout(self.conv2(y).swapaxes(-1, 1))

        return self.norm2(x+y)


class Encoder(nn.Cell):
    """Encoder of the transformer"""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.CellList(attn_layers)
        self.conv_layers = nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def construct(self, x, attn_mask=None):
        """Forward function of the encoder"""
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(nn.Cell):
    """decoder layer"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        """Decoder layer of the transformer"""
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.norm3 = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(keep_prob=1-dropout)
        self.activation = ops.ReLU() if activation == "relu" else ops.GeLU()

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        """Applies a transformer decoder layer on the input."""
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        ))

        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        ))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(0, 2, 1))))
        y = self.dropout(self.conv2(y).transpose(0, 2, 1))

        return self.norm3(x+y)


class Decoder(nn.Cell):
    """Transformer decoder module"""
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
