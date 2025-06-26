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
"GTeam model"
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class MLP(nn.Cell):
    """
    A Multi-Layer Perceptron (MLP) class using MindSpore's nn.Cell.
    Parameters:
        input_shape: Tuple representing the shape of the input data.
        dims: Tuple containing the dimensions of each layer. Default is (100, 50).
        final_activation: The activation function for the final layer. Default is nn.ReLU.
    """

    def __init__(self, input_shape, dims=(100, 50), final_activation=nn.ReLU(), is_mlp=False):
        super().__init__()
        layers = []
        in_dim = input_shape[0]
        if is_mlp:
            for dim in dims[:-1]:
                layers.append(nn.Dense(in_dim, dim))
                layers.append(nn.LayerNorm((dim,)))
                layers.append(nn.ReLU())
                in_dim = dim
            layers.append(nn.Dense(in_dim, dims[-1]))

            if final_activation:
                layers.append(final_activation)
            self.model = nn.SequentialCell(*layers)
        else:
            for dim in dims[:-1]:
                layers.append(nn.Dense(in_dim, dim))
                layers.append(nn.ReLU())
                in_dim = dim
            layers.append(nn.Dense(in_dim, dims[-1]))

            if final_activation:
                layers.append(final_activation)
            self.model = nn.SequentialCell(*layers)

    def construct(self, x):
        """
        Forward pass through the network.
        Parameters:
          x: Input data to the MLP.
        Returns:
          Output after passing through the MLP.
        """
        return self.model(x)


class NormalizedScaleEmbedding(nn.Cell):
    """
    A neural network module that normalizes input data, extracts features using a series of
    convolutional and pooling layers, and processes the features through a multi-layer perceptron (MLP).
    """

    def __init__(self, downsample=5, mlp_dims=(500, 300, 200, 150), eps=1e-8, use_mlp=False):
        """
        Initialize the module with given parameters.
        Parameters:
            :downsample: Downsampling factor for the first convolutional layer.
            :mlp_dims: Dimensions for the MLP layers.
            :eps: A small value for numerical stability.
        """
        super().__init__()
        self.downsample = downsample
        self.mlp_dims = mlp_dims
        self.eps = eps

        self.conv2d_1 = nn.Conv2d(
            1,
            8,
            kernel_size=(downsample, 1),
            stride=(downsample, 1),
            has_bias=True,
            pad_mode="pad",
        )
        self.conv2d_2 = nn.Conv2d(
            8, 32, kernel_size=(16, 3), stride=(1, 1), has_bias=True, pad_mode="pad"
        )

        self.conv1d_1 = nn.Conv1d(32, 64, kernel_size=16, has_bias=True, pad_mode="pad")
        self.maxpool_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(
            64, 128, kernel_size=16, has_bias=True, pad_mode="pad"
        )
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(128, 32, kernel_size=8, has_bias=True, pad_mode="pad")
        self.maxpool_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_4 = nn.Conv1d(32, 32, kernel_size=8, has_bias=True, pad_mode="pad")
        self.conv1d_5 = nn.Conv1d(32, 16, kernel_size=4, has_bias=True, pad_mode="pad")

        self.flatten = nn.Flatten()
        self.mlp = MLP((865,), dims=self.mlp_dims, is_mlp=use_mlp)
        self.leaky_relu = nn.LeakyReLU(alpha=0.01)
        self._initialize_weights()

    def _initialize_weights(self):
        self.conv2d_1.bias.set_data(ms.numpy.zeros_like(self.conv2d_1.bias))
        self.conv2d_2.bias.set_data(ms.numpy.zeros_like(self.conv2d_2.bias))

        # For Conv1d layers
        self.conv1d_1.bias.set_data(ms.numpy.zeros_like(self.conv1d_1.bias))
        self.conv1d_2.bias.set_data(ms.numpy.zeros_like(self.conv1d_2.bias))
        self.conv1d_3.bias.set_data(ms.numpy.zeros_like(self.conv1d_3.bias))
        self.conv1d_4.bias.set_data(ms.numpy.zeros_like(self.conv1d_4.bias))
        self.conv1d_5.bias.set_data(ms.numpy.zeros_like(self.conv1d_5.bias))

    def construct(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Processed output tensor.
        """
        original_input = x
        x = (
            x
            / (
                ops.max(
                    ops.max(ops.abs(x), axis=1, keepdims=True)[0], axis=2, keepdims=True
                )[0]
                + self.eps
            )
            + self.eps
        )
        x = ops.unsqueeze(x, dim=1)

        scale = (
            ops.log(
                ops.max(ops.max(ops.abs(original_input), axis=1)[0], axis=1)[0]
                + self.eps
            )
            / 100
            + self.eps
        )
        scale = ops.unsqueeze(scale, dim=1)

        x = self.leaky_relu(self.conv2d_1(x))
        x = self.leaky_relu(self.conv2d_2(x))

        tmp_x = ops.Squeeze(axis=-1)
        x = tmp_x(x)
        x = self.leaky_relu(self.conv1d_1(x))
        x = self.maxpool_1(x)
        x = self.leaky_relu(self.conv1d_2(x))
        x = self.maxpool_2(x)
        x = self.leaky_relu(self.conv1d_3(x))
        x = self.maxpool_3(x)
        x = self.leaky_relu(self.conv1d_4(x))
        x = self.leaky_relu(self.conv1d_5(x))

        x = self.flatten(x)
        x = ops.cat((x, scale), axis=1)
        x = self.mlp(x)
        return x


class TransformerEncoder(nn.Cell):
    """
    TransformerEncoder class, used to implement the Transformer encoder.
    Parameters:
    d_model: Dimension of the input data.
    nhead: Number of heads in multi-head attention.
    num_layers: Number of layers in the encoder.
    batch_first: Whether to consider the first dimension of the input data as the batch dimension.
    activation: Type of activation function.
    dim_feedforward: Dimension of the hidden layer in the feedforward network.
    dropout: Proportion of dropout.
    Methods:
    __init__: Initialize the TransformerEncoder object.
    construct: Construct the TransformerEncoder network.
    """

    def __init__(
            self,
            d_model=500,
            nhead=10,
            num_layers=6,
            batch_first=True,
            activation="gelu",
            dim_feedforward=1000,
            dropout=0.0,
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=batch_first,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

    def construct(self, x, src_key_padding_mask=None):
        """Construct the TransformerEncoder network"""
        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)


class PositionEmbedding(nn.Cell):
    """
    PositionEmbedding class, used to implement position embeddings.
    Parameters:
    wavelengths: Range of wavelengths.
    emb_dim: Dimension of the embeddings.
    Methods:
    __init__: Initialize the PositionEmbedding object.
    construct: Construct the PositionEmbedding network.
    """

    def __init__(self, wavelengths, emb_dim):
        super().__init__()
        self.wavelengths = wavelengths
        self.emb_dim = emb_dim

        min_lat, max_lat = wavelengths[0]
        min_lon, max_lon = wavelengths[1]
        min_depth, max_depth = wavelengths[2]
        assert emb_dim % 10 == 0
        lat_dim = emb_dim // 5
        lon_dim = emb_dim // 5
        depth_dim = emb_dim // 10
        self.lat_coeff = (
            2
            * np.pi
            * 1.0
            / min_lat
            * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        )
        self.lon_coeff = (
            2
            * np.pi
            * 1.0
            / min_lon
            * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))
        )
        self.depth_coeff = (
            2
            * np.pi
            * 1.0
            / min_depth
            * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))
        )
        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3

        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9

        self.mask = np.zeros(emb_dim)
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = (
            2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        )
        self.mask = ms.tensor(self.mask.astype("int32"))

    def construct(self, x):
        """position embedding"""
        lat_base = x[:, :, 0:1] * ms.tensor(self.lat_coeff, dtype=ms.float32)
        lon_base = x[:, :, 1:2] * ms.tensor(self.lon_coeff, dtype=ms.float32)
        depth_base = x[:, :, 2:3] * ms.tensor(self.depth_coeff, dtype=ms.float32)

        output = ops.cat(
            [
                ops.sin(lat_base),
                ops.cos(lat_base),
                ops.sin(lon_base),
                ops.cos(lon_base),
                ops.sin(depth_base),
                ops.cos(depth_base),
            ],
            axis=-1,
        )
        output = ops.index_select(output, axis=-1, index=self.mask)

        return output


class AddEventToken(nn.Cell):
    """
    AddEventToken class, used to implement adding event tokens.

    Parameters:
    emb_dim: Dimension of the embeddings.
    init_range: Initialization range.

    Methods:
    __init__: Initialize the AddEventToken object.
    construct: Construct the AddEventToken network.
    """

    def __init__(self, emb_dim, init_range):
        super().__init__()
        self.emb_dim = emb_dim
        init_value = np.random.uniform(-init_range, init_range, (1, 1, emb_dim)).astype(
            np.float32
        )
        self.event_token = ms.Parameter(ms.Tensor(init_value), name="event_token")

    def construct(self, x):
        """add eventtoken"""
        event_token = self.event_token
        pad = ops.ones_like(x[:, :1, :]) * event_token
        x = ops.cat([pad, x], axis=1)

        return x

class SingleStationModel(nn.Cell):
    """
    A neural network model for processing seismic waveforms from a single station.
    This class implements a two-stage processing pipeline: waveform embedding followed by feature extraction.
    """
    def __init__(self, waveform_model_dims=(500, 500, 500),
                 output_mlp_dims=(150, 100, 50, 30, 10), downsample=5, use_mlp=False):
        """
        Initialize the SingleStationModel.

        Args:
            waveform_model_dims (tuple): Dimensions of the MLP in the waveform embedding module.
                Format: (input_dim, hidden_dim1, hidden_dim2, ...)
            output_mlp_dims (tuple): Dimensions of the final MLP for feature extraction.
                Format: (input_dim, hidden_dim1, hidden_dim2, ...)
            downsample (int): Factor by which to downsample the input waveform data.
        """
        super().__init__()

        self.waveform_model = NormalizedScaleEmbedding(downsample=downsample, mlp_dims=waveform_model_dims,
                                                       use_mlp=use_mlp)
        self.mlp_mag_single_station = MLP((self.waveform_model.mlp_dims[-1],), output_mlp_dims)

    def construct(self, x):
        """
       Forward pass of the SingleStationModel.

       Args:
           x (Tensor): Input waveform data with shape (batch_size, time_steps, features)

       Returns:
           Tensor: Extracted features with shape (batch_size, output_features)
       """
        emb = self.waveform_model(x)
        emb_mlp = self.mlp_mag_single_station(emb)

        return emb_mlp
def _init_pad_mask(waveforms, pga_targets):
    """
    _init_pad_mask function, used to initialize the padding mask.
    """
    station_pad_mask = abs(waveforms) < 1e-8
    station_pad_mask = ops.all(station_pad_mask, axis=2)
    station_pad_mask = ops.all(station_pad_mask, axis=2)

    event_token_mask = ops.zeros((station_pad_mask.shape[0], 1), dtype=ms.dtype.bool_)
    pad_mask = ops.cat([event_token_mask, station_pad_mask], axis=1)

    target_pad_mask = ms.numpy.ones_like(pga_targets, dtype=ms.dtype.bool_)
    target_pad_mask = ops.all(target_pad_mask, 2)

    pad_mask = ops.cat((pad_mask, target_pad_mask), axis=1)

    return pad_mask


class WaveformFullmodel(nn.Cell):
    """
    Waveform full model class, used for processing and predicting waveform data."
    """

    def __init__(
            self,
            waveform_model_dims=(500, 500, 500),
            output_mlp_dims=(150, 100, 50, 30, 10),
            output_location_dims=(150, 100, 50, 50, 50),
            wavelength=((0.01, 10), (0.01, 10), (0.01, 10)),
            n_heads=10,
            hidden_dim=1000,
            transformer_layers=6,
            hidden_dropout=0.0,
            n_pga_targets=0,
            downsample=5,
            use_mlp=False
    ):
        super().__init__()
        self.waveform_model = NormalizedScaleEmbedding(
            downsample=downsample, mlp_dims=waveform_model_dims, use_mlp=use_mlp
        )
        self.transformer = TransformerEncoder(
            d_model=waveform_model_dims[-1],
            nhead=n_heads,
            num_layers=transformer_layers,
            dim_feedforward=hidden_dim,
            dropout=hidden_dropout,
        )

        self.mlp_mag = MLP((waveform_model_dims[-1],), output_mlp_dims, is_mlp=use_mlp)
        self.mlp_loc = MLP(
            (waveform_model_dims[-1],), output_location_dims, final_activation=None, is_mlp=use_mlp
        )
        self.mlp_pga = MLP(
            (waveform_model_dims[-1],), output_mlp_dims, final_activation=None, is_mlp=use_mlp
        )

        self.position_embedding = PositionEmbedding(
            wavelengths=wavelength, emb_dim=waveform_model_dims[-1]
        )
        self.addeventtoken = AddEventToken(emb_dim=500, init_range=0.02)
        self.n_pga_targets = n_pga_targets

    def cal_waveforms_emb_normalized(self, waveforms_emb):
        """Normalize the waveform embeddings"""
        mean_vals = waveforms_emb.mean(axis=2, keep_dims=True)
        std_vals = waveforms_emb.std(axis=2, keepdims=True)
        waveforms_emb_normalized = (waveforms_emb - mean_vals) / (std_vals + 1e-8)
        return waveforms_emb_normalized

    def construct(self, waveforms, metadata, pga_targets):
        """
        Construct method to process the input waveforms, metadata, and PGA targets.
        """
        batch_size, num_stations, seq_length, num_channels = waveforms.shape
        waveforms_reshape = waveforms.reshape(-1, seq_length, num_channels)

        waveforms_emb = self.waveform_model(waveforms_reshape)
        waveforms_emb = waveforms_emb.reshape(batch_size, num_stations, -1)
        waveforms_emb_normalized = self.cal_waveforms_emb_normalized(waveforms_emb)
        coords_emb = self.position_embedding(metadata)
        pga_target_emb = self.position_embedding(pga_targets)
        pad_mask = _init_pad_mask(waveforms, pga_targets)

        emb_pos = waveforms_emb_normalized + coords_emb
        emb_pos = self.addeventtoken(emb_pos)
        emb_pos_pga = ops.cat((emb_pos, pga_target_emb), axis=1)
        emb_pos_pga_trans = self.transformer(emb_pos_pga, pad_mask)
        emb_pga = emb_pos_pga_trans[:, -self.n_pga_targets :, :]
        emb_mag_loc = emb_pos_pga_trans[:, 0, :]

        mag = self.mlp_mag(emb_mag_loc)
        loc = self.mlp_loc(emb_mag_loc)

        pga_all = self.mlp_pga(emb_pga)
        outputs = [mag, loc, pga_all]

        return outputs
