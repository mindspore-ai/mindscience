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
"""precipitation model"""
from mindspore import nn, ops


class Pad2d(nn.Cell):
    """
    Recurrent padding

    Args:
        pad_width (int): the padding size

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        - **out** (Tensor) - The tensor after padding.
    """

    def __init__(self, pad_width):
        super(Pad2d, self).__init__()
        self.pad_width = pad_width
        self.pad_tb = nn.Pad(paddings=((0, 0), (0, 0), (pad_width, pad_width), (0, 0)), mode="CONSTANT")

    def construct(self, x):
        out = ops.concat([x[..., -self.pad_width:].copy(), x, x[..., :self.pad_width].copy()], axis=-1)
        out = self.pad_tb(out)
        return out


class PrecipNet(nn.Cell):
    r"""
    The PrecipNet stacks two backbones. one is the foundation backbone, and the other learns representations of
    the precipitation forecasting.

    Args:
        backbone_no_grad (Cell): backbone with frozen parameters.
        backbone (Cell): backbone with trainable parameters.
        data_params (dict): the configurations of dataset.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, height\_size * width\_size, feature\_size)` .

    Outputs:
        - **out** (Tensor) - Tensor of shape :math:`(batch\_size, 1, height\_size, width\_size)` .
        - **out_recon** (Tensor) - Tensor of shape :math:`(height\_size * width\_size, feature\_size)` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, Tensor
        >>> from precip import PrecipNet
        >>> from mindearth.cell.graphcast.graphcastnet import GraphCastNet
        >>>
        >>> data_params = {'feature_dim': 69, 'w_size': 720, 'h_size': 360}
        >>> mesh_node_num = 2562
        >>> grid_node_num = 259200
        >>> mesh_edge_num = 20460
        >>> g2m_edge_num = 50184
        >>> m2g_edge_num = 98304
        >>> vm_in_channels = 3
        >>> em_in_channels = 4
        >>> eg2m_in_channels = 4
        >>> em2g_in_channels = 4
        >>> feature_num = 69
        >>> g2m_src_idx = Tensor(np.random.randint(0, grid_node_num, size=[g2m_edge_num]), ms.int32)
        >>> g2m_dst_idx = Tensor(np.random.randint(0, mesh_node_num, size=[g2m_edge_num]), ms.int32)
        >>> m2m_src_idx = Tensor(np.random.randint(0, mesh_node_num, size=[mesh_edge_num]), ms.int32)
        >>> m2m_dst_idx = Tensor(np.random.randint(0, mesh_node_num, size=[mesh_edge_num]), ms.int32)
        >>> m2g_src_idx = Tensor(np.random.randint(0, mesh_node_num, size=[m2g_edge_num]), ms.int32)
        >>> m2g_dst_idx = Tensor(np.random.randint(0, grid_node_num, size=[m2g_edge_num]), ms.int32)
        >>> mesh_node_feats = Tensor(np.random.rand(mesh_node_num, vm_in_channels).astype(np.float32), ms.float32)
        >>> mesh_edge_feats = Tensor(np.random.rand(mesh_edge_num, em_in_channels).astype(np.float32), ms.float32)
        >>> g2m_edge_feats = Tensor(np.random.rand(g2m_edge_num, eg2m_in_channels).astype(np.float32), ms.float32)
        >>> m2g_edge_feats = Tensor(np.random.rand(m2g_edge_num, em2g_in_channels).astype(np.float32), ms.float32)
        >>> per_variable_level_mean = Tensor(np.random.rand(feature_num,).astype(np.float32), ms.float32)
        >>> per_variable_level_std = Tensor(np.random.rand(feature_num,).astype(np.float32), ms.float32)
        >>> backbone_no_grad = GraphCastNet(vg_in_channels=feature_num,
        >>>                                 vg_out_channels=feature_num,
        >>>                                 vm_in_channels=vm_in_channels,
        >>>                                 em_in_channels=em_in_channels,
        >>>                                 eg2m_in_channels=eg2m_in_channels,
        >>>                                 em2g_in_channels=em2g_in_channels,
        >>>                                 latent_dims=512,
        >>>                                 processing_steps=4,
        >>>                                 g2m_src_idx=g2m_src_idx,
        >>>                                 g2m_dst_idx=g2m_dst_idx,
        >>>                                 m2m_src_idx=m2m_src_idx,
        >>>                                 m2m_dst_idx=m2m_dst_idx,
        >>>                                 m2g_src_idx=m2g_src_idx,
        >>>                                 m2g_dst_idx=m2g_dst_idx,
        >>>                                 mesh_node_feats=mesh_node_feats,
        >>>                                 mesh_edge_feats=mesh_edge_feats,
        >>>                                 g2m_edge_feats=g2m_edge_feats,
        >>>                                 m2g_edge_feats=m2g_edge_feats,
        >>>                                 per_variable_level_mean=per_variable_level_mean,
        >>>                                 per_variable_level_std=per_variable_level_std)
        >>> backbone = GraphCastNet(vg_in_channels=feature_num,
        >>>                              vg_out_channels=feature_num,
        >>>                              vm_in_channels=vm_in_channels,
        >>>                              em_in_channels=em_in_channels,
        >>>                              eg2m_in_channels=eg2m_in_channels,
        >>>                              em2g_in_channels=em2g_in_channels,
        >>>                              latent_dims=512,
        >>>                              processing_steps=4,
        >>>                              g2m_src_idx=g2m_src_idx,
        >>>                              g2m_dst_idx=g2m_dst_idx,
        >>>                              m2m_src_idx=m2m_src_idx,
        >>>                              m2m_dst_idx=m2m_dst_idx,
        >>>                              m2g_src_idx=m2g_src_idx,
        >>>                              m2g_dst_idx=m2g_dst_idx,
        >>>                              mesh_node_feats=mesh_node_feats,
        >>>                              mesh_edge_feats=mesh_edge_feats,
        >>>                              g2m_edge_feats=g2m_edge_feats,
        >>>                              m2g_edge_feats=m2g_edge_feats,
        >>>                              per_variable_level_mean=per_variable_level_mean,
        >>>                              per_variable_level_std=per_variable_level_std)
        >>> model = PrecipNet(backbone_no_grad, backbone, data_params)
        >>> inputs = Tensor(np.random.rand(grid_node_num, feature_num).astype(np.float32), ms.float32)
        >>> out, out_recon = model(inputs)
        >>> print(out.shape, out_recon.shape)
        (1, 1, 360, 720) (259200, 69)
    """

    def __init__(self, backbone_no_grad, backbone, data_params):
        super(PrecipNet, self).__init__()
        self.feature_dims = data_params.get("feature_dims", 69)
        self.h_size, self.w_size = data_params.get("h_size", 360), data_params.get("w_size", 720)
        self.backbone_no_grad = backbone_no_grad
        for w in self.backbone_no_grad.trainable_params():
            w.requires_grad = False
        self.backbone = backbone
        self.pad = Pad2d(1)
        self.conv = nn.Conv2d(self.feature_dims, 1, 3, has_bias=True, pad_mode='valid', weight_init='normal')
        self.act = nn.ReLU()

    def construct(self, x):
        out_recon = self.backbone_no_grad(x)
        x = self.backbone(out_recon)
        out = x.expand_dims(0).transpose(0, 2, 1).reshape(1, self.feature_dims, self.h_size, self.w_size)
        out = self.pad(out)
        out = self.conv(out)
        out = self.act(out)
        return out, out_recon
