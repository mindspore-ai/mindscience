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
models
"""
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, COOTensor

class GraphConv(nn.Cell):
    """
    Graph convolution operation
    Args:
        in_channels: input channel number
        units: hidden unit number
        Adj: adjacency matrix (after regularization), sparse matrix, shape: (N,N)
    """
    def __init__(self, in_channels, units, adj):
        """init"""
        super().__init__()
        self.adj = adj
        self.n = adj.shape[0]
        self.fc = nn.Dense(in_channels, units)
        self.units = units

    def construct(self, x):
        """construct"""
        x = self.fc(x)
        x = x.reshape([-1, self.n, self.units])
        x = ops.stack([ops.SparseTensorDenseMatmul()(self.adj.indices,
                                                     self.adj.values,
                                                     self.adj.shape, x[i]) for i in range(x.shape[0])],
                      axis=0)
        x = x.reshape([-1, self.units])
        return x

class GraphProlongation(nn.Cell):
    """
    Graph prolongation operation
    Args:
        P: Prolongation matrix, sparse matrix, shape: (N,N/2)
    """
    def __init__(self, p):
        """init"""
        super().__init__()
        self.p = p
        self.n = p.shape[1]

    def construct(self, x):
        """construct"""
        channels = x.shape[-1]
        x = x.reshape([-1, self.n, channels])
        x = ops.stack([ops.SparseTensorDenseMatmul()(self.p.indices,
                                                     self.p.values,
                                                     self.p.shape, x[i]) for i in range(x.shape[0])],
                      axis=0)
        x = x.reshape([-1, channels])
        return x

class GraphRestriction(nn.Cell):
    """
    Graph restriction operation
    Args:
        R: restriction matrix, sparse matrix, shape: (N/2,N)
    """
    def __init__(self, r):
        """init"""
        super().__init__()
        self.r = r
        self.n = r.shape[1]

    def construct(self, x):
        """construct"""
        channels = x.shape[-1]
        x = x.reshape([-1, self.n, channels])
        x = ops.stack([ops.SparseTensorDenseMatmul()(self.r.indices,
                                                     self.r.values,
                                                     self.r.shape, x[i]) for i in range(x.shape[0])],
                      axis=0)
        x = x.reshape([-1, channels])
        return x

class MultiScaleGNN(nn.Cell):
    """
    Multi-scale convolutional neural network
    Args:
        in_channels: input channel number
        out_channels: output channel number
        activation: activation function
        a0: Laplacian matrix, sparse matrix, shape: (N,N)
    """
    def __init__(self, in_channels, out_channels, activation, a0):
        """init"""
        super().__init__()
        ah, a2h, a4h, a8h, r2h, r4h, r8h, p2h, p4h, p8h = self.hierarchy_matrices(a0)

        self.conv_level1 = nn.SequentialCell(GraphConv(in_channels, 32, ah), activation,
                                             GraphConv(32, 32, ah), activation,
                                             GraphConv(32, 32, ah), activation)
        self.conv_level2 = nn.SequentialCell(GraphRestriction(r2h),
                                             GraphConv(32, 64, a2h), activation,
                                             GraphConv(64, 64, a2h), activation)
        self.conv_level3 = nn.SequentialCell(GraphRestriction(r4h),
                                             GraphConv(64, 128, a4h), activation,
                                             GraphConv(128, 128, a4h), activation)
        self.conv_level4 = nn.SequentialCell(GraphRestriction(r8h),
                                             GraphConv(128, 256, a8h), activation,
                                             GraphConv(256, 256, a8h), activation)
        self.deconv_level4 = nn.SequentialCell(GraphConv(256, 256, a8h), activation,
                                               GraphConv(256, 128, a8h), activation,
                                               GraphProlongation(p8h))
        self.deconv_level3 = nn.SequentialCell(GraphConv(128+128, 128, a4h), activation,
                                               GraphConv(128, 64, a4h), activation,
                                               GraphProlongation(p4h))
        self.deconv_level2 = nn.SequentialCell(GraphConv(64+64, 64, a2h), activation,
                                               GraphConv(64, 32, a2h), activation,
                                               GraphProlongation(p2h))
        self.deconv_level1 = nn.SequentialCell(GraphConv(32+32, 32, ah), activation,
                                               GraphConv(32, 32, ah), activation,
                                               GraphConv(32, out_channels, ah))

    def hierarchy_matrices(self, a0):
        """hierarchy_matrices"""
        ah = sparse.coo_array((np.ones_like(a0.data), (a0.row, a0.col)), shape=a0.shape)
        dh = sparse.coo_array((1./ah.sum(axis=0)**0.5, (np.arange(ah.shape[0]),
                                                        np.arange(ah.shape[1]))), shape=ah.shape)
        ah = (dh @ ah @ dh).tocoo()
        r2h = self.restrict_matrix(ah.shape)
        p2h = r2h.T
        a2h = (r2h @ ah @ p2h).tocoo()
        a2h = sparse.coo_array((np.ones_like(a2h.data), (a2h.row, a2h.col)), shape=a2h.shape)
        d2h = sparse.coo_array((1./a2h.sum(axis=0)**0.5, (np.arange(a2h.shape[0]),
                                                          np.arange(a2h.shape[1]))), shape=a2h.shape)
        a2h = (d2h @ a2h @ d2h).tocoo()
        r4h = self.restrict_matrix(a2h.shape)
        p4h = r4h.T
        a4h = (r4h @ a2h @ p4h).tocoo()
        a4h = sparse.coo_array((np.ones_like(a4h.data), (a4h.row, a4h.col)), shape=a4h.shape)
        d4h = sparse.coo_array((1./a4h.sum(axis=0)**0.5, (np.arange(a4h.shape[0]),
                                                          np.arange(a4h.shape[1]))), shape=a4h.shape)
        a4h = (d4h @ a4h @ d4h).tocoo()
        r8h = self.restrict_matrix(a4h.shape)
        p8h = r8h.T
        a8h = (r8h @ a4h @ p8h).tocoo()
        a8h = sparse.coo_array((np.ones_like(a8h.data), (a8h.row, a8h.col)), shape=a8h.shape)
        d8h = sparse.coo_array((1./a8h.sum(axis=0)**0.5, (np.arange(a8h.shape[0]),
                                                          np.arange(a8h.shape[1]))), shape=a8h.shape)
        a8h = (d8h @ a8h @ d8h).tocoo()
        ah = COOTensor(Tensor(np.stack((ah.row, ah.col), axis=1)), Tensor(ah.data, dtype=mindspore.float32), ah.shape)
        a2h = COOTensor(Tensor(np.stack((a2h.row, a2h.col), axis=1)),
                        Tensor(a2h.data, dtype=mindspore.float32), a2h.shape)
        a4h = COOTensor(Tensor(np.stack((a4h.row, a4h.col), axis=1)),
                        Tensor(a4h.data, dtype=mindspore.float32), a4h.shape)
        a8h = COOTensor(Tensor(np.stack((a8h.row, a8h.col), axis=1)),
                        Tensor(a8h.data, dtype=mindspore.float32), a8h.shape)
        r2h = COOTensor(Tensor(np.stack((r2h.row, r2h.col), axis=1)),
                        Tensor(r2h.data, dtype=mindspore.float32), r2h.shape)
        r4h = COOTensor(Tensor(np.stack((r4h.row, r4h.col), axis=1)),
                        Tensor(r4h.data, dtype=mindspore.float32), r4h.shape)
        r8h = COOTensor(Tensor(np.stack((r8h.row, r8h.col), axis=1)),
                        Tensor(r8h.data, dtype=mindspore.float32), r8h.shape)
        p2h = COOTensor(Tensor(np.stack((p2h.row, p2h.col), axis=1)),
                        Tensor(p2h.data, dtype=mindspore.float32), p2h.shape)
        p4h = COOTensor(Tensor(np.stack((p4h.row, p4h.col), axis=1)),
                        Tensor(p4h.data, dtype=mindspore.float32), p4h.shape)
        p8h = COOTensor(Tensor(np.stack((p8h.row, p8h.col), axis=1)),
                        Tensor(p8h.data, dtype=mindspore.float32), p8h.shape)

        return ah, a2h, a4h, a8h, r2h, r4h, r8h, p2h, p4h, p8h

    def restrict_matrix(self, shape):
        """restrict_matrix"""
        nx = int(shape[0]**0.5)
        ny = nx
        nx_coarse, ny_coarse = nx // 2, ny // 2
        r = lil_matrix((nx_coarse*ny_coarse, nx*ny), dtype='float64')
        for i in range(nx_coarse):
            for j in range(ny_coarse):
                k_coarse = i*ny_coarse + j
                k_fine = 2*i*ny + 2*j
                r[k_coarse, k_fine] = 0.25
                r[k_coarse, k_fine+1] = 0.25
                r[k_coarse, k_fine+ny] = 0.25
                r[k_coarse, k_fine+ny+1] = 0.25
        return r.tocoo()

    def construct(self, inputs):
        """construct"""
        conv1 = self.conv_level1(inputs)
        conv2 = self.conv_level2(conv1)
        conv3 = self.conv_level3(conv2)
        conv4 = self.conv_level4(conv3)

        deconv4 = self.deconv_level4(conv4)
        deconv3 = self.deconv_level3(ops.cat([deconv4, conv3], axis=-1))
        deconv2 = self.deconv_level2(ops.cat([deconv3, conv2], axis=-1))
        outputs = self.deconv_level1(ops.cat([deconv2, conv1], axis=-1))

        return outputs

class GraphConvStructure(nn.Cell):
    """
    Graph convolution operation for structure grid, do not use sparse matrix multiplication
    Args:
        in_channels: input channel number
        units: hidden unit number
    """
    def __init__(self, in_channels, units):
        """init"""
        super().__init__()
        self.fc = nn.Conv2d(in_channels, units, kernel_size=1, stride=1)
        self.pad = nn.ZeroPad2d(1)

    def construct(self, x):
        """construct"""
        x = self.fc(x)
        x = self.pad(x)
        x = (x[:, :, 0:-2, 1:-1] + x[:, :, 1:-1, 1:-1] + x[:, :, 2:, 1:-1] +
             x[:, :, 1:-1, 0:-2] + x[:, :, 1:-1, 2:]) / 5.
        return x

class MultiScaleGNNStructure(nn.Cell):
    """
    Multi-scale convolutional neural network
    Args:
        in_channels: input channel number
        out_channels: output channel number
        activation: activation function
    """
    def __init__(self, in_channels, out_channels, activation):
        """init"""
        super().__init__()

        self.conv_level1 = nn.SequentialCell(GraphConvStructure(in_channels, 32), activation,
                                             GraphConvStructure(32, 32), activation,
                                             GraphConvStructure(32, 32), activation)
        self.conv_level2 = nn.SequentialCell(nn.Conv2d(32, 32, 2, 2),
                                             GraphConvStructure(32, 64), activation,
                                             GraphConvStructure(64, 64), activation)
        self.conv_level3 = nn.SequentialCell(nn.Conv2d(64, 64, 2, 2),
                                             GraphConvStructure(64, 128), activation,
                                             GraphConvStructure(128, 128), activation)
        self.conv_level4 = nn.SequentialCell(nn.Conv2d(128, 128, 2, 2),
                                             GraphConvStructure(128, 256), activation,
                                             GraphConvStructure(256, 256), activation)
        self.deconv_level4 = nn.SequentialCell(GraphConvStructure(256, 256), activation,
                                               GraphConvStructure(256, 128), activation,
                                               nn.Conv2dTranspose(128, 128, 2, 2))
        self.deconv_level3 = nn.SequentialCell(GraphConvStructure(128+128, 128), activation,
                                               GraphConvStructure(128, 64), activation,
                                               nn.Conv2dTranspose(64, 64, 2, 2))
        self.deconv_level2 = nn.SequentialCell(GraphConvStructure(64+64, 64), activation,
                                               GraphConvStructure(64, 32), activation,
                                               nn.Conv2dTranspose(32, 32, 2, 2))
        self.deconv_level1 = nn.SequentialCell(GraphConvStructure(32+32, 32), activation,
                                               GraphConvStructure(32, 32), activation,
                                               GraphConvStructure(32, out_channels))

    def construct(self, inputs):
        """construct"""
        conv1 = self.conv_level1(inputs)
        conv2 = self.conv_level2(conv1)
        conv3 = self.conv_level3(conv2)
        conv4 = self.conv_level4(conv3)

        deconv4 = self.deconv_level4(conv4)
        deconv3 = self.deconv_level3(ops.cat([deconv4, conv3], axis=1))
        deconv2 = self.deconv_level2(ops.cat([deconv3, conv2], axis=1))
        outputs = self.deconv_level1(ops.cat([deconv2, conv1], axis=1))

        return outputs
