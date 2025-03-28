# Copyright 2025 Huawei Technologies Co., Ltd
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
"""transform"""
import scipy.spatial
from mindspore import ops, Tensor
import mindspore as ms

from .utils import NodeType
from .utils import to_undirected


class BaseTransform:
    def __call__(self, data):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Compose(BaseTransform):
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        args = [f'  {transform}' for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))


class Cartesian(BaseTransform):
    """Saves the relative Cartesian coordinates of linked nodes in its edge
    attributes."""
    def __init__(self, norm: bool = False):
        self.norm = norm

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[col] - pos[row]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart
        data.rel_pos = cart

        if self.norm and cart.numel() > 0:
            max_value = cart.abs().max()
            cart = cart / (2 * max_value) + 0.5

        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = ops.cat([pseudo, cart.type_as(pseudo)], axis=-1)
        else:
            data.edge_attr = cart

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(norm={self.norm}'


class Distance(BaseTransform):
    """Saves the Euclidean distance of linked nodes in its edge attributes."""
    def __init__(self, norm: bool = False):
        self.norm = norm

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        dist = ms.numpy.norm(pos[col] - pos[row], axis=-1).view(-1, 1)
        data.distance = dist

        if self.norm and dist.numel() > 0:
            dist = dist / dist.max()

        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = ops.cat([pseudo, dist.type_as(pseudo)], axis=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(norm={self.norm}'


class Delaunay(BaseTransform):
    """Delaunay transform"""
    def __call__(self, data):
        if data.pos.shape[0] < 2:
            data.edge_index = ops.Tensor([], dtype=ms.int64,
                                         device=data.pos.device).view(2, 0)
        if data.pos.shape[0] == 2:
            data.edge_index = ops.Tensor([[0, 1], [1, 0]], dtype=ms.int64,
                                         device=data.pos.device)
        elif data.pos.shape[0] == 3:
            data.face = ops.Tensor([[0], [1], [2]], dtype=ms.int64,
                                   device=data.pos.device)
        if data.pos.shape[0] > 3:
            pos = data.pos.numpy()
            tri = scipy.spatial.Delaunay(pos, qhull_options='QJ')
            face = ms.from_numpy(tri.simplices)

            data.face = face.t().contiguous().to(ms.int64)

        return data


class FaceToEdge(BaseTransform):
    """FaceToEdge transform"""
    def __init__(self, remove_faces: bool = True):
        self.remove_faces = remove_faces

    def __call__(self, data):
        if hasattr(data, 'face'):
            face = data.face
            edge_index = ops.cat([face[:2], face[1:], face[::2]], axis=1)
            edge_index = to_undirected(edge_index, num_nodes=data.pos.shape[0])

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data


class Dirichlet(BaseTransform):
    """Dirichlet transform"""
    def __init__(self):
        self.index = None

    def set_index(self, index):
        self.index = Tensor(index, dtype=ms.int64)

    def __call__(self, data):
        data.dirichlet_index = self.index
        return data


class DirichletInlet(BaseTransform):
    """DirichletInlet transform"""
    def __init__(self):
        self.index = None

    def set_index(self, index):
        self.index = Tensor(index, dtype=ms.int64)

    def __call__(self, data):
        data.inlet_index = self.index
        return data


class MaskFace(BaseTransform):
    """MaskFace transform"""
    def __init__(self):
        self.cylinder_index = None
        self.new_face_index = None

    def is_none(self):
        return self.new_face_index is None

    def set_cylinder_index(self, cylinder_index):
        self.cylinder_index = Tensor(cylinder_index, dtype=ms.int64)

    def __call__(self, data):
        if self.is_none():
            self.new_face_index = self.cal_mask_face(data)

        data.face = data.face[:, self.new_face_index]
        return data

    def cal_mask_face(self, graph):
        on_circle_index = self.cylinder_index
        new_face_index = []
        for i in range(graph.face.shape[1]):
            if ms.numpy.isin(graph.face[:, i], on_circle_index).all():
                continue
            else:
                new_face_index.append(i)
        return Tensor(new_face_index)


class NodeTypeInfo(BaseTransform):
    """NodeTypeInfo transform"""
    def __init__(self):
        self.type_dict = None
        self.node_type = None

    def is_none(self):
        return self.node_type is None

    def set_type_dict(self, type_dict):
        self.type_dict = type_dict

    def __call__(self, data):
        if self.is_none():
            self.node_type = self.cal_node_type(data)

        data.node_type = self.node_type
        return data

    def cal_node_type(self, data):
        """compute node type"""
        node_num = data.pos.shape[0]
        node_type = ops.ones(node_num, dtype=ms.int64) * NodeType.NORMAL
        if hasattr(data, 'dirichlet_index'):
            node_type[data.dirichlet_index] = NodeType.OBSTACLE
        if hasattr(data, 'inlet_index'):
            node_type[data.inlet_index] = NodeType.INLET

        outlet_index = self.type_dict['outlet'][:]
        outlet_index = Tensor(outlet_index, dtype=ms.int64)
        node_type[outlet_index] = NodeType.OUTLET
        return node_type
