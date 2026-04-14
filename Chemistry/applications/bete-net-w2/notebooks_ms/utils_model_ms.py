# MindSpore & MindChemistry.e3 版本的模型定义

# 设置导入路径
import setup_paths
import  math

import mindspore as ms
import mindspore.nn as nn
from mindchemistry.e3 import o3
from mindchemistry.e3.nn.one_hot import soft_one_hot_linspace
from mindchemistry.e3.nn import Gate
from conv_e3nn import Convolution
from sharker.utils import scatter
import numpy as np

# 默认数据类型
default_dtype = ms.float32

def smooth_cutoff(x):
    """
    平滑截断函数 - 与原版一致
    """
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y

def tp_path_exists(irreps1, irreps2, irreps_out):
    """
    检查张量积路径是否存在
    """
    irreps1 = o3.Irreps(irreps1).simplify()
    irreps2 = o3.Irreps(irreps2).simplify()
    irreps_out = o3.Irrep(irreps_out)

    for _, ir1 in irreps1:
        for _, ir2 in irreps2:
            if irreps_out in ir1 * ir2:
                return True
    return False

def radius_graph(pos, r, batch=None):
    """
    构建半径图
    """
    n = pos.shape[0]
    
    # 计算距离矩阵
    pos_expanded = pos.unsqueeze(1)  # [n, 1, 3]
    pos_expanded_t = pos.unsqueeze(0)  # [1, n, 3]
    distances = ms.ops.norm(pos_expanded - pos_expanded_t, dim=2)  # [n, n]
    
    # 创建掩码
    mask = (distances <= r).astype(ms.int64) & (distances > 0).astype(ms.int64)  # 排除自连接
    
    # 获取边索引
    edge_indices = ms.ops.nonzero(mask)
    if edge_indices.shape[0] == 0:
        return ms.ops.zeros((2, 0), dtype=ms.int64)
    
    return edge_indices.T

class CustomCompose(nn.Cell):
    """
    自定义组合层 - 与原版一致
    """
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = first.irreps_node_input
        self.irreps_out = second.irreps_out
        
    def construct(self, *input):
        x = self.first(*input)
        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/test_network_conv_0.npy"))
        # print((torch_res - x).abs().max())
        self.first_out = x  # 保存中间结果用于调试
        x = self.second(x)
        self.second_out = x  # 保存中间结果用于调试
        return x

class Dropout(nn.Cell):
    """
    Dropout层 - 适配MindSpore
    """
    def __init__(self, irreps, p):
        super().__init__()
        self.irreps = irreps
        # MindSpore的Dropout使用p参数
        if p <= 0 or p >= 1:
            # 如果p无效，创建一个恒等映射
            self.dropout = None
        else:
            self.dropout = nn.Dropout(p=p)
        
    def construct(self, x):
        if self.dropout is not None:
            return self.dropout(x)
        else:
            return x

class Network(nn.Cell):
    """
    等变神经网络 - 使用正确的MindChemistry.e3 API
    """
    
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        radial_layers=1,
        radial_neurons=100,
        num_neighbors=1.0,
        num_nodes=1.0,
        reduce_output=True,
        p=0.2,
        dropout=False
    ):
        super().__init__()
        self.dropout = dropout
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        # 构建隐藏层的不可约表示 - 与原版一致
        self.irreps_hidden = o3.Irreps(
            [(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = (
            o3.Irreps(irreps_node_attr)
            if irreps_node_attr is not None
            else o3.Irreps("0e")
        )
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        # 激活函数定义 - 与原版一致
        act = {1: ms.ops.silu, -1: ms.ops.tanh}
        act_gates = {1: ms.ops.sigmoid, -1: ms.ops.tanh}

        self.layers = nn.CellList()
        self.drop_outs = nn.CellList()

        # 构建层 - 与原版逻辑一致
        for _ in range(layers):
            # 标量不可约表示
            irreps_scalars = o3.Irreps([
                (mul, ir)
                for mul, ir in self.irreps_hidden
                if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
            ])
            
            # 门控不可约表示
            irreps_gated = o3.Irreps([
                (mul, ir)
                for mul, ir in self.irreps_hidden
                if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
            ])
            
            # 门的不可约表示
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            # 创建门控层 - 使用正确的MindChemistry API
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            
            # 创建卷积层 - 使用MindChemistry的Convolution，添加必需参数
            conv = Convolution(
                irreps_node_input=irreps,
                irreps_node_attr=self.irreps_node_attr,
                irreps_node_output=gate.irreps_in,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_edge_scalars=o3.Irreps(f"{number_of_basis}x0e"),
                invariant_layers=radial_layers,
                invariant_neurons=radial_neurons,
                avg_num_neighbors=num_neighbors,
                nonlin_scalars={"e": "silu", "o": "tanh"},  # 添加必需的nonlin_scalars参数
            )
            
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        # 最后一层卷积
        self.layers.append(
            Convolution(
                irreps_node_input=irreps,
                irreps_node_attr=self.irreps_node_attr,
                irreps_node_output=self.irreps_out,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_edge_scalars=o3.Irreps(f"{number_of_basis}x0e"),
                invariant_layers=radial_layers,
                invariant_neurons=radial_neurons,
                avg_num_neighbors=num_neighbors,
                nonlin_scalars={"e": "silu", "o": "tanh"},  # 添加必需的nonlin_scalars参数
            )
        )
        self.drop_outs.append(Dropout(self.irreps_out, p))

    def preprocess(self, data):
        """
        预处理输入数据 - 与原版一致
        """
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = ms.ops.zeros(data.pos.shape[0], dtype=ms.int64)

        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_src = data.edge_index[0]
            edge_dst = data.edge_index[1]
            edge_vec = data.edge_vec
        else:
            edge_index = radius_graph(data.pos, self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        return batch, edge_src, edge_dst, edge_vec

    def construct(self, data):
        """
        前向传播 - 与原版一致
        """
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        
        # 计算球谐函数 - 与原版一致
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/test_network_edge_sh.npy"))
        # print((torch_res - edge_sh).abs().max())
        # 边长度嵌入 - 与原版一致
        edge_length = edge_vec.norm(dim=1)
        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/test_network_edge_length.npy"))
        # print((torch_res - edge_length).abs().max())
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=True,
        ) * (self.number_of_basis ** 0.5)
        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/test_network_edge_length_embedded.npy"))
        # print((torch_res - edge_length_embedded).abs().max())

        # 边属性 - 与原版一致
        edge_attr = smooth_cutoff(edge_length / self.max_radius).unsqueeze(-1) * edge_sh
        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/test_network_edge_attr.npy"))
        # print((torch_res - edge_attr).abs().max())

        # 处理节点输入特征
        if self.input_has_node_in and hasattr(data, 'x') and data.x is not None:
            assert self.irreps_in is not None
            x = data.x
        else:
            assert self.irreps_in is None
            x = ms.ops.ones((data.pos.shape[0], 1), dtype=ms.float32)

        # 处理节点属性
        if self.input_has_node_attr and hasattr(data, 'z') and data.z is not None:
            z = data.z
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = ms.ops.ones((data.pos.shape[0], 1), dtype=ms.float32)

        # 通过所有层 - 与原版一致
        for i, lay in enumerate(self.layers):
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            # torch_res = ms.Tensor(np.load(f"/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/test_network_x_{i}.npy"))
            # print((torch_res - x).abs().max())
            # Dropout
            if self.dropout and i < len(self.drop_outs):
                do = self.drop_outs[i]
                x = do(x)
        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/test_network_x.npy"))
        # print((torch_res - x).abs().max())

        # 输出聚合 - 与原版一致，使用Sharker的scatter
        if self.reduce_output:
            return scatter(x, batch, dim=0, reduce="sum") / (self.num_nodes ** 0.5)
        else:
            return x

class PeriodicNetwork(Network):
    """
    周期性网络的实现 - 与原版一致
    """
    def __init__(self, in_dim, em_dim, **kwargs):
        # 覆盖 reduce_output 关键字以执行原子贡献的平均
        self.pool = False
        if kwargs.get("reduce_output", False) == True:
            kwargs["reduce_output"] = False
            self.pool = True

        super().__init__(**kwargs)

        # 嵌入质量加权的独热编码 - 确保输出维度匹配irreps
        # 如果irreps_in不为None，em_dim应该匹配irreps_in的维度
        if self.irreps_in is not None:
            expected_dim = self.irreps_in.dim
        else:
            expected_dim = em_dim
            
        self.em = nn.Dense(in_dim, expected_dim)

    def construct(self, data):
        """
        周期性网络的前向传播 - 与原版一致
        """
        # 应用嵌入层
        embedded_x = nn.ReLU()(self.em(data.x))
        embedded_z = nn.ReLU()(self.em(data.z))
        
        # 创建新的数据对象或修改现有数据
        data.x = embedded_x
        data.z = embedded_z
        
        output = super().construct(data)
        output = nn.ReLU()(output)

        # 如果设置了pool_nodes，使用scatter进行聚合
        if self.pool == True:
            # 确保batch存在且有效
            if hasattr(data, 'batch') and data.batch is not None:
                output = scatter(output, data.batch, dim=0, reduce="mean")
            else:
                # 如果没有batch信息，直接返回平均值
                output = output.mean(axis=0, keep_dims=True)
        return output

class PeriodicNetworkPhdos(Network):
    """
    包含声子态密度的周期性网络 - 与原版一致
    """
    def __init__(self, in_dim, em_dim, out_dim, **kwargs):
        # 覆盖 reduce_output 关键字以执行原子贡献的平均
        self.pool = False
        if kwargs.get("reduce_output", False) == True:
            kwargs["reduce_output"] = False
            self.pool = True

        super().__init__(**kwargs)

        # 嵌入质量加权的独热编码
        self.em = nn.Dense(in_dim, em_dim)
        self.output = nn.Dense(out_dim * 2, out_dim)

    def construct(self, data):
        """
        前向传播 - 与原版一致
        """
        data.x = nn.ReLU()(self.em(data.x))
        data.z = nn.ReLU()(self.em(data.z))
        output = super().construct(data)
        output = nn.ReLU()(output)
        output = scatter(output, data.batch, dim=0, reduce="mean")
        output = ms.ops.concat((output, data.phdos), axis=1)
        output = self.output(output)
        output = nn.ReLU()(output)
        return output

# 损失函数类 - 与原版一致
class EMDLoss(nn.Cell):
    """
    Earth Mover's Distance Loss
    """
    def __init__(self):
        super().__init__()

    def construct(self, p, q):
        print(f' p = {p.shape}')
        cdf_p = ms.ops.cumsum(p, axis=1)
        cdf_q = ms.ops.cumsum(q, axis=1)
        emd = ms.ops.abs(cdf_p - cdf_q).sum(axis=1).mean()
        return emd

class WeightedMSELoss(nn.Cell):
    """
    加权均方误差损失
    """
    def __init__(self):
        super().__init__()
        
    def construct(self, inputs, targets, weights):
        return (((inputs - targets) ** 2) * weights).mean() 