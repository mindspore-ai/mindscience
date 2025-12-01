# E3NN：欧几里得神经网络

## 什么是 E3NN

E3NN 是一个基于 MindSpore 框架的等变神经网络库，专注于处理三维空间数据，并在旋转变换下保持网络的一致性。

**核心优势**：

- 旋转不变：分子旋转后预测结果保持不变
- 数据效率高：减少对大量数据增强的依赖
- 物理意义明确：遵循物理定律的对称性

## 基本概念

### 数据表示

E3NN 使用不可约表示（Irreps, Irreducible Representations）来描述不同类型的数据：

- `0e`：标量（如温度、能量）
- `1o`：向量（如位置、速度）
- `2e`：张量（如应力）

## 主要特性

### 1. 数据表示与操作

```python
from mindscience.e3nn import o3

# 创建不可约表示
irreps = o3.Irreps("2x0e + 3x1o")  # 2 个标量 + 3 个向量
print(irreps.dim)  # 总维度：2 + 9 = 11

# 生成随机数据
x = irreps.randn(-1)
```

### 2. 张量积运算

```python
from mindscience.e3nn import o3

# 组合不同类型特征
tp = o3.TensorProduct(
    irreps_in1="2x1o",      # 输入1：2 个向量
    irreps_in2="1x0e",      # 输入2：1 个标量
    irreps_out="2x1o"       # 输出：2 个向量
)
```

### 3. 等变神经网络层

```python
from mindscience.e3nn import nn
import mindspore.ops as ops

# 激活函数（仅作用于标量部分）
act = nn.Activation("3x0e + 2x1o", acts=[ops.tanh, None])

# 门控机制
gate = nn.Gate(
    irreps_scalars="8x0e",      # 标量通道
    acts=[ops.tanh],             # 标量激活函数
    irreps_gates="8x0e",        # 门控标量
    act_gates=[ops.sigmoid],     # 门控激活函数
    irreps_gated="8x1o"         # 被门控的向量通道
)
```

## 库结构

```text
mindscience.e3nn/
├── o3/                      # 基础数学与表示模块
│   ├── irreps.py           # 不可约表示（Irreps）
│   ├── tensor_product.py   # 张量积运算
│   ├── spherical_harmonics.py  # 球谐函数
│   ├── rotation.py         # 旋转矩阵与角度运算
│   ├── wigner.py           # Wigner D 矩阵
│   ├── norm.py             # 等变范数计算
│   └── sub.py              # 子表示操作
├── nn/                      # 神经网络层模块
│   ├── activation.py       # 等变激活函数
│   ├── gate.py             # 门控机制
│   ├── batchnorm.py        # 等变批归一化
│   ├── fc.py               # 等变线性层（全连接）
│   ├── normact.py          # 归一化-激活组合
│   ├── one_hot.py          # One-hot 编码
│   └── scatter.py          # 图聚合（scatter）操作
├── so2_conv/                # SO(2) 卷积与边框旋转
│   ├── __init__.py         # 公共 API 导出
│   ├── so2.py              # SO2Convolution 及子模块
│   ├── so3.py              # SO3Rotation，嵌入旋转/逆旋转
│   ├── wigner.py           # Wigner D 分块构造
│   └── init_edge_rot_mat.py# 由边向量构造旋转矩阵
└── utils/                   # 工具函数模块
    ├── batch_dot.py        # 批量点积运算
    ├── func.py             # 通用工具函数
    ├── initializer.py      # 参数初始化器
    ├── linalg.py           # 线性代数工具
    ├── ncon.py             # 张量网络收缩
    ├── perm.py             # 置换操作
    └── radius.py           # 半径图构造工具
```

### 模块详解

#### o3 模块（基础数学与表示）

- `irreps.py`：不可约表示的数据类型与维度定义
- `tensor_product.py`：等变张量积运算实现
- `spherical_harmonics.py`：球谐函数计算
- `rotation.py`：旋转矩阵生成、角度转换等操作
- `wigner.py`：用于旋转变换的 Wigner D 矩阵计算
- `norm.py`：等变范数计算
- `sub.py`：子表示提取与操作

#### nn 模块（神经网络层）

- `activation.py`：等变激活函数，仅作用于标量部分
- `gate.py`：控制向量特征激活的门控机制
- `batchnorm.py`：等变批归一化层
- `fc.py`：等变线性层（全连接）
- `normact.py`：归一化与激活的组合层
- `one_hot.py`：One-hot 编码工具
- `scatter.py`：图神经网络中的聚合操作

#### utils 模块（工具函数）

- `batch_dot.py`：高效批量点积操作
- `func.py`：常用工具函数集合
- `initializer.py`：网络参数初始化器
- `linalg.py`：线性代数相关工具
- `ncon.py`：张量网络收缩操作
- `perm.py`：置换与转置操作
- `radius.py`：构造半径图的辅助函数

#### so2_conv 模块（SO(2) 卷积）

- 用途：在与边对齐的局部坐标系下，对磁量子数 `m` 通道进行等变卷积，保持沿边轴的 SO(2) 对称性。

**关键组件**

- `SO2Convolution`：面向 `Irreps` 输入/输出的按 `m` 通道混合网络；区分 `m=0`（实值）与 `m>0`（成对通道），并按 `l` 汇组输出。
- `SO3Rotation`：将边旋转矩阵转换为 Wigner D 分块，用于嵌入在局部坐标系与全局坐标系之间的旋转/逆旋转。
- `init_edge_rot_mat`：从边向量稳健构造 3×3 旋转矩阵。

```python
from mindscience.e3nn.so2_conv import SO2Convolution, SO3Rotation
from mindscience.e3nn.so2_conv.init_edge_rot_mat import init_edge_rot_mat

irreps_in = "2x0e + 1x1o"
irreps_out = "1x0e + 1x1o"
so2 = SO2Convolution(irreps_in, irreps_out)
so3 = SO3Rotation(lmax=1, irreps_in=irreps_in, irreps_out=irreps_out)

# edge_vecs: [num_edges, 3]
edge_rot = init_edge_rot_mat(edge_vecs)              # [num_edges, 3, 3]
wigner, wigner_inv = so3.set_wigner(edge_rot)

embedding_local = so3.rotate(embedding_global, wigner)
out_tuple = so2(embedding_local, edge_attrs)         # 使用边数（num_edges）
out_global = so3.rotate_inv(out_tuple, wigner_inv)
```

**说明**

- `SO2Convolution` 接收已旋转到局部边框的嵌入（按输入 irreps 的元组），并从 `x_edge` 获取 `num_edges`。
- `SO3Rotation.set_wigner` 基于边旋转生成 `l ∈ [0, lmax]` 的 Wigner D 分块。
- 典型流程：构建边旋转 → Wigner D → 局部旋转 → SO(2) 卷积 → 逆旋转。

## 核心组件示例

### 1. 不可约表示（Irreps）

```python
from mindscience.e3nn import o3

# 创建不可约表示
irreps = o3.Irreps("2x0e + 3x1o + 1x2e")
print(irreps.dim)        # 总维度：2 + 9 + 5 = 16
print(irreps.ls)         # 角动量量子数：[0, 1, 2]

# 生成数据
x = irreps.randn(-1)
```

### 2. 张量积运算

```python
from mindscience.e3nn import o3

# 全连接张量积
tp = o3.TensorProduct(
    irreps_in1="2x1o",      # 输入1：向量
    irreps_in2="1x0e",      # 输入2：标量
    irreps_out="2x1o"       # 输出：向量
)

result = tp(x1, x2)  # 默认 weight_mode="inner"，无需手动提供权重
```

### 3. 等变网络层

```python
from mindscience.e3nn import nn
import mindspore.ops as ops

# 激活函数（仅作用于标量）
act = nn.Activation("3x0e", acts=[ops.tanh])

# 门控机制
gate = nn.Gate(
    irreps_scalars="2x0e",      # 标量通道
    acts=[ops.tanh],             # 标量激活函数
    irreps_gates="2x0e",        # 门控标量
    act_gates=[ops.sigmoid],     # 门控激活函数
    irreps_gated="2x1o"         # 被门控的向量通道
)

# 批归一化
bn = nn.BatchNorm("2x0e + 3x1o")
```

### 4. 球谐函数

```python
from mindscience.e3nn import o3
import mindspore as ms

# 计算球谐函数
pos = ms.Tensor([[1.0, 0.0, 0.0]])  # 位置向量
sh = o3.spherical_harmonics(l=2, x=pos, normalize=True)
```

### 5. 旋转与 Wigner D 矩阵

```python
from mindscience.e3nn import o3
import mindspore as ms

# 构造旋转矩阵
alpha, beta, gamma = 0.1, 0.2, 0.3  # 欧拉角
R = o3.angles_to_matrix(alpha, beta, gamma)

# 对不可约表示施加旋转
irreps = o3.Irreps("1x1o")  # 一个向量
x = irreps.randn(-1)
D = irreps.wigD_from_matrix(R)  # Wigner D 矩阵
x_rotated = D @ x  # 旋转后的向量
```

### 6. 等变线性层

```python
from mindscience.e3nn import o3

# 创建等变线性层
linear = o3.Linear(
    irreps_in="2x0e + 1x1o",   # 输入：2 个标量 + 1 个向量
    irreps_out="1x0e + 2x1o"   # 输出：1 个标量 + 2 个向量
)

# 前向计算
x = o3.Irreps("2x0e + 1x1o").randn(-1)
y = linear(x)
```

### 7. 批归一化

```python
from mindscience.e3nn import nn

# 等变批归一化
bn = nn.BatchNorm("4x0e + 2x1o")

# 应用归一化
x = o3.Irreps("4x0e + 2x1o").randn(32, -1)  # batch=32
x_normalized = bn(x)
```

### 8. 范数计算

```python
from mindscience.e3nn import o3

# 针对不同不可约表示计算范数
irreps = o3.Irreps("2x0e + 3x1o")
x = irreps.randn(-1)

# 使用 Norm 计算范数
norm_layer = o3.Norm(irreps)
norm_result = norm_layer(x)
```

### 9. Scatter 聚合

```python
from mindscience.e3nn import nn
import mindspore as ms

# 图神经网络中的 scatter 聚合
scatter = nn.Scatter(mode="add")  # 支持：'add', 'sum', 'div', 'max', 'min', 'mul'

# 示例
src = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 源特征
index = ms.Tensor([0, 0, 1], dtype=ms.int32)           # 目标索引
result = scatter(src, index, dim_size=2)
```

## 快速开始

### 基本使用流程

```python
from mindscience.e3nn import o3
import mindspore as ms

# 1. 定义数据类型
irreps_in = o3.Irreps("3x0e + 2x1o")   # 3 个标量 + 2 个向量
irreps_out = o3.Irreps("1x0e")         # 1 个标量输出

# 2. 创建等变层
layer = o3.Linear(irreps_in, irreps_out)

# 3. 前向传播
x = irreps_in.randn(-1)  # 生成输入数据
y = layer(x)             # 等变变换

print(f"Input dimension: {x.shape}")
print(f"Output dimension: {y.shape}")
```

### 构建简单网络

```python
from mindscience.e3nn import o3, nn
import mindspore as ms
import mindspore.nn as ms_nn
import mindspore.ops as ops

class SimpleE3NN(ms_nn.Cell):
    def __init__(self):
        super().__init__()
        # 特征抽取
        self.linear1 = o3.Linear("3x0e + 1x1o", "8x0e + 4x1o")
        self.act = nn.Activation("8x0e + 4x1o", acts=[ops.tanh, None])

        # 输出层
        self.linear2 = o3.Linear("8x0e + 4x1o", "1x0e")

    def construct(self, x):
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(x)

# 使用模型
model = SimpleE3NN()
input_data = o3.Irreps("3x0e + 1x1o").randn(-1)
output = model(input_data)
```