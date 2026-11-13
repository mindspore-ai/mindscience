# TP/CP 混合并行库（distributed）

## 背景介绍

在大模型训练中，单卡显存往往无法容纳完整的模型或序列。张量并行（TP）把权重切到多张卡上，上下文并行（CP）把长序列切到多张卡上，与之配套的序列并行（SP）让层间激活也保持切分状态、不必每层拼回全量。三者叠加时会出现"同一份权重在不同维度上被重复切分"的问题：哪些参数是分片、哪些是副本、哪些梯度是部分和，一旦判断错，训练不会报错，只会静默地差 `size` 倍。

本项目提供了一套与模型无关的并行基础设施，把这类判断从训练脚本里收进库里，核心思路如下：

- **正交进程组**：给定 `(tp, cp, pp)` 与 worldsize，按 `tp / cp / dp / dp-cp` 四个维度一次性划出互不重叠的通信组。任意两个维度可以自由组合，成员划分保持 Megatron 的 rank 布局。
- **带反向的通信原语**：`all_reduce` / `all_gather` / `reduce_scatter` / `take_shard` 四个原语按 `dim` 参数化，并配有四个手写反向的 `_Function` 包装，使"前向拼起来、反向切回去"由计算图自动完成，不在每个调用点手写通信。
- **显式的梯度计划**：把所有可训练参数一次性分成 `sharded` / `complete` / `partial` 三桶，配合按 dtype 融合的归约与正确的范数计算，替代"按参数名子串猜"的启发式规则。

全库遵循一条硬约束：**组大小为 1 时所有路径返回入参本身、不发起任何通信**。因此 `tp=cp=1` 的运行结果与未接入本库的串行基线逐位相同，单卡即可完成回归，多卡只用于验证数值等价。

## 项目结构

```

GNN_TPCP/
├── distributed/
│    ├── __init__.py        # 包初始化，导出进程组、并行层与梯度计划
│    ├── context.py         # 并行上下文：正交进程组 Group / init_parallel / group
│    ├── comm.py            # 通信原语与四个带反向的 _Function 包装
│    ├── layers.py          # 列/行并行线性层、权重初始化、参数标记
│    └── grad.py            # 梯度归约计划 ReducePlan、融合归约、全局范数
├── requirements.txt   # 运行期依赖声明
├── README_CN.md       # 项目中文说明文档
└── README.md          # 项目英文说明文档
```

各模块职责说明：

| 模块 | 说明 |
|------|------|
| `context.py` | `Group` 类（`.size` / `.rank` / `.name`）、`_group_members` 正交分组算法、`init_parallel` 建组入口 |
| `comm.py` | `GatherMode` 枚举；`all_reduce`、`all_gather(dim)`、`reduce_scatter(dim)`、`take_shard(dim)` 四个原语；`CopyToAll` / `Reduce` / `Gather` / `ReduceScatter` 四个 autograd 包装 |
| `layers.py` | `initialize_affine_weight`（全局形状初始化再切片）、`ColumnParallelLinear`、`RowParallelLinear`、`Shard`；参数标记 `mark_sharded` / `mark_grad_complete` 与查询 `tag_of` / `is_sharded` / `is_grad_complete` |
| `grad.py` | `ReducePlan` 三桶结构、`build_plan` 分类、`reduce_grads` 按 dtype 融合归约、`global_norm` 分片感知的 L2 范数 |

## 环境要求

### 硬件环境

| 硬件 | 说明 |
|------|------|
| Ascend 处理器 | 推荐 Ascend 910 系列，其中 `tp * cp * pp > 1` 时必须使用多卡进行分布式训练 |
| CPU | 可用于 `size == 1` 路径的单元测试与逻辑校验 |

### 软件环境

| 软件 | 版本要求 |
|------|----------|
| Python | >= 3.9 |
| MindSpore | >= 2.4.0（需提供 `mint` 与 `communication.comm_func`） |

> 本库依赖 MindSpore 的 `_Function` 私有接口实现自定义反向。升级 MindSpore 版本时需回归 `comm.py` 的四个包装，确认 `forward` / `backward` 的签名与返回值个数未变。

## 安装与配置

### 1. 安装

本库以源码形式随训练工程分发，将其放到工程根目录并保证可被 `import` 即可，无需单独打包：

```bash
pip install -r requirements.txt
```

运行期只依赖 MindSpore，其余 import 全部来自标准库（`functools` / `operator` / `enum` / `math` / `dataclasses`）。Ascend 环境请从 [MindSpore 官网](https://www.mindspore.cn/install) 安装与固件匹配的 whl：PyPI 上的 wheel 是 CPU 构建，不含 HCCL 集合通信，装上去之后 `comm.init()` 会失败。

如需设为独立包，在工程根目录添加 `pyproject.toml` 后 `pip install -e .`。

### 2. 配置并行拓扑

拓扑由 `tp / cp / pp` 三个参数决定，`dp` 由世界大小自动推导，无需手动指定：

```
world_size = tp * cp * dp * pp
```

`init_parallel` 要求先完成 `mindspore.communication.init()`，并要求 `world_size` 能被 `tp * cp * pp` 整除，否则直接抛错而不是静默降级。rank 布局由 `order` 参数决定，默认 `"tp-cp-dp-pp"`，即 tp 为最内层维度。更换 `order` 会改变组划分，但不会改变"每位 rank 在每个维度上恰好属于一个组"这一性质。

可配置项如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tp` | `1` | 张量并行度，决定权重沿输出/输入维切几片 |
| `cp` | `1` | 上下文并行度，决定序列切几段 |
| `pp` | `1` | 流水并行度，仅参与 rank 布局与 `dp` 推导，本库不提供流水调度 |
| `order` | `"tp-cp-dp-pp"` | rank 布局顺序，靠前的维度为内层（stride 更小） |

### 3. 替换线性层

把模型中的 `nn.Dense` / `mint.nn.Linear` 换成对应的并行层。切分维度由层类型决定，调用方不需要再传：

| 原层 | 替换为 | 切分对象 |
|------|--------|----------|
| `mint.nn.Linear(in, out)` | `ColumnParallelLinear(in, out)` | 切 `out_features` |
| `mint.nn.Linear(in, out)` | `RowParallelLinear(in, out)` | 切 `in_features` |

### 4. 建立梯度计划

训练开始前对 `model.trainable_params()` 调用一次 `build_plan`，训练循环中复用：

```python
from distributed import build_plan

plan = build_plan(model.trainable_params(), fuse=False)
```

`fuse=True` 会按 dtype 把待归约的梯度拼成桶，减少通信次数，但需要额外的临时显存，默认关闭。

## 使用方法

### 初始化与拓扑查看
示例：

```python
import mindspore.communication as comm
from distributed import init_parallel, group

comm.init()
init_parallel(tp=2, cp=2, pp=1)

tp, cp, dp = group("tp"), group("cp"), group("dp")
print(f"world={comm.get_group_size()}, tp={tp.size}, cp={cp.size}, dp={dp.size}, rank={tp.rank}")
```

`world=8`、`tp=2`、`cp=2` 时的组划分（实测）：

| 维度 | 组大小 | 分组结果 | 用途 |
|------|--------|----------|------|
| `tp` | 2 | `[0,1] [2,3] [4,5] [6,7]` | 权重切分与激活拼接 |
| `cp` | 2 | `[0,2] [1,3] [4,6] [5,7]` | 序列切分 |
| `dp` | 2 | `[0,4] [1,5] [2,6] [3,7]` | 数据并行梯度归约 |
| `dp-cp` | 4 | `[0,2,4,6] [1,3,5,7]` | DP 与 CP 合并的一次归约 |

日志示例：

```
Parallel topology: world_size=8, tp=2, cp=2, dp=2, pp=1
rank 3 -> tp group [2,3] rank 1 | cp group [1,3] rank 1 | dp group [3,7] rank 0
```

### 张量并行（TP）

列并行与行并行必须成对使用：列并行切输出维、产生分片输出，行并行切输入维、消费分片输入并归约回全量。

示例：

```python
from mindspore import nn
from distributed import ColumnParallelLinear, RowParallelLinear

class TPFeedForward(nn.Cell):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff, bias=False, gather_output=False)
        self.fc2 = RowParallelLinear(d_ff, d_model, bias=False, input_is_parallel=True)
        self.act = nn.GELU()

    def construct(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

`gather_output=False` 让列并行层保留分片输出，行并行层以 `input_is_parallel=True` 直接消费，两层之间不需要任何通信。`gather_output=True` 会在出口把分片拼成全量，只适用于下游不是并行层的场合（例如输出投影接损失函数）。这两个参数必须对应设置：`gather_output=True` 配 `input_is_parallel=True` 会让全量输入直接乘上分片权重，在 `matmul` 处因维数不匹配报错——这类错误是显式失败，比静默算错好排查。反过来两个都传 `False` 同样会报错，只是报错位置在下一层入口。

### 序列并行与上下文并行（CP）

CP 把序列维切开，层间的激活因此也是分片的。此时用 `use_sequence_parallel=True` 与 `seq_dim` 指定"哪一维是序列"，列并行层在入口把序列拼回、行并行层在出口归约并重新切开：

示例：

```python
class SPFeedForward(nn.Cell):
    def __init__(self, d_model, d_ff, seq_dim=1):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff, gather_output=False,
                                        use_sequence_parallel=True, seq_dim=seq_dim)
        self.fc2 = RowParallelLinear(d_ff, d_model, input_is_parallel=True,
                                     use_sequence_parallel=True, seq_dim=seq_dim)
        self.act = nn.GELU()

    def construct(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

三点约定：

1. `gather_output=False` 与 `input_is_parallel=True` 是序列并行的固定配对，配错会在 `matmul` 处报维数不匹配；
2. `seq_dim` 由宿主指定，因为不同模型的序列维可能是第 0 维或第 1 维。这是本库唯一无法用形状发现的配置项：各维长度接近时选错不会报错，只会静默算错序列；
3. CP 的注意力部分（跨 rank 交换 K/V）不在本库范围内，本库只提供进程组与层间激活的切分能力。

### 训练循环三步

梯度同步分三步，顺序不能调换。下例中 `grad_fn`、`dp_all_reduce`、`clip_by_global_norm` 为宿主侧的示意实现：

```python
from distributed import build_plan, reduce_grads, global_norm
from distributed.comm import all_reduce
from distributed import group

tp = group("tp")
plan = build_plan(model.trainable_params(), fuse=False)   # 训练前一次

for step in range(total_steps):
    grads = grad_fn(model, batch)                          # 反向已由原语自动完成 TP 通信
    grads = reduce_grads(grads, plan, lambda x: all_reduce(x, tp))
    grads = dp_all_reduce(grads)                           # 数据并行归约，必须在上一行之后
    norm = global_norm(grads, plan=plan, all_reduce=lambda s: all_reduce(s, tp))
    grads = clip_by_global_norm(grads, norm, max_norm)
    optimizer(grads)
```

顺序的理由：

- `reduce_grads` 只处理 TP 维的**部分和**，必须早于 DP 归约；否则 DP 归约会把尚未补全的部分和当作完整梯度在 `dp` 组内累加，更新量偏大且难以归因；
- `global_norm` 只对 `sharded` 参数补一次跨组求和，必须早于裁剪，否则裁掉的是未经修正的范数；
- 三步在对应桶为空时都是空操作：`partial` 为空时 `reduce_grads` 直接返回原对象，不发起任何通信。

### 单卡等价性

不调用 `init_parallel` 或全传 1 时，所有组退化为 `size == 1`，全部原语返回入参本身：

```python
from mindspore import mint
from distributed import group
from distributed.comm import all_gather

x = mint.ones((2, 4))
assert all_gather(x, group("tp")) is x      # 同一对象，无拷贝、无通信
```

这条性质让同一份代码可以在单卡上跑回归：`tp=cp=1` 的结果应与串行基线**逐位相同**，而不是"误差在 1e-6 以内"。

### 验证要点

接入后建议依次确认以下四件事，都可以在单卡或双卡上完成：

1. **单卡逐位**：`tp=cp=1` 跑若干步，`loss` 与未接入本库的基线做字符串级或 `==` 比较；
2. **权重可切片**：同一 seed 下，`size=N` 的初始权重应等于 `size=1` 权重的对应切片（把 N 个分片按 `rank` 顺序拼接后比对）；
3. **参数口径**：打印三桶长度，`len(partial) + len(complete) + len(sharded)` 必须等于 `len(params)`；
4. **范数一致**：`tp=1` 与 `tp=N` 的全局范数应基本相等；若相差 `√tp`，说明 `complete` 标漏或标多。

## 核心机制

### 正交分组

给定各维度大小 `sizes` 与一个布尔 `mask`（哪些维度属于该组），rank 按 Megatron 布局编码为 `rank = Σ coord_i · prefix_prod(sizes)[i]`。选取 `mask` 为真的轴，**其他轴坐标相同的 rank 落入同一组**，因此任意两个维度天然正交，不需要为每种组合单独写分组逻辑。

`world=8` 的三组实测配置：

| 配置 | `tp` 组 | `cp` 组 | `dp` 组 |
|------|---------|---------|---------|
| `tp=2, cp=2, pp=1` | `[0,1] [2,3] [4,5] [6,7]` | `[0,2] [1,3] [4,6] [5,7]` | `[0,4] [1,5] [2,6] [3,7]` |
| `tp=4, cp=2, pp=1` | `[0,1,2,3] [4,5,6,7]` | `[0,4] [1,5] [2,6] [3,7]` | 每组 1 个 rank（`dp=1`） |
| `tp=2, cp=2, pp=2` | `[0,1] [2,3] [4,5] [6,7]` | `[0,2] [1,3] [4,6] [5,7]` | 每组 1 个 rank（`dp=1`） |

三种配置下各组都是全体 rank 的一个精确分割：没有 rank 被漏掉，也没有 rank 重复入组。

### 通信原语与反向矩阵

原语本身只做前向，方向由包装类的 `backward` 补齐。这张表是本库最需要逐行核对的部分：

| 包装 | forward | backward | 使用位置 |
|------|---------|----------|----------|
| `CopyToAll` | 恒等 | `all_reduce` | 列并行层入口（无序列并行） |
| `Reduce` | `all_reduce` | 恒等 | 行并行层出口（无序列并行） |
| `Gather(dim, REDUCE)` | `all_gather` | `reduce_scatter` | 列并行层入口（序列并行） |
| `Gather(dim, SLICE)` | `all_gather` | `take_shard`（零通信） | 列并行层出口拼接 |
| `ReduceScatter(dim)` | `reduce_scatter` | `all_gather` | 行并行层出口（序列并行） |
| `Shard(dim)` | `take_shard`（零通信） | `all_gather` | 行并行层入口 |

`GatherMode.REDUCE` 与 `SLICE` 的区别是"下游拿到的是分片还是副本"：拿到分片时反向必须 `reduce_scatter` 把部分和归约掉，拿到副本时每张卡已算出完整梯度，反向只需切片。两者用反会静默差 `1/tp` 或 `×tp`，且形状完全相同，因此必须靠显式枚举而不是布尔开关表达。

### 三个梯度桶

参数与梯度的属性是两个不同的问题，混在一起是 `size` 倍错误的根源：

| 桶 | 参数形态 | 本地梯度 | 需要 `all_reduce` | 需要范数归约 |
|----|----------|----------|------------------|--------------|
| `sharded` | 全局权重的 `1/size` | 已是全局值 | 否 | 是 |
| `complete` | 副本 | 已是全局值 | 否 | 否 |
| `partial` | 副本 | 部分和 | 是 | 否 |

分类依据是参数上的标记而非参数名：`mark_sharded` 表示"这个参数只有一片"，`mark_grad_complete` 表示"副本且梯度已完整"，未标记的一律按 `partial` 处理——**默认从严**，漏标只会多做一次归约，不会漏归约。

### 按 dtype 融合

`partial` 桶按 dtype 分组，组内 `cat` 成一条扁平 buffer，每桶一次 `all_reduce`，再按 `offsets` 切回原梯度。必须按 dtype 分组，因为混合 dtype 拼接会触发隐式提升（`bf16` → `fp32`），让一部分梯度多走一次类型转换；同时通信量也会按更高精度结算。

融合不改变数值：`all_reduce` 是逐元素操作，buffer 内每个元素的求和序列与单独通信时相同，因此结果逐位等价于逐参数归约。代价是融合期间该 dtype 的梯度瞬时占用接近翻倍，并多一次 `cat` 拷贝，这是默认关闭 `fuse` 的主要原因。

## 宿主与库的边界

本库刻意不做以下事情，它们应由训练工程负责：

| 事项 | 归属 | 原因 |
|------|------|------|
| 模型结构改造（哪一层用列并行、哪一层用行并行） | 宿主 | 取决于模型的语义，库无法推断 |
| CP 的注意力实现（跨 rank 交换 K/V） | 宿主 | ring-attention 的形态随注意力变体而变，强行抽象会限制实现 |
| 优化器、混合精度、梯度裁剪策略 | 宿主 | 与并行正交 |
| 手工构造的层注册为 `partial` | 宿主 | 通过 `build_plan(..., extra_partial=...)` 显式传入，库不猜 |
| 按逻辑组切分时的 `num_groups` | 宿主 | 例如按注意力头切分时必传，否则分片可能横跨两个头 |

`num_groups` 的校验规则是 `num_groups % group.size == 0`。注意不能用"轴长度能被组大小整除"代替：轴长 48、6 个逻辑组、切 4 片时整除成立，但每片会跨 1.5 个组，切出来的权重无法与逻辑组对应。

## 已知限制

- CP 目前只提供进程组与层间激活的切分能力，未包含 ring-attention；
- 顶层 `__init__.py` 尚未导出函数式原语，`all_reduce` / `all_gather` / `take_shard` / `reduce_scatter` 需从 `distributed.comm` 导入，`tag_of` 需从 `distributed.layers` 导入；
- 单卡路径下两个并行层内部持有一个原生 `Linear` 子 Cell，参数键名会多一层 `layer.` 前缀，加载既有 ckpt 时需要做键名映射；
- `all_gather` / `reduce_scatter` 的任意维支持依赖 `mint.movedim`，在缺少该接口的版本上需退化为仅支持首维与末维；
- 参数标记表持有参数的强引用且不自动清理，反复构造模型的长进程需要显式释放。

## 许可证说明

本项目遵循 **Apache License 2.0** 开源协议。

核心条款：

- **授权**：允许自由使用、修改、分发与商业化，无需额外授权
- **归属**：分发时必须保留原始版权声明与许可证文本
- **专利**：许可证涵盖贡献者授权的专利使用权
- **免责**：软件按"原样"提供，不承担任何明示或暗示的担保责任
- **责任限制**：作者/贡献者对因使用本软件导致的任何损失不承担责任

完整许可证文本请参阅项目源码头部引用的 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)。
