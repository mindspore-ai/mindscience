# Distributed library for TP/CP mixed parallelism

## Background

During large-model training a single device rarely holds the full model, and almost never the full sequence. Tensor parallelism (TP) splits weights across devices; context parallelism (CP) splits a long sequence across devices; sequence parallelism (SP) keeps the inter-layer activations split as well, so they are not gathered back after every layer. Combine the three and the same weight ends up split along several axes at once: which parameters are shards, which are replicas, which gradients are partial sums. Get that judgement wrong and training does not fail — it silently drifts by a factor of `size`.

This library provides model-agnostic parallel infrastructure and moves those judgements out of the training script:

- **Orthogonal process groups**: given `(tp, cp, pp)` and the world size, it carves out non-overlapping groups for `tp / cp / dp / dp-cp` in one pass. Any two axes can be combined freely, and membership follows the Megatron rank layout.
- **Primitives with hand-written backward**: `all_reduce` / `all_gather` / `reduce_scatter` / `take_shard` are parameterised by `dim`, and four `_Function` wrappers supply their backward passes, so "gather forward, shard backward" is handled by the graph instead of at every call site.
- **An explicit gradient plan**: every trainable parameter is classified once into `sharded` / `complete` / `partial`, paired with dtype-fused reduction and a correct global norm, replacing the "guess by parameter-name substring" heuristic.

The whole library follows one hard rule: **when a group has size 1, every path returns its input object and issues no communication.** A `tp=cp=1` run is therefore bit-identical to a serial baseline that never uses the library, so regression testing needs a single device; multiple devices are only required to prove numerical equivalence.

## Project structure

```

GNN_TPCP/
├── distributed/
│    ├── __init__.py        # package exports: process groups, parallel layers, gradient plan
│    ├── context.py         # parallel context: orthogonal groups, init_parallel, group
│    ├── comm.py            # collective primitives and the four _Function wrappers
│    ├── layers.py          # column/row parallel linear layers, weight init, parameter tags
│    └── grad.py            # ReducePlan, fused reduction, global norm
├── requirements.txt   # runtime dependencies
├── README_CN.md       # Chinese documentation
└── README.md          # English documentation
```

Module responsibilities:

| Module | Description |
|--------|-------------|
| `context.py` | `Group` (`.size` / `.rank` / `.name`), the `_group_members` orthogonal grouping algorithm, and the `init_parallel` entry point |
| `comm.py` | `GatherMode`; the `all_reduce`, `all_gather(dim)`, `reduce_scatter(dim)` and `take_shard(dim)` primitives; the `CopyToAll` / `Reduce` / `Gather` / `ReduceScatter` autograd wrappers |
| `layers.py` | `initialize_affine_weight` (build at global shape, then slice), `ColumnParallelLinear`, `RowParallelLinear`, `Shard`; tags `mark_sharded` / `mark_grad_complete` and queries `tag_of` / `is_sharded` / `is_grad_complete` |
| `grad.py` | the `ReducePlan` three-bucket structure, `build_plan` classification, dtype-fused `reduce_grads`, and the shard-aware `global_norm` |

## Requirements

### Hardware

| Hardware | Notes |
|----------|-------|
| Huawei Ascend | Ascend 910 series recommended; multiple devices are required when `tp * cp * pp > 1` |
| CPU | Sufficient for unit tests and logic checks on the `size == 1` path |

### Software

| Software | Version |
|----------|---------|
| Python | >= 3.9 |
| MindSpore | >= 2.4.0 (must provide `mint` and `communication.comm_func`) |

> The library relies on MindSpore's private `_Function` interface for custom backward passes. When upgrading MindSpore, re-run the checks for the four wrappers in `comm.py` and confirm that the `forward` / `backward` signatures and their return arity are unchanged.

## Installation and configuration

### 1. Install

The library ships as source inside the training project. Drop it into the project root and make sure it is importable; no separate packaging step is needed:

```bash
pip install -r requirements.txt
```

The only runtime dependency is MindSpore; every other import comes from the standard library (`functools` / `operator` / `enum` / `math` / `dataclasses`). On Ascend, install the wheel that matches your firmware from [mindspore.cn](https://www.mindspore.cn/install): the PyPI wheel is a CPU build without HCCL collectives, and `comm.init()` will fail once it is installed on a multi-device host.

To publish the library as a standalone package, add a `pyproject.toml` at the project root and run `pip install -e .`.

### 2. Configure the parallel topology

The topology is determined by `tp / cp / pp`; `dp` is derived from the world size and never set by hand:

```
world_size = tp * cp * dp * pp
```

`init_parallel` requires a prior `mindspore.communication.init()` call and requires `world_size` to be divisible by `tp * cp * pp`; otherwise it raises instead of degrading silently. The rank layout is set by `order`, defaulting to `"tp-cp-dp-pp"` where `tp` is the innermost axis. Changing `order` changes the group partition but not the invariant that every rank belongs to exactly one group per axis.

Available settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tp` | `1` | Tensor parallel degree: how many shards the weight is cut into along its output/input axis |
| `cp` | `1` | Context parallel degree: how many chunks the sequence is cut into |
| `pp` | `1` | Pipeline parallel degree: affects only the rank layout and the derived `dp`; pipeline scheduling is not provided |
| `order` | `"tp-cp-dp-pp"` | Rank layout order; earlier axes are the inner ones (smaller stride) |

### 3. Replace linear layers

Replace `nn.Dense` / `mint.nn.Linear` with the matching parallel layer. The split axis follows from the layer type, so callers never pass it:

| Original layer | Replacement | What gets split |
|----------------|-------------|-----------------|
| `mint.nn.Linear(in, out)` | `ColumnParallelLinear(in, out)` | `out_features` |
| `mint.nn.Linear(in, out)` | `RowParallelLinear(in, out)` | `in_features` |

### 4. Build the gradient plan

Call `build_plan` once on `model.trainable_params()` before training starts, then reuse the plan in the loop:

```python
from distributed import build_plan

plan = build_plan(model.trainable_params(), fuse=False)
```

`fuse=True` groups the gradients awaiting reduction by dtype into buckets, which cuts the number of collectives but needs extra temporary memory. It is off by default.

## Usage

### Initialise and inspect the topology

```python
import mindspore.communication as comm
from distributed import init_parallel, group

comm.init()
init_parallel(tp=2, cp=2, pp=1)

tp, cp, dp = group("tp"), group("cp"), group("dp")
print(f"world={comm.get_group_size()}, tp={tp.size}, cp={cp.size}, dp={dp.size}, rank={tp.rank}")
```

Group partition for `world=8`, `tp=2`, `cp=2` (measured):

| Axis | Group size | Partition | Purpose |
|------|------------|-----------|---------|
| `tp` | 2 | `[0,1] [2,3] [4,5] [6,7]` | weight sharding and activation gathering |
| `cp` | 2 | `[0,2] [1,3] [4,6] [5,7]` | sequence sharding |
| `dp` | 2 | `[0,4] [1,5] [2,6] [3,7]` | data-parallel gradient reduction |
| `dp-cp` | 4 | `[0,2,4,6] [1,3,5,7]` | one reduction covering both DP and CP |

Sample log:

```
Parallel topology: world_size=8, tp=2, cp=2, dp=2, pp=1
rank 3 -> tp group [2,3] rank 1 | cp group [1,3] rank 1 | dp group [3,7] rank 0
```

### Tensor parallelism (TP)

Column and row parallel layers are used as a pair: the column layer splits the output axis and produces a shard, the row layer splits the input axis, consumes that shard and reduces the result back to a replica.

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

With `gather_output=False` the column layer keeps its sharded output and the row layer consumes it directly through `input_is_parallel=True`; no communication happens between the two. `gather_output=True` gathers the shards into a full replica at the exit and is only appropriate when the consumer is not a parallel layer (an output projection feeding the loss, for example). The two flags must be set consistently: `gather_output=True` together with `input_is_parallel=True` multiplies a full input by a sharded weight and fails in `matmul` on a dimension mismatch — an explicit failure, which is easier to diagnose than a wrong number. Passing `False` for both fails as well, only at the next layer's entry.

### Sequence and context parallelism (CP)

CP splits the sequence axis, so the activations between layers are sharded too. Use `use_sequence_parallel=True` plus `seq_dim` to declare which axis is the sequence: the column layer gathers the sequence back at its entry, and the row layer reduces at its exit and re-splits it.

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

Three conventions:

1. `gather_output=False` with `input_is_parallel=True` is the fixed pairing for sequence parallelism; flipping either flag (so that both are `True`, or both are `False`) fails in `matmul` on a dimension mismatch.
2. `seq_dim` is supplied by the host, because the sequence axis may be axis 0 or axis 1 depending on the model. This is the one setting the library cannot validate from shapes: when the axis lengths are similar, a wrong value does not raise, it silently shards the wrong axis.
3. The attention side of CP — exchanging K/V across ranks — is out of scope. The library provides the process group and the inter-layer activation sharding only.

### The three gradient steps

Gradient synchronisation runs in three steps, and the order is not interchangeable. In the example below `grad_fn`, `dp_all_reduce` and `clip_by_global_norm` are host-side stand-ins:

```python
from distributed import build_plan, reduce_grads, global_norm
from distributed.comm import all_reduce
from distributed import group

tp = group("tp")
plan = build_plan(model.trainable_params(), fuse=False)   # once, before training

for step in range(total_steps):
    grads = grad_fn(model, batch)                          # backward already handled TP comms
    grads = reduce_grads(grads, plan, lambda x: all_reduce(x, tp))
    grads = dp_all_reduce(grads)                           # data-parallel, must come after
    norm = global_norm(grads, plan=plan, all_reduce=lambda s: all_reduce(s, tp))
    grads = clip_by_global_norm(grads, norm, max_norm)
    optimizer(grads)
```

Why the order matters:

- `reduce_grads` handles the **partial sums** along the TP axis and must run before the DP reduction; otherwise the DP reduction accumulates an unfinished partial sum as if it were a complete gradient, inflating the update in a way that is hard to trace;
- `global_norm` adds one cross-group sum for `sharded` parameters only and must run before clipping, otherwise the norm being clipped is the uncorrected one;
- all three are no-ops when their bucket is empty: with an empty `partial` bucket, `reduce_grads` returns the original object and issues no communication.

### Single-device equivalence

When `init_parallel` is not called, or every degree is 1, all groups degrade to `size == 1` and every primitive returns its input object:

```python
from mindspore import mint
from distributed import group
from distributed.comm import all_gather

x = mint.ones((2, 4))
assert all_gather(x, group("tp")) is x      # same object, no copy, no communication
```

This is what lets one code path be regressed on a single device: a `tp=cp=1` run must be **bit-identical** to the serial baseline, not "within 1e-6".

### What to verify

Four checks after integration, all of them doable on one or two devices:

1. **Single-device bit-exactness**: run a few steps with `tp=cp=1` and compare `loss` against a baseline that does not use the library, by string or by `==`;
2. **Weights are sliceable**: under the same seed, the `size=N` initial weight must equal the corresponding slice of the `size=1` weight (concatenate the N shards in `rank` order and compare);
3. **Parameter census**: print the three bucket lengths; `len(partial) + len(complete) + len(sharded)` must equal `len(params)`;
4. **Norm consistency**: the global norm at `tp=1` and `tp=N` should match; a gap of `√tp` means `complete` is under- or over-tagged.

## Core mechanisms

### Orthogonal grouping

Given per-axis sizes and a boolean `mask` (which axes belong to this group), a rank is encoded in the Megatron layout as `rank = Σ coord_i · prefix_prod(sizes)[i]`. Ranks that agree on every unselected axis fall into the same group, so any two axes are orthogonal by construction and no per-combination grouping code is needed.

Three measured configurations at `world=8`:

| Configuration | `tp` groups | `cp` groups | `dp` groups |
|---------------|-------------|-------------|-------------|
| `tp=2, cp=2, pp=1` | `[0,1] [2,3] [4,5] [6,7]` | `[0,2] [1,3] [4,6] [5,7]` | `[0,4] [1,5] [2,6] [3,7]` |
| `tp=4, cp=2, pp=1` | `[0,1,2,3] [4,5,6,7]` | `[0,4] [1,5] [2,6] [3,7]` | one rank per group (`dp=1`) |
| `tp=2, cp=2, pp=2` | `[0,1] [2,3] [4,5] [6,7]` | `[0,2] [1,3] [4,6] [5,7]` | one rank per group (`dp=1`) |

In all three, the groups form an exact partition of the world: no rank is dropped and none appears twice.

### Primitives and their backward passes

The primitives only run forward; the direction is supplied by the wrapper's `backward`. This table is the part of the library that deserves a line-by-line review:

| Wrapper | forward | backward | Used by |
|---------|---------|----------|---------|
| `CopyToAll` | identity | `all_reduce` | column layer entry (no SP) |
| `Reduce` | `all_reduce` | identity | row layer exit (no SP) |
| `Gather(dim, REDUCE)` | `all_gather` | `reduce_scatter` | column layer entry (SP) |
| `Gather(dim, SLICE)` | `all_gather` | `take_shard` (no communication) | column layer exit gather |
| `ReduceScatter(dim)` | `reduce_scatter` | `all_gather` | row layer exit (SP) |
| `Shard(dim)` | `take_shard` (no communication) | `all_gather` | row layer entry |

What separates `GatherMode.REDUCE` from `SLICE` is what the consumer holds: a shard or a replica. With a shard, the backward pass must `reduce_scatter` the partial sum away; with a replica, every rank already holds the complete gradient and the backward pass only slices. Swap the two and the result is silently off by `1/tp` or `×tp` with identical shapes, which is why the direction is an explicit enum rather than a boolean flag.

### The three gradient buckets

Whether a parameter is sharded and whether its gradient is a partial sum are two different questions, and conflating them is the root of the `size`-fold errors:

| Bucket | Parameter | Local gradient | Needs `all_reduce` | Needs norm reduction |
|--------|-----------|----------------|--------------------|----------------------|
| `sharded` | `1/size` of the global weight | already global | no | yes |
| `complete` | replica | already global | no | no |
| `partial` | replica | partial sum | yes | no |

Classification uses tags on the parameter, not its name: `mark_sharded` means "this parameter holds one shard", `mark_grad_complete` means "replica with a complete gradient", and anything untagged falls back to `partial`. The default is the strict direction — an untagged parameter costs one extra reduction instead of a missing one.

### Dtype-fused reduction

The `partial` bucket is grouped by dtype, each group is `cat`-ed into one flat buffer, one `all_reduce` is issued per bucket, and the results are sliced back by the recorded `offsets`. Grouping by dtype is mandatory: concatenating mixed dtypes triggers an implicit promotion (`bf16` → `fp32`), which puts part of the gradients through an extra cast, and the volume is then counted at the higher precision as well.

Fusing does not change the numbers: `all_reduce` is element-wise, and every element inside the buffer sees the same summation sequence it would see in a standalone collective, so the result is bit-identical to per-parameter reduction. The cost is memory — the gradients of that dtype briefly take close to twice their footprint, plus one `cat` copy — and that is the main reason `fuse` is off by default.

## Scope: what the library leaves to the host

The following items are deliberately out of scope and belong to the training project:

| Item | Owner | Why |
|------|-------|-----|
| Model surgery (which layer goes column-parallel, which row-parallel) | Host | Depends on model semantics; the library cannot infer it |
| The attention side of CP (exchanging K/V across ranks) | Host | Ring-attention shapes vary with the attention variant; abstracting it would constrain implementations |
| Optimizer, mixed precision, clipping policy | Host | Orthogonal to parallelism |
| Registering hand-built layers as `partial` | Host | Pass them through `build_plan(..., extra_partial=...)`; the library does not guess |
| `num_groups` for grouped splits | Host | Required when splitting by attention head, for instance; otherwise a shard can straddle two heads |

The `num_groups` check is `num_groups % group.size == 0`. Do not substitute "the axis length is divisible by the group size": with an axis of 48, 6 logical groups and 4 shards the division holds, yet each shard straddles 1.5 groups and the resulting weights no longer line up with the logical groups.

## Known limitations

- CP currently provides the process group and inter-layer activation sharding only; ring-attention is not included;
- the top-level `__init__.py` does not export the functional primitives yet, so `all_reduce` / `all_gather` / `take_shard` / `reduce_scatter` must be imported from `distributed.comm` and `tag_of` from `distributed.layers`;
- on the single-device path each parallel layer holds a native `Linear` sub-cell, so parameter keys gain a `layer.` prefix and loading an existing checkpoint needs a key mapping;
- arbitrary-axis `all_gather` / `reduce_scatter` depend on `mint.movedim`; on versions without it, fall back to first-axis and last-axis support only;
- the parameter tag table holds strong references and never cleans up, so a long process that builds models repeatedly must release them explicitly.

## License

This project is released under the **Apache License 2.0**.

Key terms:

- **Grant**: free use, modification, distribution and commercialisation, with no additional authorisation required
- **Attribution**: the original copyright notice and license text must be retained on redistribution
- **Patents**: the license covers patent rights granted by contributors
- **Warranty**: the software is provided "as is", without warranties of any kind, express or implied
- **Liability**: authors and contributors are not liable for any damages arising from use of the software

The full text is available in the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) referenced in the source file headers.
