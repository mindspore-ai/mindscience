# Bucket Dataset Generator

## Background

In graph neural network (GNN) training scenarios, the graph scale (number of nodes) of different samples often varies significantly. If a fixed batch size is used for training, small-graph samples are over-padded to align with large graphs, resulting in wasted computational resources and redundant memory usage. To address this issue, this project provides a dynamic bucket-based dataset generation tool. The core approach is as follows:

- **Bucketing mechanism**: Samples are divided into multiple bucket intervals by node count, each with its own batch size. Large-graph samples are processed in smaller batches while small-graph samples use larger batches, balancing padding overhead against computational efficiency.
- **Dynamic iteration**: During training, a random selection list controls which bucket to draw data from at each iteration step, ensuring even consumption across all buckets. Remaining data is automatically processed after iteration completes.
- **MindSpore integration**: Built on the MindSpore `GeneratorDataset` interface, supporting distributed multi-card parallel data loading and sharding, seamlessly integrating with MindSpore training workflows.

This project supports three task modes: **train**, **test**, and **infer**, and provides loading capabilities for both HDF5 and NPY data formats.
Based on actual requirements, bucketing is currently provided only for the training mode, while test/infer modes use conventional fixed batch_size for comparison.

## Project Structure

```
GNN_BucketDatasetOnlinePadding/
├── configs/
│   └── config.yaml            # Configuration for train/test/infer datasets and model parameters
├── src/
│   ├── __init__.py            # Package initialization, exporting core classes and functions
│   ├── BucketDatasetGenerator.py  # Bucket dataset iterator, implementing dynamic bucket batch generation
│   ├── padding.py             # Batch-level alignment: declarative column tables + pad/concat/edge ownership
│   ├── DataGenerator.py       # Data generator, responsible for per-sample reading and feature reorganization
│   └── tools.py               # Utility functions: YAML loading, NPY directory loading
├── tests/
│   ├── test_padding.py               # Unit tests for the padding module (pure numpy)
│   ├── test_package_data.py          # Unit tests for batch packaging (_package_data) and schema checks
│   ├── test_bucket_state_machine.py  # Unit tests for the bucket iteration state machine (stub dataset, pure numpy)
│   └── test_tools.py                 # Unit tests for load_npy/load_yaml (requires pyyaml)
├── main.py                    # Main entry: builds MindSpore bucket dataset pipeline for training integration
├── requirements.txt           # Python dependencies and version declarations
├── README_CN.md               # Chinese documentation
└── README.md                  # English documentation
```

Module responsibilities:

| Module | Description |
|--------|-------------|
| `main.py` | Program entry point; parses configuration and invokes the appropriate dataset construction flow based on task type (train/test/infer) |
| `src/DataGenerator.py` | `DataGenerator` class; reads node/edge features per sample from HDF5 or NPY data sources and performs corresponding processing |
| `src/BucketDatasetGenerator.py` | `BucketDatasetGenerator` class; buckets by node count, dynamically pads and concatenates batches, manages remaining data recovery |
| `src/padding.py` | The single implementation of batch-level alignment: `NODE_FILLS`/`EDGE_ATTRS` declare which columns sit on the node/edge axis, `pad_nodes`/`edge_batch_ids` do the alignment |
| `src/tools.py` | Provides `load_yaml` (YAML config parsing) and `load_npy` (NPY directory batch loading) utility functions |
| `configs/config.yaml` | Unified YAML configuration file covering model parameters and train/test/infer dataset parameters |
| `tests/` | Unit tests for padding, batch packaging, the bucket iteration state machine, and `tools`; see [Unit Tests](#unit-tests) |

## Environment Requirements

### Hardware

| Hardware | Description |
|----------|-------------|
| Huawei Ascend processor | Recommended Huawei Ascend 910 series for distributed training |
| CPU | Can be used for single-card testing and inference |

### Software

| Software | Version Requirement |
|----------|---------------------|
| Python | >= 3.9 |
| MindSpore | >= 2.7.0 |
| NumPy | >= 1.26.0 |
| h5py | >= 3.7.0 |
| PyYAML | >= 6.0 |

## Installation and Configuration

### 1. Install MindSpore

Select the appropriate MindSpore installation method for your target hardware platform. See the [MindSpore Official Installation Guide](https://www.mindspore.cn/install) for details.

Huawei Ascend environment example:

```bash
pip install mindspore==2.7.0
```

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy>=1.26.0 h5py>=3.7.0 pyyaml>=6.0
```

### 3. Prepare Data Files

Place train/test/infer data files at the specified paths. The following formats are supported:

- **HDF5 files** (`.h5`): A single file containing all feature keys (`x0`, `node_feature1` ~ `node_feature3`, `edge_feature1`, `edge_feature2`, `x_vd`)
- **NPY directory**: Each `.npy` file in the directory corresponds to a feature key, with the filename serving as the key name

Feature key names are customizable.

Modify the `path` field for each dataset in `configs/config.yaml` to point to the actual data location.

### 4. Configure Parameters

Edit `configs/config.yaml` and adjust parameters as needed:

```yaml
model:
    x_features: 7          # Number of input feature dimensions
    output: 3               # Number of output feature dimensions

data:
    use_mock_data: True     # Build from in-memory random data instead of reading `path`; set to False for real data
    train:
        role: train
        path: "/home/data/train.h5"
        batch_size: 8
        bucket_boundaries: [10, 20, 30]   # Bucket upper boundary list
        bucket_batch_size: [16, 8, 4]     # Batch size per bucket
        sample_num_by_node: {5: 20, 11: 50, 15: 50, 25: 80}  # Node count to sample count mapping
        pad_quantum: 1                    # Round the pad target up to a multiple of this value; 1 pads to the exact batch maximum
```

## Usage

### Training Mode (Bucket Batching)

```bash
python main.py
```

The program reads `configs/config.yaml` and constructs a bucketed training dataset. Under training mode, `BucketDatasetGenerator` automatically:

1. Assigns each sample to the corresponding bucket based on node count
2. Draws data from buckets that meet batch conditions according to the random selection list
3. Pads node-level columns (`x0`, `node_fea1~3`, `labels`) to the batch maximum, so sample `i` always owns the slot starting at row `i * max_node`
4. Concatenates edge-level columns (`edge_fea1`, `edge_fea2`, `mean_edge`) as they are, and records which sample each edge belongs to in `batch_id`
5. Recovers remaining data after iteration completes

Which columns are node-level and which are edge-level is declared by the `NODE_FILLS`
and `EDGE_ATTRS` tables in `src/padding.py`, so registering a column there is the only
step needed to batch it. A sample keeps its own node count until the batch is assembled:
`_package_data` absorbs the spread, so the input data does not have to be padded to the
bucket boundary in advance, and a `pad_quantum` above 1 trades a little memory for
fewer graph compilations.

Log example:
```
Start Load Dataset
Task: train
Dataset Bucket Boundary: [10, 20, 30], Batch Num(estimated): 41
Create bucket train dataset successfully
```

### Distributed Training

For multi-card parallel training, start the MindSpore distributed runtime environment first. The program will automatically call `get_rank()` and `get_group_size()` for data sharding.

### Test / Inference Mode

Modify the `role` field of the corresponding dataset in `configs/config.yaml` to `test` or `infer`, then:

```bash
python main.py
```

Test and inference modes use a fixed batch size, calling `test_collate` and `infer_collate` respectively for data concatenation and `edge_batch_id` construction.

## Unit Tests

Run the whole suite from the application directory:

```bash
python -m unittest discover -s tests
```

| Test file | Scope | Dependencies |
|-----------|-------|--------------|
| `tests/test_padding.py` | `pad_to_max` / `pad_nodes` / `edge_batch_ids` / `round_up` | numpy |
| `tests/test_package_data.py` | `_package_data` batch layout and column-schema validation | numpy |
| `tests/test_bucket_state_machine.py` | Bucket iteration: one-time sample consumption, per-bucket batch sizes, boundary validation, regression for batches exceeding the 1.15x estimate | numpy (runs against a stub dataset, no MindSpore) |
| `tests/test_tools.py` | `load_npy` key derivation, `load_yaml` | numpy + pyyaml |

## License

This project is licensed under the **Apache License 2.0**.

Key provisions:

- **Authorization**: Free to use, modify, distribute, and commercialize without additional permission
- **Attribution**: Original copyright notices and license text must be retained when distributing
- **Patent**: The license covers patent rights granted by contributors
- **No Warranty**: The software is provided "as is" without any express or implied warranties
- **Limitation of Liability**: Authors/contributors are not liable for any losses caused by the use of this software

For the full license text, see [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) as referenced in the source file headers.
