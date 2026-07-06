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
GNN_BucketDataset/
├── configs/
│   └── config.yaml            # Configuration for train/test/infer datasets and model parameters
├── src/
│   ├── __init__.py            # Package initialization, exporting core classes and functions
│   ├── BucketDatasetGenerator.py  # Bucket dataset iterator, implementing dynamic bucket batch generation
│   ├── DataGenerator.py       # Data generator, responsible for per-sample reading and feature reorganization
│   └── tools.py               # Utility functions: YAML loading, NPY directory loading
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
| `src/tools.py` | Provides `load_yaml` (YAML config parsing) and `load_npy` (NPY directory batch loading) utility functions |
| `configs/config.yaml` | Unified YAML configuration file covering model parameters and train/test/infer dataset parameters |

## Environment Requirements

### Hardware

| Hardware | Description |
|----------|-------------|
| Ascend processor | Recommended Ascend 910 series for distributed training |
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

Ascend environment example:

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
    train:
        role: train
        path: "/home/data/train.h5"
        batch_size: 8
        bucket_boundaries: [10, 20, 30]   # Bucket upper boundary list
        bucket_batch_size: [16, 8, 4]     # Batch size per bucket
        sample_num_by_node: {5: 20, 11: 50, 15: 50, 25: 80}  # Node count to sample count mapping
        padding_indices: [0, 1, 3, 5]     # Feature indices requiring padding alignment
```

## Usage

### Training Mode (Bucket Batching)

```bash
python main.py
```

The program reads `configs/config.yaml` and constructs a bucketed training dataset. Under training mode, `BucketDatasetGenerator` automatically:

1. Assigns each sample to the corresponding bucket based on node count
2. Draws data from buckets that meet batch conditions according to the random selection list
3. Zero-pads features that require alignment to the maximum node count within the bucket
4. Applies cross-sample offsets to edge indices to avoid collision
5. Recovers remaining data after iteration completes

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

## License

This project is licensed under the **Apache License 2.0**.

Key provisions:

- **Authorization**: Free to use, modify, distribute, and commercialize without additional permission
- **Attribution**: Original copyright notices and license text must be retained when distributing
- **Patent**: The license covers patent rights granted by contributors
- **No Warranty**: The software is provided "as is" without any express or implied warranties
- **Limitation of Liability**: Authors/contributors are not liable for any losses caused by the use of this software

For the full license text, see [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) as referenced in the source file headers.
