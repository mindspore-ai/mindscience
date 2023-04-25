# **Contents**

- [**Contents**](#contents)
- [**Compression of Point Cloud Data**](#compression-of-point-cloud-data)
    - [**Self-supervised Training of the Compression Model**](#self-supervised-training-of-the-compression-model)
    - [**Block-wise Data Compression**](#block-wise-data-compression)
- [**Dataset**](#dataset)
- [**Requirements**](#requirements)
- [**Script Description**](#script-description)
    - [**Script and Code Sample**](#script-and-code-sample)
    - [**Parameters**](#parameters)
    - [**Self-supervised Training of the Compression Model**](#self-supervised-training-of-the-compression-model-1)
        - [**Usage**](#usage)
    - [**Block-wise Data Compression**](#block-wise-data-compression-1)
        - [**Usage**](#usage-1)
- [**Random Seed Setting**](#random-seed-setting)
- [**MindScience Home Page**](#mindscience-home-page)

# **Compression of Point Cloud Data**

When you use point cloud data to calculate scatter parameters, if the target architecture is too complex or the modification of structure is too fine, the resolution of the point cloud data needs to be set to a high value to ensure the validity of the point cloud data. However, a high resolution may lead to the problem that a single piece of data is too large. For example, a single point cloud data in this scenario consists of hundreds of millions of points. The deep learning based simulation method requires a large amount of memory and computing power to process such data, and its efficiency is also reduced.

To solve this problem, MindSpore Elec proposes a neural network-based compression model to compress the original point cloud data by blocks. This tool can greatly reduce the storage and computation cost of point cloud based AI simulation method, improve its generality and efficiency.

## **Self-supervised Training of the Compression Model**

- Randomly extract 25 x 50 x 25 data blocks from the point cloud data as the training dataset.
- Build compression-reconstruction model based on AutoEncoder, and train the model to minimize the reconstruction error.
- Save the encoder checkpoint.

## **Block-wise Data Compression**

- Split point cloud data into 25 x 50 x 25 blocks and compresses them block-wisely.
- Arrange the compressed data according to the original space positions before the block splitting to form compressed data.

# **Dataset**

Use the `generate_data` function in `src/dataset.py` to automatically obtain 25 x 50 x 25 block data from the original point cloud data for training or testing.

The model training data in this example involves commercial secrets and the download address cannot be provided. The developer can use `src/sampling.py` to generate pseudo data for functional verification.

# **Requirements**

- Hardware (Ascend)
    - Prepare the Ascend AI Processor to set up the hardware environment.
- Framework
    - [MindSpore Elec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- For more information, see the following resources:
    - [MindSpore Elec Tutorial](https://www.mindspore.cn/mindelec/docs/en/master/intro_and_install.html)
    - [MindSpore Elec Python API](https://www.mindspore.cn/mindelec/docs/en/master/mindelec.architecture.html)

# **Script Description**

## **Script and Code Sample**

```path
.
└─auto_encoder
  ├─README.md
  ├─src
    ├─dataset.py                     # Preparing and importing Datasets
    ├─metric.py                      # Evaluation Indicator
    ├─model.py                       # Model
    ├─lr_generator.py                # Learning rate generation
    ├─sampling.py                    # pseudo data generation
  ├──train.py                        # Self-supervised training compression model
  ├──data_compress.py                # Block-wise data compression
```

## **Parameters**

Training parameters can be configured in the `train.py` file.

```python
'base_channels': 8,                       # basic feature numbers
'input_channels': 4,                      # input feature numbers
'epochs': 2000,                           # number of epochs
'batch_size': 128,                        # size of mini-batch
'save_epoch': 100,                        # interval for save checkpoints
'lr': 0.001,                              # basic learning rate
'lr_decay_milestones': 5,                 # number of learning rate decays
'eval_interval': 20,                      # evaluation interval
'patch_shape': [25, 50, 25],              # dimension size of block data
```

## **Self-supervised Training of the Compression Model**

### **Usage**

You can use the `train.py` script to train the compression model. During the training, checkpoints of the Encoder are automatically saved.

``` shell
python train.py --train_input_path TRAIN_INPUT_PATH
                --test_input_path TEST_INPUT_PATH
                --device_num 0
                --checkpoint_dir CKPT_PATH
```

## **Block-wise Data Compression**

### **Usage**

After the compression model training is complete, you can run the `data_compress.py` command to start block-wise data compression.

``` shell
python data_compress.py --input_path TEST_INPUT_PATH
                        --data_config_path DATA_CONFIG_PATH
                        --device_num 0
                        --model_path CKPT_PATH
                        --output_save_path OUTPUT_PATH
```

The compressed point cloud data can be found in the configured output directory.

# **Random Seed Setting**

The seed for the `create_dataset` function and the random seed in the `generate_data` random partition training test set are set in `dataset.py`.

# **MindScience Home Page**

Visit the official website [home page](https://gitee.com/mindspore/mindscience).
