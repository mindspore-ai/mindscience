# KGNN for Ascend

- [KGNN Description](#EAST-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [KGNN Description](#contents)

KGNN can effectively capture drug and its potential neighborhoods by mining their associated relations in KG. To extract both high-order structures and semantic relations of the KG, KGNN learn from the neighborhoods for each entity in KG as their local receptive, and then integrate neighborhood information with bias from representation of the current entity. This way, the receptive field can be naturally extended to multiple hops away to model highorder topological information and to obtain drugs potential long-distance correlations. This idea was proposed in the paper "KGNN: Knowledge Graph Neural Network for Drug-Drug Interaction Prediction.", published in 2020.

[Paper](https://www.ijcai.org/Proceedings/2020/0380.pdf) Xuan Lin, Zhe Quan, Zhi-Jie Wang, Tengfei Ma, Xiangxiang Zeng College of Information Science and Engineering, Hunan University, College of Computer Science, Chongqing University, Published in IJCAI-2020.

# [Model architecture](#contents)

The network structure can be decomposed into three parts: DDI Extraction and KG Construction, KGNN Layer and Drug-drug Interaction Prediction.It takes the parsed DDI matrix and knowledge graph obtained from preprocessing (i.e., DDI extraction and KG construction) of dataset as the input. It outputs the interaction value for the drug-drug pair. Remind that the central idea of KGNN is to consider both high-order structure and semantic relation, by using graph neural network to encode the drug and its topological neighborhood information to a distributed representation.

# [Dataset](#contents)

Dataset used  [KEGG-drug](https://github.com/xzenglab/KGNN/tree/master/raw_data/kegg)

- Dataset: KEGG-drug:
    - Drugs: 1925
    - Interactions: 56983
    - Entities: 129910
    - Relation Types:  167
    - KG Triples:  362870

In this project, the file organization is recommended as below:

```shell
.
└─raw_data
  ├─kegg
    ├─approved_example.txt
    ├─entity2id.txt
    ├─relation2id.txt
    └─train2id.txt
```

# [Features](#contents)

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─kgnn
  ├─README.md                             # descriptions about kgcn
  ├─scripts
    ├─run_eval_ascend.sh                  # launch standalone training with GPU platform(1p)
    ├─run_eval_gpu.sh                     # launch evaluating with GPU platform
    ├─run_infer_310.sh                    # launch evaluating with ascend platform
    ├─run_standalone_train_ascend.sh      # launch standalone training with ascend platform(1p)
    └─run_train_gpu.sh                    # launch standalone training with GPU platform
  ├─src
    ├─model_utils
      ├─config.py                         # parameter configuration
      ├─device_adapter.py                 # device adapter
      ├─local_adapter.py                  # local adapter
      └─moxing_adapter.py                 # moxing adapter
    ├─aggregator.py                       # aggregator function
    ├─data_loader.py                      # data loader function
    ├─dataset.py                          # dataset function
    ├─kgcn.py                             # kgcn model
    ├─loss.py                             # loss
    └─utils.py                            # General components
  ├─default_config.yaml                   # parameter configuration
  ├─eval.py                               # evaluation script
  ├─export.py                             # export scripts
  ├─postprocess.py                        # postprocess script
  ├─preprocess.py                         # preprocess scripts
  └─train.py                              # training script
```

## [Training process](#contents)

### Usage

- Ascend:

```bash
# standalone training
bash run_standalone_train_ascend.sh [RAW_DATA_DIR] [DATASET] [DEVICE_ID]
# example: bash run_standalone_train_ascend.sh ./raw_data/ kegg 0
```

- GPU:

```bash
# standalone training
bash run_train_gpu.sh [RAW_DATA_DIR] [DATASET] [DEVICE_ID]
# example: bash run_train_gpu.sh ./raw_data/ kegg 0
```

### Launch

```bash
# training example
    Ascend:
      # standalone training
      bash run_standalone_train_ascend.sh [RAW_DATA_DIR] [DATASET] [DEVICE_ID]
      # example: bash run_standalone_train_ascend.sh ./raw_data/ kegg 0
    GPU:
      # standalone training
      bash run_train_gpu.sh [RAW_DATA_DIR] [DATASET] [DEVICE_ID]
      # example: bash run_train_gpu.sh ./raw_data/ kegg 0
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `ckpt_url` by default, and training log  will be redirected to `./log`

```python
(1p)
...
Train epoch time: 21830.078 ms, per step time: 496.138 ms
epoch: 19 step: 1, loss is 0.09186707437038422
epoch: 19 step: 2, loss is 0.09054221957921982
epoch: 19 step: 3, loss is 0.0800129845738411
epoch: 19 step: 4, loss is 0.07796357572078705
epoch: 19 step: 5, loss is 0.07740001380443573
epoch: 19 step: 6, loss is 0.08665880560874939
epoch: 19 step: 7, loss is 0.09129385650157928
epoch: 19 step: 8, loss is 0.08729872107505798
epoch: 19 step: 9, loss is 0.08645150810480118
epoch: 19 step: 10, loss is 0.09781090915203094
epoch: 19 step: 11, loss is 0.10565177351236343
epoch: 19 step: 12, loss is 0.08685533702373505
epoch: 19 step: 13, loss is 0.08123263716697693
epoch: 19 step: 14, loss is 0.08262551575899124
epoch: 19 step: 15, loss is 0.08548043668270111
...
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- Ascend:

```bash
# eval example
  Ascend:
    bash run_eval_ascend.sh [CKPT_PATH] [DEVICE_ID]
    # example: bash run_eval_ascend.sh "./KGNN2.0/ckpt/ckpt_0/KGNN-34_44.ckpt" 0
```

### Launch

- Modify the parameters in `eval.py` and run:

```bash
# eval example
  GPU:
    bash run_eval_gpu.sh [CKPT_PATH] [DEVICE_ID]
    # example: bash run_eval_gpu.sh "./KGNN2.0/ckpt/ckpt_0/KGNN-34_44.ckpt" 0
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the output file of evaluation script, you can find result like the followings in `log`.

```python
Logging Info - test_auc: 0.9422428483164229, test_acc: 0.8876509232954546, test_f1: 0.8932487731478626, test_aupr: 0.9188073906877906
```

## Inference Process

### [Export MindIR](#contents)

```shell
python export.py --checkpoint_path=[checkpoint_path]
```

The `checkpoint_path` parameter is required,

### Infer on Ascend310

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET_NAME` must be in ['kegg', 'drugbank'].
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### Result

Inference result is saved in current path, you can find result like this in acc.log file.

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters          | Ascend                                 | GPU                                    |
| ------------------- | -------------------------------------- | -------------------------------------- |
| Model Version       | KGNN                                   | KGNN                                   |
| Resource            | Ascend910                              | GPU3090                                |
| uploaded Date       | 10/28/2022                             | 10/28/2022                             |
| MindSpore Version   | 1.8.0                                  | 1.8.0                                  |
| Dataset             | KEGG-drug                              | KEGG-drug                              |
| Batch_size          | 2048                                   | 2048                                   |
| Training Parameters | epoch: 20, batch_size: 2048, lr=0.0005 | epoch: 20, batch_size: 2048, lr=0.0005 |
| Optimizer           | Adam                                   | Adam                                   |
| Loss Function       | binary cross-entropy                   | binary cross-entropy                   |
| Scripts             | kgnn scripts                           | kgnn scripts                           |

#### Inference Performance

| Parameters          | Ascend            | GPU               |
| ------------------- | ----------------- | ----------------- |
| Model Version       | KGNN              | KGNN              |
| Resource            | Ascend910         | GPU3090           |
| uploaded Date       | 10/28/2022        | 10/28/2022        |
| MindSpore Version   | 1.8.0             | 1.8.0             |
| Dataset             | KEGG-drug         | KEGG-drug         |
| Batch_size          | 2048              | 2048              |
| Accuracy            | 88.32%            | 88.90%            |
| Model for inference | 127M (.ckpt file) | 127M (.ckpt file) |

# [Description of Random Situation](#contents)

There are two random situations:

- Seed is set in data_loader.py. It is for neighbor random sampling.
- Seed is set in data_loader.py. It is used to partition dataset.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
