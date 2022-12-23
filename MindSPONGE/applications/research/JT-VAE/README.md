# JT-VAE

- [JT-VAE Description](#EAST-description)
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
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Inference Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [JT-VAE Description](#contents)

JT-VAE extends the variational autoencoder (Kingma & Welling, 2013) to molecular graphs by introducing a suitable encoder and a matching decoder. Deviating from previous work (G´omez-Bombarelli et al., 2016; Kusner et al., 2017), we interpret each molecule as having been built from subgraphs chosen out of a vocabulary of valid components. These components are used as building blocks both when encoding a molecule into a vector representation as well as when decoding latent vectors back into valid molecular graphs. This idea was proposed in the paper "Junction Tree Variational Autoencoder for Molecular Graph Generation.", published in 2018.

[Paper](http://proceedings.mlr.press/v80/jin18a/jin18a.pdf) Jin W, Barzilay R, Jaakkola T. Junction tree variational autoencoder for molecular graph generation[C]//International conference on machine learning. PMLR, 2018: 2323-2332.

# [Model architecture](#contents)

The overall generative approach, cast as a junction tree variational autoencoder, first generates a tree structured object (a junction tree) whose role is to represent the scaffold of subgraph components and their coarse relative arrangements. The components are valid chemical substructures automatically extracted from the training set using tree decomposition and are used as building blocks. In the second phase, the subgraphs (nodes in the tree) are assembled together into a coherent molecular graph.

# [Dataset](#contents)

Dataset used is ZINC where you can download from [here](https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc)

In this project, the file organization is recommended as below:

```shell
.
└─data
  ├─zinc
    ├─all.txt
    ├─test.txt
    ├─train.txt
    ├─valid.txt
    └─vocab.txt
```

# [Features](#contents)

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
.
└─JT-VAE
  ├─README.md                             # descriptions about kgcn
  ├─scripts
    ├─run_eval_gpu.sh                     # launch evaluating with GPU platform
    └─run_train_gpu.sh                    # launch standalone training with GPU platform
  ├─src
    ├─model_utils
      ├─config.py                         # parameter configuration
      ├─device_adapter.py                 # device adapter
      ├─local_adapter.py                  # local adapter
      └─moxing_adapter.py                 # moxing adapter
    ├─chemutils.py                        # chemutils function
    ├─datautils.py                        # datautils function
    ├─dataset.py                          # dataset function
    ├─jtmpn.py                            # jt-vae model
    ├─jtnn_dec.py                         # jt-vae model
    ├─jtnn_enc.py                         # jt-vae model
    ├─jtnn_vae.py                         # jt-vae model
    ├─jtprop_vae.py                       # jt-vae model
    ├─mol_tree.py                         # jt-vae model
    ├─mpn.py                              # jt-vae model
    ├─nnutils.py                          # nnutils
    └─utils.py                            # utils
  ├─default_config.yaml                   # parameter configuration
  ├─eval.py                               # evaluation script
  └─train.py                              # training script
```

## [Training process](#contents)

### Usage

- GPU:

```bash
# standalone training
bash run_train_gpu.sh [DEVICE_ID]
# example: bash run_train_gpu.sh 0
```

### Launch

```bash
# training example
    GPU:
      # standalone training
      bash run_train_gpu.sh [DEVICE_ID]
      # example: bash run_train_gpu.sh 0
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `save_ckpt_dir` by default, and training log  will be redirected to `./log`

```python
(1p)
...
epoch: 1 step: 1 , loss is 126.508942, time is 23 09:45:32
epoch: 1 step: 2 , loss is 150.534348, time is 23 09:46:00
epoch: 1 step: 3 , loss is 91.467941, time is 23 09:46:47
epoch: 1 step: 4 , loss is 116.076607, time is 23 09:47:14
epoch: 1 step: 5 , loss is 134.211349, time is 23 09:47:58
epoch: 1 step: 6 , loss is 121.681618, time is 23 09:48:48
epoch: 1 step: 7 , loss is 104.542259, time is 23 09:49:37
epoch: 1 step: 8 , loss is 96.500969, time is 23 09:50:29
epoch: 1 step: 9 , loss is 67.828621, time is 23 09:51:11
epoch: 1 step: 10 , loss is 95.291901, time is 23 09:52:23
epoch: 1 step: 11 , loss is 72.148476, time is 23 09:53:17
epoch: 1 step: 12 , loss is 102.087349, time is 23 09:54:18
epoch: 1 step: 13 , loss is 62.927967, time is 23 09:55:20
epoch: 1 step: 14 , loss is 56.725189, time is 23 09:56:17
epoch: 1 step: 15 , loss is 67.097397, time is 23 09:57:37
...
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:
（we also provide trained ckpt where you can download from [here](https://download.mindspore.cn/mindscience/mindsponge/JTVAE/checkpoint/model.ckpt)）

- GPU:

```bash
# eval example
  GPU:
    bash run_eval_gpu.sh [CKPT_PATH] [DEVICE_ID]
    # example: bash run_eval_gpu.sh "../ckpt/model.ckpt" 0
```

### Launch

- Modify the parameters in `eval.py` and run:

```bash
# eval example
  GPU:
    bash run_eval_gpu.sh [CKPT_PATH] [DEVICE_ID]
    # example: bash run_eval_gpu.sh "../ckpt/model.ckpt" 0
```

> checkpoint can be produced in training process.

### Result

Evaluation result will be stored in the output file of evaluation script, you can find result like the followings in `log`.

# [Model description](#contents)

## [Performance](#contents)

### Inference Performance

| Parameters          | GPU               |
| ------------------- | ----------------- |
| Model Version       | KGNN              |
| Resource            | GPU2080ti         |
| uploaded Date       | 12/23/2022        |
| MindSpore Version   | 1.9.0             |
| Dataset             | ZINC              |
| Batch_size          | 1                 |
| Accuracy            | 76.74%            |
| Model for inference | 22M (.ckpt file)  |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
