
# 模型介绍

GROVER将消息传递网络与Transformer式的架构结合起来，提供了一类具有高度表现力的分子编码器。GROVER的灵活性使它可以在不需要任何监督的情况下对大规模的分子数据进行有效的训练。然后，将预训练的GROVER模型用于下游的分子特性预测任务，并进行特定任务的微调。

# 目录结构说明

```text
.
└─grover
  ├─README.md
  ├─scripts
    ├─run_distribute_train.sh         # 多卡训练脚本
    ├─run_eval.sh                     # 评估脚本
    ├─run_infer_310.sh                # 310推理脚本
    ├─run_standalone_train.sh         # 单卡训练脚本
  ├─src
    ├─data
      ├─dataset.py                    # 数据集处理
      ├─molfeaturegenerator.py        # 特征生成工具
      ├─molgraph.py                   # 分子图表示数据
      ├─scaler.py                     # 归一化操作
      ├─task_labels.py                # 标签提取工具
      ├─mindsporevocab.py             # 建立词汇表工具
      ├─transforms.py                 # 特征生成器
    ├─model
      ├─layers.py                     # grove网络组件
      ├─models.py                     # grover网络
    ├─model_utils
      ├─config.py
      ├─device_adapter.py
      ├─local_adapter.py
      └─moxing_adapter.py
    ├─util
      ├─logger.py#                    # 日志函数
      ├─nn_utils.py                   # 模型相关工具函数
      ├─scheduler.py                  # 生成学习率
      └─utils.py                      # 工具函数
    └─__init__.py                     # python初始化文件
  ├─export.py                         # 导出网络结构
  ├─preprocess.py                     # 预处理数据，用于310推理
  ├─preprocess.py                     # 预处理数据，用于310推理
  ├─build_vocab.py                    # 建立原子/键的词汇表的脚本
  ├─save_features.py                  # 分子特征提取、语义标签提取的脚本
  ├─split_data.py                     # 数据集切分的脚本
  ├─grover_config.yaml                # 配置文件
  ├─generate_fingerprints.py          # 评估网络
  ├─eval.py                           # 评估网络
  ├─pretrain.py                       # 预训练网络
  └─train.py                          # 训练网络
```

# 数据集和可用模型

 数据集和ckpt地址: https://openi.pcl.ac.cn/dangwv/grover_local

- 预训练模型：`convert_grover_base.ckpt`

- 数据集：`exampledata`

  ```text
  .
  └─exampledata
    ├─pretune                           # 预训练数据集目录
      └─tryout.csv
    ├─finetune                          # 下游数据集目录
      ├─bbbp.csv                        # .csv文件为smiles分子式和对应标签的文件
      ├─clintox.csv
      ├─bace.csv
      ├─tox21.csv
      ├─toxcast.csv
      ├─freesolv.csv
      ├─esoll.csv
      ├─lipo.csv
      ├─qm7.csv
      └─qm8.csv
  ```

- 数据集说明：

  The classification datasets: BBBP | SIDER | ClinTox | BACE | Tox21 | ToxCast

  The regression datasets: FreeSolv | ESOL | Lipo | QM7 | QM8

  Metrics:

    For `QM7` and `QM8` datasets, you need to set `metric` as  `mae` .  
    For classification tasks, you need to set `metric` as `auc`.  
    For regression tasks, you need to set `metric` as  `rmse` .

# 环境配置

mindspore-1.8.1  
descriptastorus-2.6.0  
scipy-1.7.2  
pandas  
tensorboard  
scikit-learn  
rdkit  
tqdm  
pyyaml

# 脚本说明

## 预训练过程

### 数据预处理

- 示例数据集

  无标签分子数据：`./exampledata/pretrain/tryout.csv`.

- 数据集切分

  ```bash
  # 训练集，验证集默认以0.8: 0.2比例划分  
  python split_data.py --data_dir ./exampledata/pretrain --file_name tryout
  ```

  输出：在`./exampledata/pretrain`的目录下，生成两个文件 `tryout_train.csv` 和 `tryout_val.csv`

- 分子特征提取

  给定一个标记的分子数据集，可以提取附加的分子特征，以便从现有的预训练模型中训练模型。特征矩阵存储为' .npz '  

  ```bash
  python save_features.py --data_path ./exampledata/pretrain/tryout_train.csv  \
                          --save_path ./exampledata/pretrain/tryout_train.npz   \
                          --features_generator fgtasklabel \
                          --restart
  ```

  输出：在`./exampledata/pretrain`的目录下，生成一个文件 `tryout_train.npz`

- 建立词汇表

  ```bash
  python build_vocab.py --data_path ./exampledata/pretrain/tryout.csv  \
                        --vocab_save_folder ./exampledata/pretrain  \
                        --dataset_name tryout
  ```

  输出：在`./exampledata/pretrain`的目录下，生成原子和键的词汇表`tryout_atom_vocab.pkl`、`tryout_bond_vocab.pkl`，用于后续的预训练。

### 预训练

- 启动方式

  ```bash
  # Ascend场景下训练建议开启混合精度，即--mixed=True
  python pretrain.py --data_path_pretrain ./exampledata/pretrain \
                     --data_file_pretrain tryout \
                     --atom_vocab_path ./exampledata/pretrain/tryout_atom_vocab.pkl \
                     --bond_vocab_path ./exampledata/pretrain/tryout_bond_vocab.pkl \
                     --save_dir ./ckpt \
                     --num_attn_head 4 \
                     --hidden_size 800 \
                     --batch_size 32 \
                     --epochs 100 \
                     --mixed True

  选项：
  --data_path_pretrain        预训练数据所在目录
  --data_file_pretrain        预训练数据集名称
  --atom_vocab_path           原子词汇表路径
  --bond_vocab_path           化学键词汇表路径
  --save_dir                  保存训练过程生成的模型的目录
  --num_attn_head             注意力头的数目
  --hidden_size               隐藏层大小
  --batch_size                一个batch的大小
  --epochs                    训练轮数
  --mixed                     是否开启混合精度
  ```

  输出：每次训练生成的文件以独立文件夹的形式保存在`./ckpt`目录下

## 训练过程

### 数据预处理

- 数据集切分

  ```bash
  python split_data.py --data_dir exampledata/finetune --file_name bbbp
  ```

  输出：在`exampledata/finetune`的目录下，生成两个文件 `bbbp_train.csv` 和 `bbbp_val.csv`

- 分子特征提取

  ```bash
  python save_features.py --data_path exampledata/finetune/bbbp_train.csv  \
                          --save_path exampledata/finetune/bbbp_train.npz   \
                          --features_generator rdkit_2d_normalized \
                          --restart
  ```

  输出：在`exampledata/finetune`的目录下，生成一个文件 `bbbp_train.npz`

### 训练

需要设置配置文件`grover_config.yaml`中的数据集路径data_path_finetune、数据集名称data_file_finetune、任务类型dataset_type、预训练模型的路径resume_grover

- 以bbbp（分类）数据集为例

  ```bash
  # Ascend场景下训练建议开启混合精度，即--mixed=True
  python train.py --data_path_finetune ./exampledata/finetune \
                  --data_file_finetune bbbp \
                  --dataset_type classification \
                  --resume_grover ./convert_grover_base.ckpt \
                  --save_dir ./ckpt \
                  --num_attn_head 4 \
                  --hidden_size 800 \
                  --ffn_hidden_size 200 \
                  --batch_size 32 \
                  --epochs 100 \
                  --mixed True
  ```

- 以freesolv（回归）数据集为例

  ```bash
  # Ascend场景下训练建议开启混合精度，即--mixed=True
  python train.py --data_path_finetune ./exampledata/finetune \
                  --data_file_finetune freesolv \
                  --dataset_type regression \
                  --resume_grover ./convert_grover_base.ckpt \
                  --save_dir ./ckpt \
                  --num_attn_head 4 \
                  --hidden_size 800 \
                  --ffn_hidden_size 200 \
                  --batch_size 32 \
                  --epochs 100 \
                  --mixed True

  选项：
  --data_path_finetune        训练数据所在目录
  --data_file_finetune        训练数据集名称
  --dataset_type              数据任务类型
  --resume_grover             预训练模型路径
  --save_dir                  保存训练过程生成的模型的目录，以及保存回归任务生成归一化数据文件的目录
  --num_attn_head             注意力头的数目
  --hidden_size               隐藏层大小
  --ffn_hidden_size           前馈层大小
  --batch_size                一个batch的大小
  --epochs                    训练轮数
  --mixed                     是否开启混合精度
  ```

  输出：每次训练生成的文件以独立文件夹的形式保存在`./ckpt`目录下，以及回归任务训练生成的归一化文件也保存在`./ckpt`目录下

- bash脚本启动

  单卡训练

  ```bash
  bash run_standalone_train.sh [DEVICE_TARGET] [DEVICE_ID] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [SAVE_DIR]
  # example: bash run_standalone_train.sh Ascend 1 ../exampledata/finetune bbbp classification ../convert_grover_base.ckpt ../ckpt
  ```

  多卡训练

  ```bash
  bash run_distribute_train.sh [DEVICE_TARGET] [DEVICE_NUM] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [SAVE_DIR] [RANK_TABLE_FILE] [START_DEVICE_ID]
  # example: bash run_distribute_train.sh Ascend 2 ../exampledata/finetune bbbp classification ../convert_grover_base.ckpt ../ckpt hccl_2p.json
  ```

## 评估过程

### 数据预处理

- 分子特征提取

  ```bash
  python save_features.py --data_path ./exampledata/finetune/bbbp_val.csv  \
                          --save_path ./exampledata/finetune/bbbp_val.npz   \
                          --features_generator rdkit_2d_normalized \
                          --restart
  ```

  输出：在`exampledata/finetune`的目录下，生成一个文件 `bbbp_val.npz`

### 评估

需要设置配置文件`grover_config.yaml`中的数据集路径data_path_eval、数据集名称data_file_eval、任务类型dataset_type、评估指标metrics、预训练模型存储的路径pretrained

- 以bbbp(分类)数据集为例

  ```bash
  python eval.py --data_path_eval ./exampledata/finetune \
                 --data_file_eval bbbp \
                 --dataset_type classification \
                 --pretrained ./ckpt/bbbp/grover_100.ckpt \
                 --save_dir ./ckpt \
                 --eval_dir ./outputs \
                 --metrics auc
  ```

- 以freesolv(回归)数据集为例

  ```bash
  # 对于回归数据集还要加载训练过程中的归一化文件, 字段为--save_dir
  python eval.py --data_path_eval ./exampledata/finetune \
                 --data_file_eval freesolv \
                 --dataset_type regression \
                 --pretrained ./ckpt/freesolv/grover_100.ckpt \
                 --save_dir ./ckpt \
                 --eval_dir ./outputs \
                 --metrics rmse
  ```

  输出：评估生成的文件保存在 `./outputs`目录下

- bash脚本启动

  ```bash
  bash run_eval.sh [DEVICE_TARGET] [DEVICE_ID] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [EVAL_DIR] [METRICS] [SCALER_DIR]
  # example: bash run_eval.sh Ascend 0 ../exampledata/finetune bbbp classification ../ckpt/bbbp/grover_100.ckpt ../outputs auc ../ckpt
  ```

## 推理过程

- 导出模型

  ```bash
  python export.py --pretrained ./ckpt/bbbp/grover_100.ckpt \
                   --data_file_eval bbbp \
                   --dataset_type classification \
                   --batch_size 32 \
                   --file_format MINDIR
  ```

- 推理

  ```bash
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID] [DATASET_TYPE] [METRICS] [BATCHSIZE]
  # example: bash run_infer_310.sh ../GROVERbbbp.mindir bbbp ../exampledata/finetune y 2 classification auc 32
  ```

## 分子指纹生成过程

- 启动方式

  需要设置配置文件`grover_config.yaml`中的数据集路径data_path_fp、数据集名称data_file_fp

  ```bash
  # 以bbbp数据集为例
  python generate_fingerprints.py --data_path_fp ./exampledata/finetune \
                                  --data_file_fp bbbp \
                                  --resume_grover ./convert_grover_base.ckpt \
                                  --save_dir ./ckpt
  ```

# 参考

[1] Rong Y, Bian Y, Xu T, et al. Self-supervised graph transformer on large-scale molecular data[J]. Advances in Neural Information Processing Systems, 2020, 33: 12559-12571.
[2] https://github.com/tencent-ailab/grover