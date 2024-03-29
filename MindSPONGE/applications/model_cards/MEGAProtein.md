# MEGAProtein

## 模型介绍

使用计算机高效计算获取蛋白质空间结构的过程被称为蛋白质结构预测，传统的结构预测工具一直存在精度不足的问题，直至2020年谷歌DeepMind团队提出[AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)<sup>[1,2]</sup>，该模型相较于传统工具预测精度大幅提升，所得结构与真实结构误差接近实验方法，但是仍存在数据前处理耗时过长、缺少MSA时预测精度不准、缺乏通用评估结构质量工具的问题。针对这些问题，高毅勤老师团队与MindSpore科学计算团队合作进行了一系列创新研究，开发出更准确和更高效的蛋白质结构预测工具**MEGA-Protein**。

MEGA-Protein主要由三部分组成：

- **蛋白质结构预测工具MEGA-Fold**，网络模型部分与AlphaFold2相同，在数据预处理的多序列对比环节采用了[MMseqs2](https://www.biorxiv.org/content/10.1101/2021.08.15.456425v1.full.pdf)<sup>[3]</sup>进行序列检索，相比于原版端到端速度提升2-3倍；同时借助内存复用大幅提升内存利用效率，同硬件条件下支持更长序列的推理，基于32GB内存的Ascend910运行时最长支持2048长度序列推理（以[Pipeline](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/src/mindsponge/pipeline/models/megafold)模式运行，可以支持3072长度序列推理）；我们还提供了结构预测模型训练能力，我们自己训练的权重获得了CAMEO-3D蛋白质结构预测赛道22年4月月榜第一。

<div align=center>
<img src="../../docs/megafold_contest.png" alt="MEGA-Fold获得CAMEO-3D蛋白质结构预测赛道月榜第一" width="600"/>
</div>

- **MSA生成工具MEGA-EvoGen**，能显著提升单序列的预测速度，并且能够在MSA较少（few shot）甚至没有MSA（zero-shot，即单序列）的情况下，帮助MEGA-Fold/AlphaFold2等模型维持甚至提高推理精度，突破了在「孤儿序列」、高异变序列和人造蛋白等MSA匮乏场景下无法做出准确预测的限制。基于32GB内存的Ascend910运行时最长支持768长度序列推理。该方法获得了CAMEO-3D蛋白质结构预测赛道22年7月月榜第一。

<div align=center>
<img src="../../docs/evogen_contest.jpg" alt="MEGA-EvoGen方法获得CAMEO-3D蛋白质结构预测赛道月榜第一" width="600"/>
</div>

- **蛋白质结构评分工具MEGA-Assessment**，该工具可以评价蛋白质结构每个残基的准确性以及残基-残基之间的距离误差，同时可以基于评价结果对蛋白结构作出进一步的优化。基于32GB内存的Ascend910运行时最长支持2048长度序列推理。该方法获得了CAMEO-QE结构质量评估赛道22年7月月榜第一。

<div align=center>
<img src="../../docs/assess_contest.png" alt="MEGA-Assessment方法获得CAMEO-QE结构质量评估赛道月榜第一" width="600"/>
</div>

## 数据集

MEGA-Fold训练数据集为[PSP蛋白质结构数据集](http://ftp.cbi.pku.edu.cn/psp/)，数据集大小为1.6TB，解压后为25TB。
MEGA-Assessment训练数据集为PSP数据集中的[PSP lite](http://ftp.cbi.pku.edu.cn/psp/psp_lite/)。

```shell
.
└─PSP
  ├─true_structure_dataset
  | ├─pkl
  | | └─256 pkl packages
  | ├─pdb
  | | └─256 pdb packages
  | └─true_structure_data_statistics_729.json
  ├─distillation_dataset
  | ├─pkl
  | | └─256 pkl packages
  | ├─pdb
  | | └─256 pdb packages
  | └─distill_data_statistics_729.json
  ├─new_validation_dataset
  | ├─pkl.tar.gz
  | ├─pdb.tar.gz
  | └─nv_data_statistics.json
  └─psp_lite
    ├─true_structure_mini
    | ├─pkl
    | | └─32 pkl packages
    | └─true_structure_mini.pdb.tar.gz
    └─distillation_mini
      ├─pkl
      | └─32 pkl packages
      └─distillation_mini.pdb.tar.gz
```

## 如何使用

mindsponge.PipeLine中分别提供了三个模型的推理流程，在使用时，

1. 可将氨基酸序列输入MEGA-EvoGen中获取该蛋白的共进化信息，也可以将*传统数据库检索*生成的共进化信息输入MEGA-EvoGen进行强化
2. 将共进化输入MEGA-Fold中进行蛋白质的结构预测
3. 最后将蛋白质共进化与结构信息共同输入MEGA-Assessment中进行打分评估

以CASP14蛋白质T1082-D1为例，整体推理流程如下所示。

*传统数据库检索请参考`application/common_utils/database_query/README.md`配置。*

```python
import numpy as np
import mindspore as ms
from mindsponge import PipeLine

ms.set_context(mode=ms.GRAPH_MODE)

# MEGA-EvoGen推理获取蛋白质生成MSA后的特征
fasta = "GYDKDLCEWSMTADQTEVETQIEADIMNIVKRDRPEMKAEVQKQLKSGGVMQYNYVLYCDKNFNNKNIIAEVVGE"
msa_generator = PipeLine(name="MEGAEvoGen")
msa_generator.set_device_id(0)
msa_generator.initialize(key="evogen_predict_256")
msa_generator.model.from_pretrained()
msa_feature = msa_generator.predict(fasta)

# MEGA-Fold推理获取蛋白质结构信息
fold_prediction = PipeLine(name="MEGAFold")
fold_prediction.set_device_id(0)
fold_prediction.initialize(key="predict_256")
fold_prediction.model.from_pretrained()
final_atom_positions, final_atom_mask, aatype, _, _ = fold_prediction.model.predict(msa_feature)

# MEGA-Assessment对蛋白质结构进行评价
protein_assessment = PipeLine(name = "MEGAAssessment")
protein_assessment.set_device_id(0)
protein_assessment.initialize("predict_256")
protein_assessment.model.from_pretrained()
msa_feature['decoy_aatype'] = np.pad(aatype, (0, 256 - aatype.shape[0]))
msa_feature['decoy_atom_positions'] = np.pad(final_atom_positions, ((0, 256 - final_atom_positions.shape[0]), (0, 0), (0, 0)))
msa_feature['decoy_atom_mask'] = np.pad(final_atom_mask, ((0, 256 - final_atom_mask.shape[0]), (0, 0)))

res = protein_assessment.predict(msa_feature)
print("score is:", np.mean(res))
```

### 使用场景

MEGAEvoGen，MEGAFold，MEGAAssessment均支持多种不同场景下的不同输入格式进行推理，详情如下：

为方便说明使用场景，默认下载好config文件，通过修改内置参数的方式选择不同场景，用户使用时也可按照如下方式执行，若未提前下载config文件，可通过替换样例内代码的方式下载的同时进行config的修改与加载。

- MEGAEvoGen

    - 序列作为输入，样例如下：

    ```python
    from mindsponge import PipeLine
    from mindsponge.common.config_load import load_config

    fasta = "GYDKDLCEWSMTADQTEVETQIEADIMNIVKRDRPEMKAEVQKQLKSGGVMQYNYVLYCDKNFNNKNIIAEVVGE"
    msa_generator = PipeLine(name="MEGAEvoGen")

    # 未获取config文件时，执行如下两行命令即可自动下载config文件，之后所有案例同理替换，仅提供代码样例，不做相同说明
    # from mindsponge.pipeline.pipeline import download_config
    # download_config(msa_generator.config["evogen_predict_256"], msa_generator.config_path + "evogen_predict_256.yaml")

    conf = load_config(msa_generator.config_path + "evogen_predict_256.yaml")
    conf.use_pkl = False
    msa_generator.initialize(conf=conf)
    msa_generator.model.from_pretrained()
    features = msa_generator.predict(fasta)

    with open("./examples/MEGA-Protein/pkl/T1082-D1.pkl", "rb") as f:
        data = pickle.load(f)
    for k, v in features:
        print(k, v.shape, v.dtype)
    ```

    - 序列搜索MSA后所获得的pickle文件作为输入，样例如下：

    ```python
    import pickle
    from mindsponge import PipeLine

    with open("./test.pkl", "rb") as f:
        data = pickle.load(f)
    msa_generator = PipeLine(name="MEGAEvoGen")

    # from mindsponge.pipeline.pipeline import download_config
    # download_config(msa_generator.config["evogen_predict_256"], msa_generator.config_path + "evogen_predict_256.yaml")

    conf = load_config(msa_generator.config_path + "evogen_predict_256.yaml")
    conf.use_pkl = True
    msa_generator.initialize(conf=conf)
    msa_generator.model.from_pretrained()
    feature, mask = msa_generator.predict(data)
    with open("./test.pkl", "rb") as f:
        data = pickle.load(f)
    for k, v in features:
        print(k, v.shape, v.dtype)
    ```

- MEGAFold

  - 使用搜索后所得pickle文件作为输入，样例如下：

  ```python
  import pickle
  import mindspore as ms
  from mindsponge import PipeLine
  ms.set_context(mode=ms.GRAPH_MODE)

  with open("./test.pkl", "rb") as f:
      feature = pickle.load(f)
  fold_prediction = PipeLine(name="MEGAFold")
  fold_prediction.set_device_id(0)
  fold_prediction.initialize(key="predict_256")
  fold_prediction.model.from_pretrained()
  res = fold_prediction.predict(feature)
  protein_structure = res[-1]
  pdb_file = to_pdb(protein_structure)
  os.makedirs(f'res.pdb', exist_ok=True)
  os_flags = os.O_RDWR | os.O_CREAT
  os_modes = stat.S_IRWXU
  pdb_path = './res.pdb'
  with os.fdopen(os.open(pdb_path, os_flags, os_modes), 'w') as fout:
      fout.write(pdb_file)

  print(protein_structure)
  ```

  - 单序列进行MSA检索并进行推理（完整流程），其中MSA检索配置请参考`application/common_utils/database_query/README.md`。检索完成后使用pickle进行推理场景与上述另一场景完全相同，不重复提供代码。

  - 后续MEGAFold会支持将蛋白质序列与template作为输入，不提供MSA进行推理的场景。

- MEGAAssessment

  - MEGAAssessment仅支持序列搜索所得pickle文件和MEGAFold推理所得pdb作为输入单场景，样例如下：

  ```python
  import pickle
  import numpy as np
  from mindspore import context
  from mindsponge import PipeLine
  from mindsponge.common.config_load import load_config
  from mindsponge.common.protein import from_pdb_string

  protein_assessment = PipeLine(name="MEGAAssessment")
  protein_assessment.set_device_id(0)

  # from mindsponge.pipeline.pipeline import download_config
  # download_config(protein_assessment.config["predict_256"], protein_assessment.config_path + "predict_256.yaml")

  conf = load_config(protein_assessment.config_path + "predict_256.yaml")
  protein_assessment.initialize(key="predict_256")
  protein_assessment.model.from_pretrained()

  # load raw feature
  with open("./test.pkl", "rb") as f:
    raw_feature = pickle.load(f)
  # load decoy pdb
  with open('./res.pdb', 'r') as f:
      decoy_prot_pdb = from_pdb_string(f.read())
  raw_feature['decoy_aatype'] = decoy_prot_pdb.aatype
  raw_feature['decoy_atom_positions'] = decoy_prot_pdb.atom_positions
  raw_feature['decoy_atom_mask'] = decoy_prot_pdb.atom_mask

  res = protein_assessment.predict(raw_feature)
  print("score is:", np.mean(res))
  ```

- 后处理

  AI结构预测方法如MEGA-Fold/AlphaFold2结果只包含碳/氮等重原子的位置信息，缺少氢原子；同时AI方法预测的蛋白质结构可能违反物理化学原理，比如键长键角超出理论值范围等。MindSPONGE提供基于Amber力场的结构弛豫工具，补全氢原子位置信息的同时使结构更符合物理规律，请参考`application/common_utils/openmm_relaxation/README.md`配置

## 训练过程

Pipeline中提供了MEGAFold和MEGAAssessment两个模型的训练代码。MEGAFold的训练集为PSP数据集，MEGAAssessment的训练集为PSP lite数据集。

MEGAFold的训练样例代码如下所示：

```bash
import mindspore as ms
from mindsponge import PipeLine

ms.set_context(mode=ms.GRAPH_MODE)

pipe = PipeLine(name="MEGAFold")
pipe.set_device_id(0)
pipe.initialize(key="initial_training")
pipe.train({YOUR_DATA_PATH}, num_epochs=1)
```

MEGAAssessment的训练样例代码如下所示：

```bash
from mindsponge import PipeLine

pipe = PipeLine(name="MEGAAssessment")
pipe.set_device_id(0)
pipe.initialize(key="initial_training")
pipe.train({YOUR_DATA_PATH}, num_epochs=1)
```

由于训练和推理代码网络结构存在差异，因此利用训练得到的权重进行推理、利用推理权重继续训练时，需要进行权重转换，示例代码如下：

```bash
from mindsponge.common.utils import get_predict_checkpoint, get_train_checkpoint

# 将训练得到的权重转换为推理权重
# training.ckpt: 训练得到的权重；
# 48：msa堆叠层数；
# predict.ckpt：需要被转换成的预测权重
get_predict_checkpoint("training.ckpt", 48, "predict.ckpt")

# 将推理时的权重转换为训练权重
# training.ckpt: 需要进行训练使用的权重；
# 48：msa堆叠层数；
# predict.ckpt：预测时使用的权重
get_train_checkpoint("training.ckpt", 48, "predict.ckpt")
```

## 引用

### 结构预测工具MEGA-Fold与训练数据集PSP

```bash
@misc{https://doi.org/10.48550/arxiv.2206.12240,
doi = {10.48550/ARXIV.2206.12240},
url = {https://arxiv.org/abs/2206.12240},
author = {Liu, Sirui and Zhang, Jun and Chu, Haotian and Wang, Min and Xue, Boxin and Ni, Ningxi and Yu, Jialiang and Xie, Yuhao and Chen, Zhenyu and Chen, Mengyun and Liu, Yuan and Patra, Piya and Xu, Fan and Chen, Jie and Wang, Zidong and Yang, Lijiang and Yu, Fan and Chen, Lei and Gao, Yi Qin},
title = {PSP: Million-level Protein Sequence Dataset for Protein Structure Prediction},
publisher = {arXiv},
year = {2022},
copyright = {Creative Commons Attribution 4.0 International}
}
```

### MSA生成修正工具MEGA-EvoGen

```bash
@article{doi:10.1021/acs.jctc.3c00528,
author = {Zhang, Jun and Liu, Sirui and Chen, Mengyun and Chu, Haotian and Wang, Min and Wang, Zidong and Yu, Jialiang and Ni, Ningxi and Yu, Fan and Chen, Dechin and Yang, Yi Isaac and Xue, Boxin and Yang, Lijiang and Liu, Yuan and Gao, Yi Qin},
title = {Unsupervisedly Prompting AlphaFold2 for Accurate Few-Shot Protein Structure Prediction},
journal = {Journal of Chemical Theory and Computation},
volume = {19},
number = {22},
pages = {8460-8471},
year = {2023},
doi = {10.1021/acs.jctc.3c00528},
note ={PMID: 37947474},
}
```
