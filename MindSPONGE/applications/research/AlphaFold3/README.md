# AlphaFold3-MindSpore

[**MindSpore版 AlphaFold3实现**] 一个基于MindSpore深度学习框架的AlphaFold3推理网络结构实现。

> 📖 **语言版本**: [中文](README.md) | [English](README_EN.md)

## 📑 目录

- [项目简介](#项目简介)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [许可证](#许可证)
- [致谢](#致谢)
- [参考文献](#参考文献)

## 项目简介

**项目背景**：
AlphaFold3是DeepMind在2024年发布的革命性生物分子结构预测模型，能够预测蛋白质、DNA、RNA等生物大分子的三维结构。本项目基于Ascend NPU和MindSpore框架，实现了AlphaFold3的推理功能。

AlphaFold3 的模型结构如下图所示：

![AlphaFold3 模型结构](image/af3_structure.jpg)

- **推理流程**：首先输入的蛋白，核酸，配体等序列信息，经过模板搜索（Template Search）、多序列比对（Multiple Sequence Alignment, MSA）等预处理步骤，然后通过embeding部分对输入信息进行编码，之后通过Pairformer模块，获取序列及结构的关系，接着进入扩散模块生成三维结构，最后通过置信度模块给出预测的置信度评分
- **生物分子结构预测**: 基于AlphaFold3算法的生物分子结构预测模型,支持包括蛋白质，DNA，RNA，小分子在内的多种输入形式；支持多链输入，预测相互作用和相对位置
- **MindSpore支持**: 基于MindSpore对模型推理功能进行适配

### 硬件要求

- Atlas 800T A2

### 软件要求

- Python >= 3.11
- MindSpore >= 2.5.0
- CANN = 8.0.0
- cmake >= 3.28.1

## 安装

### 1. 克隆仓库

```bash
git clone https://gitee.com/mindspore/mindscience
cd mindsience/MindSPONGE/application/research/AlphaFold3
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
#`{PATH}` 为当前目录
export PYTHONPATH={PATH}/mindscience/MindSPONGE/src
export PYTHONPATH={PATH}/mindscience/MindChemistry
```

### 3. 安装软件包

[hmmer](http://eddylab.org/software/hmmer/) 在链接处下载安装包，如 `hmmer-3.4.tar.gz`，并放置在当前目录下，然后执行以下命令：

```bash
mkdir /path/to/hmmer_build /path/to/hmmer && \
mv ./hmmer-3.4.tar.gz /path/to/hmmer_build && \
cd /path/to/hmmer_build && tar -zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz && \
cd /path/to/hmmer_build/hmmer-3.4 && ./configure --prefix=/path/to/hmmer && \
make -j8 && make install && \
cd /path/to/hmmer_build/hmmer-3.4/easel && make install && \
rm -rf /path/to/hmmer_build
export PATH=/hmmer/bin:$PATH
which jackhmmer
```

如果出现`/path/to/hmmer/bin/jackhmmer`则安装成功

### 4. 编译

```bash
cd {PATH}/mindscience/MindSPONGE/applications/research/AlphaFold3
mkdir build
cd build
cmake ..
make
cp ./cpp.cpython-311-aarch64-linux-gnu.so ../src/alphafold
cd ..
```

生成数据文件：

```bash
python ./src/alphafold3/build_data.py
```

如出现报错找不到components.cif,可以去[wwpdb](https://files.wwpdb.org/pub/pdb/data/monomers/components.cif)下载components.cif文件，放置在conda环境中`{CONDA_ENV_DIR}/lib/python3.11/site-packages/share/libcifpp`文件夹下。如不存在`share/libcifpp`文件夹，则需要手动创建。

### 5. 下载数据库

可以从DeepMind官网下载测试用小数据库[miniature_databases](https://github.com/google-deepmind/alphafold3/tree/main/src/alphafold3/test_data/miniature_databases)（影响推理结果，仅测试使用！）
下载后放置在统一文件夹中并修改文件名如下所示(如统一放置在`/mindscience/MindSPONGE/applications/research/AlphaFold3/public_databases`可省略`--db_dir=/PATH/TO/DB_DIR`)：

```txt
miniature_databases
    └─ mmcif_files
    │  bfd-first_non_consensus_sequences.fasta
    │  mgy_clusters_2022_05.fa
    │  pdb_seqres_2022_09_28.fasta
    │  uniprot_all_2021_04.fa
    │  uniref90_2022_05.fa
    │  nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
    │  rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
    │  rnacentral_active_seq_id_90_cov_80_linclust.fasta
```

如果想要搜索完整的数据库，请从以下链接下载数据库，放置到同一文件夹中(如统一放置在`/mindscience/MindSPONGE/applications/research/AlphaFold3/public_databases`可省略`--db_dir=/PATH/TO/DB_DIR`):

- [mmcif](https://storage.googleapis.com/alphafold-databases/v3.0/pdb_2022_09_28_mmcif_files.tar.zst)
- [BFD](https://storage.googleapis.com/alphafold-databases/v3.0/bfd-first_non_consensus_sequences.fasta.zst)
- [MGnify](https://storage.googleapis.com/alphafold-databases/v3.0/mgy_clusters_2022_05.fa.zst)
- [PDB seqres](https://storage.googleapis.com/alphafold-databases/v3.0/pdb_seqres_2022_09_28.fasta.zst)
- [UniProt](https://storage.googleapis.com/alphafold-databases/v3.0/uniprot_all_2021_04.fa.zst)
- [uniref90](https://storage.googleapis.com/alphafold-databases/v3.0/uniref90_2022_05.fa.zst)
- [NT](https://storage.googleapis.com/alphafold-databases/v3.0/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst)
- [RFam](https://storage.googleapis.com/alphafold-databases/v3.0/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst)
- [RNACentral](https://storage.googleapis.com/alphafold-databases/v3.0/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst)

请确保磁盘中有足够空间：
|   DataBase   |   Compressed Size   | Uncompressed Size|
|--------------|---------------------|------------------|
|    mmcif     |   233G              |    233G          |
|    BFD       |   9.2G              |    16.9G         |
|    MGnify    |   64.5G             |    119G          |
|    PDB seqres|   25.3M             |    217M          |
|    UniProt   |   45.3G             |    101G          |
|    uniref90  |   30.9G             |    66.8G         |
|    NT        |   15.8G             |    75.4G         |
|    RFam      |   53.9M             |    217M          |
|    RNACentral|   3.27G             |    12.9G         |
|    total     |   402G              |    534G          |

解压下载的数据文件：

```bash
cd /PATH/TO/YOUR/DATA_DIR
tar –use-compress-program=unzstd -xf pdb_2022_09_28_mmcif_files.tar.zst
zstd -d bfd-first_non_consensus_sequences.fasta.zst
zstd -d mgy_clusters_2022_05.fa.zst
zstd -d pdb_seqres_2022_09_28.fasta.zst
zstd -d uniprot_all_2021_04.fa.zst
zstd -d uniref90_2022_05.fa.zst
zstd -d nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst
zstd -d rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst
zstd -d rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst
```

如统一放置在`/mindscience/MindSPONGE/applications/research/AlphaFold3/public_databases`可在运行时省略`--db_dir=/PATH/TO/DB_DIR`

## 快速开始

### 输入数据格式

示例输入JSON:

```json
{
  "name": "5tgy",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "SEFEKLRQTGDELVQAFQRLREIFDKGDDDSLEQVLEEIEELIQKHRQLFDNRQEAADTEAAKQGDQWVQLFQRFREAIDKGDKDSLEQLLEELEQALQKIRELAEKKN"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

### 运行流程

使用以下命令运行模型（计算精度float32）：

```bash
source set_path.sh
python run_alphafold.py \
  --json_path=example_input.json \
  --output_dir=output \
  --run_data_pipeline=true \
  --run_inference=true \
  --db_dir=/PATH/TO/DB_DIR \
  --model_dir=/PATH/TO/MODEL_DIR\
  --buckets=256
```

### 参数说明

- `--json_path`输入文件名称
- `--output_dir`: 输出文件路径
- `--run_data_pipeline`: 是否运行数据处理模块
- `--run_inference`: 是否运行推理模块
- `--db_dir`: 数据库存放路径, 默认 `{HOME}/public_databases`
- `--model_dir`: 模型文件路径, 默认 `{HOME}/ckpt`
- `--buckets`: 设定序列长度，如不设置会将序列长度padding到256的倍数，如传入则使用传入值作为序列长度

### 输入与输出文件说明

- **JSON格式数据输入**: 包含蛋白质核酸等的序列信息。当前支持输入种类与DeepMind版本相同，支持蛋白质，DNA，RNA及Ligand作为输入，当前推理版本为单卡版本支持序列长度不超过1000

- **输出文件**: 5个标准的蛋白质结构文件，及置信度信息

```txt
└─name_in_your_json
    └─ seed-1_sample-0                # 第一个生成样本
      │  confidence.json              # 第一个样本的详细置信度文件
      │  model.cif                    # 第一个样本的结构文件
      │  summary_confidence.json      # 第一个样本的总体置信度文件
    └─ seed-1_sample-1                # 第二个生成样本
    └─ seed-1_sample-2                # 第三个生成样本
    └─ seed-1_sample-3                # 第四个生成样本
    └─ seed-1_sample-4                # 第五个生成样本
    │  {name}_confidences.json        # 最优样本的详细置信度文件
    │  {name}_data.json               # 数据处理后的数据文件
    │  {name}_model.cif               # 最优样本的结构文件
    │  {name}_summary_confidence.json # 最优样本的总体置信度文件
    │  ranking_scores.csv             # 五个样本的ranking score；ranking score越高，表明置信度越高
```

### 推理完成

当看到如下日志，表明推理正常结束：

```txt
=======write output to /PATH/TO/OUTPUT/DIR/name_of_your_input==========
Done processing fold input name_of_your_input.
Done processing 1 fold inputs.
```

## 许可证

详情请参阅 [LICENSE](LICENSE) 文件。

## 致谢

- `data`，`structure`，`common`，`constant`等模块使用了[DeepMind](https://deepmind.com/)实现
- `model`，`utils`等模块基于[MindSpore](https://www.mindspore.cn/)实现

## 联系我们

如果您在使用过程中遇到任何问题或有任何建议，请通过以下方式与我们联系：

- **Gitee仓库**：[AlphaFold3](https://gitee.com/mindspore/mindscience/tree/main/MindSPONGE/applications/research/AlphaFold3)
- **问题跟踪**：[问题单跟踪](https://gitee.com/mindspore/mindscience/issues)

## 参考文献

- Abramson J, Adler J, Dunger J, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3[J]. Nature, 2024, 630(8016): 493-500.