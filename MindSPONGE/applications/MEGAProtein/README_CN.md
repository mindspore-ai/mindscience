
# MEGA-Protein

使用计算机高效计算获取蛋白质空间结构的过程被称为蛋白质结构预测，传统的结构预测工具一直存在精度不足的问题，直至2020年谷歌DeepMind团队提出[AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)<sup>[1,2]</sup>，该模型相较于传统工具预测精度大幅提升，所得结构与真实结构误差接近实验方法，但是仍存在数据前处理耗时过长、缺少MSA时预测精度不准、缺乏通用评估结构质量工具的问题。针对这些问题，高毅勤老师团队与MindSpore科学计算团队合作进行了一系列创新研究，开发出更准确和更高效的蛋白质结构预测工具**MEGA-Protein**，本目录即为MEGA-Protein的开源代码。

MEGA-Protein主要由三部分组成：

- **蛋白质结构预测工具MEGA-Fold**，网络模型部分与AlphaFold2相同，在数据预处理的多序列对比环节采用了[MMseqs2](https://www.biorxiv.org/content/10.1101/2021.08.15.456425v1.full.pdf)<sup>[3]</sup>进行序列检索，相比于原版端到端速度提升2-3倍。

- **MSA生成工具MEGA-EvoGen**，能显著提升单序列的预测速度，并且能够在MSA较少（few shot）甚至没有MSA（zero-shot，即单序列）的情况下，帮助MEGA-Fold/AlphaFold2等模型维持甚至提高推理精度，突破了在「孤儿序列」、高异变序列和人造蛋白等MSA匮乏场景下无法做出准确预测的限制，该方法获得了CAMEO-3D蛋白质结构预测赛道22年7月月榜第一

<div align=center>
<img src="../../docs/evogen_contest.jpg" alt="MEGA-EvoGen方法获得CAMEO-3D蛋白质结构预测赛道月榜第一" width="600"/>
</div>

- **蛋白质结构评分工具MEGA-Assessement**，该工具可以评价蛋白质结构每个残基的准确性以及残基-残基之间的距离误差，同时可以基于评价结果对蛋白结构作出进一步的优化，该方法获得了CAMEO-QE结构质量评估赛道22年7月月榜第一

<div align=center>
<img src="../../docs/assess_contest.png" alt="MEGA-Assessement方法获得CAMEO-QE结构质量评估赛道月榜第一" width="600"/>
</div>

<details><summary>引用我们</summary>

- 结构预测工具MEGA-Fold与训练数据集PSP:

    ```bibtex
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

- MSA生成工具MEGA-EvoGen:

- 蛋白质结构评分工具MEGA-Assessement:

</details>

<details><summary>README目录</summary>

<!-- TOC -->

- [MEGA-Protein](#MEGA-Protein)
  - [环境配置](#环境配置)
    - [硬件环境与框架](#硬件环境与框架)
    - [配置数据库检索](#配置数据库检索)
  - [代码目录](#代码目录)
  - [运行示例](#运行示例)
    - [MEGA-Fold蛋白质结构预测](#mega-fold蛋白质结构预测)
    - [MEGA-EvoGen MSA生成/增强](#mega-evogen-msa生成增强)
    - [MEGA-Assessement 蛋白质结构评分&优化](#mega-assessement-蛋白质结构评分优化)
    - [MEGA-Protein整体使用](#mega-protein整体使用)
  - [可用的模型和数据集](#可用的模型和数据集)
  - [引用](#引用)
  - [致谢](#致谢)

<!-- /TOC -->

</details>

<details><summary>近期更新</summary>

- 2022.04: 蛋白质结构预测工具MEGA-Fold训练开源.
- 2021.11: 蛋白质结构预测工具MEGA-Fold推理开源.

</details>

## 环境配置

### 硬件环境与框架

本工具基于[MindSPONGE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE)生物计算库与[MindSpore](https://www.mindspore.cn/)AI框架开发，MindSpore 1.8及以后的版本均可运行，MindSpore安装和配置可以参考[MindSpore安装页面](https://www.mindspore.cn/install)，其余python依赖请参见[requirements.txt](to_add)。

本工具可以Ascend910或GPU上运行：基于Ascend910运行时需配置环境变量`export MS_DEV_ENABLE_CLOSURE=0`，运行时默认调用混合精度推理；基于GPU运行时默认使用全精度推理。

蛋白质结构预测工具MEGA-Fold依赖多序列比对(MSA，multiple sequence alignments)与模板检索生成等传统数据库搜索工具提供的共进化与模板信息，配置数据库搜索需**2.5T硬盘**（推荐SSD）和与Kunpeng920性能持平的CPU。

### 配置数据库检索

- 配置MSA检索

    首先安装MSA搜索工具**MMseqs2**，该工具的安装和使用可以参考[MMseqs2 User Guide](https://mmseqs.com/latest/userguide.pdf)，安装完成后运行以下命令配置环境变量：

    ``` shell
    export PATH=$(pwd)/mmseqs/bin/:$PATH
    ```

    然后下载MSA所需数据库：

    - [uniref30_2103](http://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2103.tar.gz)：压缩包68G，解压后375G
    - [colabfold_envdb_202108](http://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz)：压缩包110G，解压后949G

    下载完成后需解压并使用MMseqs2处理数据库，数据处理参考[colabfold](http://colabfold.mmseqs.com)，主要命令如下：

    ``` bash
    tar xzvf "uniref30_2103.tar.gz"
    mmseqs tsv2exprofiledb "uniref30_2103" "uniref30_2103_db"
    mmseqs createindex "uniref30_2103_db" tmp1 --remove-tmp-files 1

    tar xzvf "colabfold_envdb_202108.tar.gz"
    mmseqs tsv2exprofiledb "colabfold_envdb_202108" "colabfold_envdb_202108_db"
    mmseqs createindex "colabfold_envdb_202108_db" tmp2 --remove-tmp-files 1
    ```

- 配置模板检索

    首先安装模板搜索工具[**HHsearch**](https://github.com/soedinglab/hh-suite)
    与[**kalign**](https://msa.sbc.su.se/downloads/kalign/current.tar.gz)，然后下载模板检索所需数据库：

    - [pdb70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz)：压缩包19G，解压后56G
    - [mmcif database](https://ftp.rcsb.org/pub/pdb/data/structures/divided/mmCIF/)： 零散压缩文件～50G，解压后～200G，需使用爬虫脚本下载，下载后需解压所有mmcif文件放在同一个文件夹内。
    - [obsolete_pdbs](http://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat)：140K

  *数据库下载网站均为国外网站，下载速度可能较慢，需要自行配置VPN*。

## 代码目录

```bash
├── MEGA-Protein
    ├── main.py                         // MEGA-Protein主脚本
    ├── README_CN.md                    // MEGA-Protein相关中文说明
    ├── config
        ├── data.yaml                   //数据处理参数配置
        ├── model.yaml                  //模型参数配置
    ├── data
        ├── dataset.py                  // 异步数据读取脚本
        ├── hhsearch.py                 // python封装的HHsearch工具
        ├── kalign.py                   // python封装的Kalign工具
        ├── msa_query.py                // python封装的MSA搜索及处理工具
        ├── msa_search.sh               // 调用MMseqs2搜索MSA的shell脚本
        ├── multimer_pipeline.py        // 复合物数据预处理脚本
        ├── preprocess.py               // 数据预处理脚本
        ├── protein_feature.py          // MSA与template特征搜索与整合脚本
        ├── templates.py                // 模板搜索脚本
        ├── utils.py                    // 数据处理所需功能函数
    ├── model
        ├── fold.py                     // MEGA-Fold主模型脚本
    ├── module
        ├── evoformer.py                // evoformer特征提取模块
        ├── fold_wrapcell.py            // 训练迭代封装模块
        ├── head.py                     // MEGA-Fold附加输出模块
        ├── loss_module.py              // MEGA-Fold训练loss模块
        ├── structure.py                // 3D结构生成模块
        ├── template_embedding.py       // 模板信息提取模块
```

## 运行示例

### MEGA-Fold蛋白质结构预测

加载已经训好的checkpoint，下载地址[点击这里](https://download.mindspore.cn/model_zoo/research/hpc/molecular_dynamics/MEGA_Fold_1.ckpt)，根据数据库安装情况配置启动命令：

```bash
用法：run.py [--seq_length PADDING_SEQENCE_LENGTH]
             [--input_fasta_path INPUT_PATH][--msa_result_path MSA_RESULT_PATH]
             [--database_dir DATABASE_PATH][--database_envdb_dir DATABASE_ENVDB_PATH]
             [--hhsearch_binary_path HHSEARCH_PATH][--pdb70_database_path PDB70_PATH]
             [--template_mmcif_dir TEMPLATE_PATH][--max_template_date TRMPLATE_DATE]
             [--kalign_binary_path KALIGN_PATH][--obsolete_pdbs_path OBSOLETE_PATH]


选项：
  --seq_length             补零后序列长度，目前支持256/512/1024/2048
  --input_fasta_path       FASTA文件，用于预测蛋白质结构的蛋白质序列
  --msa_result_path        保存mmseqs2检索得到的msa结果路径
  --database_dir           uniref30文件夹路径
  --database_envdb_dir     colabfold_envdb_202108文件夹路径
  --hhsearch_binary_path   HHsearch可执行文件路径
  --pdb70_database_path    pdb70文件夹路径
  --template_mmcif_dir     mmcif文件夹路径
  --max_template_date      模板最新发布的时间
  --kalign_binary_path     kalign可执行文件路径
  --obsolete_pdbs_path     PDB IDs的映射文件路径
```

推理结果保存在 `./result` 中，共有两个文件，pdb文件即为蛋白质结构预测结果，文件中导数第二列为单个残基的预测置信度，timings文件保存了运行过程中的时间信息。

```bash
{"pre_process_time": 418.57, "model_time": 122.86, "pos_process_time": 0.14, "all_time ": 541.56, "confidence ": 94.61789646019058}
```

34条CASP14序列MEGA-Fold与AlphaFold2预测TMscore对比：

<div align=center>
<img src="../../docs/all_experiment_data.jpg" alt="all_data" width="300"/>
</div>

MEGA-Fold预测结果与真实结果对比：

- T1079(长度505)：

<div align=center>
<img src="../../docs/seq_64.gif" alt="T1079" width="300"/>
</div>

- T1044(长度2180)：

<div align=center>
<img src="../../docs/seq_21.jpg" alt="T1044" width="300"/>
</div>

### MEGA-EvoGen MSA生成/增强

To be released

### MEGA-Assessement 蛋白质结构评分&优化

To be released

### MEGA-Protein整体使用

To be released

## 可用的模型和数据集

| 所属模块      | 文件名        | 大小 | 描述  |Model URL  |
|-----------|---------------------|---------|---------------|-----------------------------------------------------------------------|
| MEGA-Fold    | `MEGA_Fold_1.ckpt` | 356MB       | MEGA-Fold在PSP数据集训练的数据库与checkpoint链接 |  [下载链接](https://download.mindspore.cn/model_zoo/research/hpc/molecular_dynamics/MEGA_Fold_1.ckpt)  |
| PSP          | `PSP`         | 2TB(解压后25TB)    | PSP蛋白质结构数据集，可用于MEGA-Fold训练 |  [下载链接](To be released)  |

## 引用

[1] Jumper J, Evans R, Pritzel A, et al. Applying and improving AlphaFold at CASP14[J].  Proteins: Structure, Function, and Bioinformatics, 2021.

[2] Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold[J]. Nature, 2021, 596(7873): 583-589.

[3] Mirdita M, Ovchinnikov S, Steinegger M. ColabFold-Making protein folding accessible to all[J]. BioRxiv, 2021.

## 致谢

MEGA-Fold使用或参考了以下开源工具：

- [AlphaFold2](https://github.com/deepmind/alphafold)
- [Biopython](https://biopython.org)
- [ColabFold](https://github.com/sokrypton/ColabFold)
- [HH Suite](https://github.com/soedinglab/hh-suite)
- [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)
- [ML Collections](https://github.com/google/ml_collections)
- [NumPy](https://numpy.org)
- [OpenMM](https://github.com/openmm/openmm)

我们感谢这些开源工具所有的贡献者和维护者！
