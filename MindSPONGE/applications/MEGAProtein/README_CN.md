[ENGLISH](README.md)|简体中文

# MEGA-Protein

使用计算机高效计算获取蛋白质空间结构的过程被称为蛋白质结构预测，传统的结构预测工具一直存在精度不足的问题，直至2020年谷歌DeepMind团队提出[AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)<sup>[1,2]</sup>，该模型相较于传统工具预测精度大幅提升，所得结构与真实结构误差接近实验方法，但是仍存在数据前处理耗时过长、缺少MSA时预测精度不准、缺乏通用评估结构质量工具的问题。针对这些问题，高毅勤老师团队与MindSpore科学计算团队合作进行了一系列创新研究，开发出更准确和更高效的蛋白质结构预测工具**MEGA-Protein**，本目录即为MEGA-Protein的开源代码。

MEGA-Protein主要由三部分组成：

- **蛋白质结构预测工具MEGA-Fold**，网络模型部分与AlphaFold2相同，在数据预处理的多序列对比环节采用了[MMseqs2](https://www.biorxiv.org/content/10.1101/2021.08.15.456425v1.full.pdf)<sup>[3]</sup>进行序列检索，相比于原版端到端速度提升2-3倍；同时借助内存复用大幅提升内存利用效率，同硬件条件下支持更长序列的推理（基于32GB内存的Ascend910运行时最长支持3072长度序列推理）；我们还提供了结构预测模型训练能力，我们自己训练的权重获得了CAMEO-3D蛋白质结构预测赛道22年4月月榜第一。

<div align=center>
<img src="../../docs/megafold_contest.png" alt="MEGA-Fold获得CAMEO-3D蛋白质结构预测赛道月榜第一" width="600"/>
</div>

- **MSA生成工具MEGA-EvoGen**，能显著提升单序列的预测速度，并且能够在MSA较少（few shot）甚至没有MSA（zero-shot，即单序列）的情况下，帮助MEGA-Fold/AlphaFold2等模型维持甚至提高推理精度，突破了在「孤儿序列」、高异变序列和人造蛋白等MSA匮乏场景下无法做出准确预测的限制，该方法获得了CAMEO-3D蛋白质结构预测赛道22年7月月榜第一

<div align=center>
<img src="../../docs/evogen_contest.jpg" alt="MEGA-EvoGen方法获得CAMEO-3D蛋白质结构预测赛道月榜第一" width="600"/>
</div>

- **蛋白质结构评分工具MEGA-Assessment**，该工具可以评价蛋白质结构每个残基的准确性以及残基-残基之间的距离误差，同时可以基于评价结果对蛋白结构作出进一步的优化，该方法获得了CAMEO-QE结构质量评估赛道22年7月月榜第一

<div align=center>
<img src="../../docs/assess_contest.png" alt="MEGA-Assessment方法获得CAMEO-QE结构质量评估赛道月榜第一" width="600"/>
</div>

## 可用的模型和数据集

| 所属模块      | 文件名        | 大小 | 描述  | Model URL                                                                                                   |
|-----------|---------------------|---------|---------------|-------------------------------------------------------------------------------------------------------------|
| MEGA-Fold    | `MEGA_Fold_1.ckpt` | 356MB       | MEGA-Fold在PSP数据集训练的数据库与checkpoint链接 | [下载链接](https://download.mindspore.cn/mindscience/mindsponge/MEGAFold/checkpoint/MEGA_Fold_1.ckpt)           |
| MEGA-EvoGen     | `MEGAEvoGen.ckpt`  | 535.7MB        | MEGA-EvoGen的checkpoint链接 | [下载链接](https://download.mindspore.cn/mindscience/mindsponge/MEGAEvoGen/checkpoint/MEGAEvoGen.ckpt)          |
| MEGA-Assessment     | `MEGA_Assessment.ckpt`  | 77MB        | MEGA-Assessment的checkpoint链接 | [下载链接](https://download.mindspore.cn/mindscience/mindsponge/MEGAAssessment/checkpoint/MEGA_Assessment.ckpt) |
| PSP          | `PSP`         | 1.6TB(解压后25TB)    | PSP蛋白质结构数据集，可用于MEGA-Fold训练 | [下载链接](http://ftp.cbi.pku.edu.cn/psp/)                                                                      |

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

    ```bibtex
    @article{zhang2022few,
        title={Few-shot learning of accurate folding landscape for protein structure prediction},
        author={Zhang, Jun and Liu, Sirui and Chen, Mengyun and Chu, Haotian and Wang, Min and Wang, Zidong and Yu, Jialiang and Ni, Ningxi and Yu, Fan and Chen, Diqing and others},
        journal={arXiv preprint arXiv:2208.09652},
        year={2022}
      }
    ```

- 蛋白质结构评分工具MEGA-Assessment:

</details>

<details><summary>README目录</summary>

<!-- TOC -->

- [MEGA-Protein](#mega-protein)
  - [环境配置](#环境配置)
    - [硬件环境与框架](#硬件环境与框架)
    - [配置数据库检索](#配置数据库检索)
  - [代码目录](#代码目录)
  - [运行示例](#运行示例)
    - [MEGA-Fold蛋白质结构预测](#mega-fold蛋白质结构预测)
    - [MEGA-EvoGen MSA生成/增强](#mega-evogen-msa生成增强)
    - [MEGA-Assessment 蛋白质结构评分&优化](#mega-assessment-蛋白质结构评分优化)
    - [MEGA-Protein整体使用](#mega-protein整体使用)
  - [可用的模型和数据集](#可用的模型和数据集)
  - [引用](#引用)
  - [致谢](#致谢)

<!-- /TOC -->

</details>

<details><summary>近期更新</summary>

- 2022.11：MSA生成/增强工具MEGA-Assessment训练推理开源。
- 2022.11：MSA生成/增强工具MEGA-EvoGen推理开源。
- 2022.04: 蛋白质结构预测工具MEGA-Fold训练开源。
- 2021.11: 蛋白质结构预测工具MEGA-Fold推理开源。

</details>

## 环境配置

### 硬件环境与框架

本工具基于[MindSPONGE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE)生物计算库与[MindSpore](https://www.mindspore.cn/)AI框架开发，MindSpore 1.8及以后的版本均可运行，MindSpore安装和配置可以参考[MindSpore安装页面](https://www.mindspore.cn/install)。本工具可以Ascend910或32G以上内存的GPU上运行，基于Ascend运行时默认调用混合精度，基于GPU运行时使用全精度计算。由于训练中使用了重计算功能，所以当前训练仅支持图模式。

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

- 配置MSA检索加速(可选)

    下载MSA加速缓存工具：
    - [FoldMSA.tar.gz](https://download.mindspore.cn/mindscience/mindsponge/msa_tools/Fold_MSA.tar.gz)：按照工具内说明操作进行MSA搜索加速。

- 配置模板检索

    首先安装模板搜索工具[**HHsearch**](https://github.com/soedinglab/hh-suite)
    与[**kalign**](https://msa.sbc.su.se/downloads/kalign/current.tar.gz)，然后下载模板检索所需数据库：

    - [pdb70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz)：压缩包19G，解压后56G
    - [mmcif database](https://ftp.rcsb.org/pub/pdb/data/structures/divided/mmCIF/)： 零散压缩文件～50G，解压后～200G，需使用爬虫脚本下载，下载后需解压所有mmcif文件放在同一个文件夹内。
    - [obsolete_pdbs](http://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat)：140K

  *数据库下载网站均为国外网站，下载速度可能较慢，需要自行配置VPN*。

    - 配置数据库检索config

    根据数据库安装情况配置`config/data.yaml`中数据库搜索的相关配置`database_search`，相关参数含义如下：

    ```bash
    # configuration for template search
    hhsearch_binary_path   HHsearch可执行文件路径
    kalign_binary_path     kalign可执行文件路径
    pdb70_database_path    pdb70文件夹路径
    mmcif_dir              mmcif文件夹路径
    obsolete_pdbs_path     PDB IDs的映射文件路径
    max_template_date      模板搜索截止时间，该时间点之后的模板会被过滤掉，默认值"2100-01-01"
    # configuration for Multiple Sequence Alignment
    mmseqs_binary          MMseqs2可执行文件路径
    uniref30_path          uniref30文件夹路径
    database_envdb_dir     colabfold_envdb_202108文件夹路径
    a3m_result_path        mmseqs2检索结果(msa)的保存路径，默认值"./a3m_result/"
    ```

## 代码示例

<details><summary><font size=4 color="blue">代码目录</font></summary>

```bash
├── MEGA-Protein
    ├── main.py                         // MEGA-Protein主脚本
    ├── README.md                       // MEGA-Protein相关英文说明
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
        ├── parsers.py                  // mmcif文件读取脚本
        ├── preprocess.py               // 数据预处理脚本
        ├── protein_feature.py          // MSA与template特征搜索与整合脚本
        ├── templates.py                // 模板搜索脚本
        ├── utils.py                    // 数据处理所需功能函数
    ├── examples
        ├── pdb                         //样例输入数据（.pkl文件）
        ├── pkl                         //样例输出数据（.pdb文件）
    ├── model
        ├── fold.py                     // MEGA-Fold主模型脚本
    ├── module
        ├── evoformer.py                // evoformer特征提取模块
        ├── fold_wrapcell.py            // 训练迭代封装模块
        ├── head.py                     // MEGA-Fold附加输出模块
        ├── loss_module.py              // MEGA-Fold训练loss模块
        ├── structure.py                // 3D结构生成模块
        ├── template_embedding.py       // 模板信息提取模块
    ├── scripts
        ├── run_fold_infer_gpu.sh       // GPU运行MEGA-Fold推理示例
        ├── run_fold_train_ascend.sh    // Ascend运行MEGA-Fold推理示例
```

</details>

### MEGA-Fold蛋白质结构预测推理

配置数据库搜索与`config/data.yaml`中的相关参数，下载已经训好的模型权重[MEGA_Fold_1.ckpt](https://download.mindspore.cn/mindscience/mindsponge/MEGAFold/checkpoint/MEGA_Fold_1.ckpt)，运行以下命令启动推理。

```bash
用法：python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --run_platform PLATFORM
            --input_path INPUT_FILE_PATH --checkpoint_path CHECKPOINT_PATH

选项：
--data_config        数据预处理参数配置
--model_config       模型超参配置
--input_path         输入文件目录，可包含多个.fasta/.pkl文件
--checkpoint_path    模型权重文件路径
--use_pkl            使用pkl数据作为输入，默认False
--run_platform       运行后端，Ascend或者GPU，默认Ascend
```

对于多条序列推理，MEGA-Fold会基于所有序列的最长长度自动选择编译配置，避免重复编译。如需推理的序列较多，建议根据序列长度分类放入不同文件夹中分批推理。由于数据库搜索硬件要求较高，MEGA-Fold支持先做数据库搜索生成`raw_feature`并保存为pkl文件，然后使用`raw_feature`作为预测工具的输入，此时须将`use_pkl`选项置为True，`examples`文件夹中提供了样例pkl文件与对应的真实结构，供测试运行，测试命令参考`scripts/run_fold_infer_gpu.sh`。

推理结果保存在 `./result/` 目录下，每条序列的结果存储在独立文件夹中，以序列名称命名，文件夹中共有两个文件，pdb文件即为蛋白质结构预测结果，其中倒数第二列为氨基酸残基的预测置信度；timings文件保存了推理不同阶段时间信息以及推理结果整体的置信度。

```log
{"pre_process_time": 0.61, "model_time": 87.5, "pos_process_time": 0.02, "all_time ": 88.12, "confidence ": 93.5}
```

MEGA-Fold预测结果与真实结果对比：

- 7VGB_A，长度711，lDDT 92.3：

<div align=center>
<img src="../../docs/7VGB_A.png" alt="7VGB_A" width="400"/>
</div>

### MEGA-Fold蛋白质结构预测训练

下载开源结构训练数据集[PSP dataset](http://ftp.cbi.pku.edu.cn/psp/)，使用以下命令启动训练：

```bash
用法：python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --is_training True
            --input_path INPUT_PATH --pdb_path PDB_PATH --run_platform PLATFORM

选项：
--data_config        数据预处理参数配置
--model_config       模型超参配置
--is_training        设置为训练模式 (推理无需添加此参数)
--input_path         训练输入数据（pkl文件，包含MSA与模板信息）路径
--pdb_path           训练标签数据（pdb文件，真实结构或知识蒸馏结构）路径
--run_platform       运行后端，Ascend或者GPU，默认Ascend
```

代码默认每50次迭代保存一次权重，权重保存在`./ckpt`目录下。数据集下载及测试命令参考参考`scripts/run_fold_train.sh`。

### MEGA-EvoGen MSA生成/增强推理

MEGA-EvoGen相关超参位于 `./config/evogen.yaml`，
然后下载模型权重 [MEGAEvoGen.ckpt](https://download.mindspore.cn/mindscience/mindsponge/MEGAEvoGen/checkpoint/MEGAEvoGen.ckpt)
， 最后使用如下命令运行模型。

```bash
用法：python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --evogen_config ./config/evogen.yaml
            --input_path INPUT_FILE_PATH --checkpoint_path CHECKPOINT_PATH --run_evogen 1

选项：
--data_config        蛋白质结构预测模型数据预处理参数配置
--model_config       蛋白质结构预测模型超参配置
--evogen_config      MSA生成/增强模型超参配置
--input_path         输入文件目录，可包含多个`.fasta/.pkl`文件
--checkpoint_path    MEGA-EvoGen模型和MEGA-Fold模型权重文件路径
--run_evogen         运行MSA生成/增强推理得到蛋白质结构
```

### MEGA-Assessment 蛋白质结构评分推理

下载已经训好的MEGA-Assessment模型权重[MEGA_Assessment.ckpt](https://download.mindspore.cn/mindscience/mindsponge/MEGAAssessment/checkpoint/MEGA_Assessment.ckpt)运行以下命令启动推理。

```bash
用法：python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --input_path INPUT_FILE_PATH
            --decoy_pdb_path INPUT_FILE_PATH --checkpoint_path_assessment CHECKPOINT_PATH_ASSESSMENT
            --run_assessment=1

选项：
--data_config                   数据预处理参数配置
--model_config                  模型超参配置
--input_path                    输入文件目录，可包含多个`.fasta/.pkl`文件
--decoy_pdb_path                待评估蛋白质结构路径，可包含多个`_decoy.pdb`文件
--checkpoint_path_assessment    MEGA-Assessment模型权重文件路径
--run_assessment                运行蛋白质结构评估
```

### MEGA-Assessment 蛋白质结构评分训练

下载开源结构训练数据集[PSP lite dataset](http://ftp.cbi.pku.edu.cn/psp/psp_lite/)，使用以下命令启动训练：
和已经训好的MEGA-Fold模型权重[MEGA_Fold_1.ckpt](https://download.mindspore.cn/mindscience/mindsponge/MEGAFold/checkpoint/MEGA_Fold_1.ckpt)，运行以下命令启动训练。

```bash
用法：python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --is_training True
            --input_path INPUT_PATH --pdb_path PDB_PATH --checkpoint_path CHECKPOINT_PATH --run_assessment 1

选项：
--data_config        数据预处理参数配置
--model_config       模型超参配置
--is_training        设置为训练模式 (推理无需添加此参数)
--input_path         输入文件目录，可包含多个`.fasta/.pkl`文件
--pdb_path           训练标签数据（pdb文件，真实结构或知识蒸馏结构）路径
--checkpoint_path    MEGA-Fold模型权重文件路径
--run_assessment     运行蛋白质结构评估
```

### MEGA-Protein整体使用

To be released

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
