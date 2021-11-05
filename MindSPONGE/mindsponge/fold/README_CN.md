# 目录

<!-- TOC -->

- [目录](#目录)
    - [模型描述](#模型描述)
    - [环境要求](#环境要求)
    - [数据准备](#数据准备)  
    - [快速入门](#快速入门)  
    - [脚本说明](#脚本说明)  
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
        - [参数配置](#参数配置)  
        - [推理过程](#推理过程)  
            - [在Ascend910执行推理](#在ascend910执行推理)
            - [结果](#结果)
    - [推理性能](#推理性能)

<!-- /TOC -->

## 模型描述

AlphaFold2与2021年提出，多序列比对结合深度学习算法，可根据蛋白质的氨基酸序列预测蛋白质的三维结构。AlphaFold2在CASP14中证明其准确性和实验结构具有竞争力，并且大大优于其他方法。

[论文](https://www.nature.com/articles/s41586-021-03819-2):Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold[J]. Nature, 2021, 596(7873): 583-589.

## 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/mindsponge/fold/requirements.txt)

### MMseqs2安装

MMseqs2用于生成多序列比对(multiple sequence alignments,MSA), MMseqs2安装和使用可以参考[MMseqs2 User Guide](https://mmseqs.com/latest/userguide.pdf)

### MindSpore Serving安装（可选）

提供MindSpore Serving服务，MindSpore Serving可以提供高效部署在线推理服务，MindSpore Serving安装和配置可以参考[MindSpore Serving安装页面](https://www.mindspore.cn/serving/docs/zh-CN/r1.5/serving_install.html).

## 数据准备

### MSA所需数据库

[uniref30_2103](http://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2103.tar.gz)：375G（下载68G）。

[colabfold_envdb_202108](http://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz)：949G（下载110G）。

### template所需工具和数据

#### 数据

[pdb70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/)

[mmcif database](https://www.rcsb.org/downloads)

[obsolete_pdbs](ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat)

### 工具

[HH-suite](https://github.com/soedinglab/hh-suite)

[kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行推理：

- Ascend处理器环境运行

```python
# 推理示例
python run.py  --input_fasta_path=[INPUT_PATH] --database_dir=[DATABASE_PATH] --database_envdb_dir=[DATABASE_ENVDB_PATH] --hhsearch_binary_path=[HHSEARCH_PATH]
 --pdb70_database_path=[PDB70_PATH]  --template_mmcif_dir=[TEMPLATE_PATH] --kalign_binary_path=[KALIGN_PATH] --obsolete_pdbs_path=[OBSOLETE_PATH] > output.log 2>&1 &
```

## 脚本说明

### 脚本及样例代码

```bash
├── mindscience
    ├── MindSPONGE
        ├── fold
            ├── README_CN.md                    // fold 相关中文说明
            ├── run.py                          // 推理脚本
            ├── model.py                        // 主模型
            ├── module
                ├── basic_module.py                 // 基础模块
                ├── evoformer_module.py             // evoformer模块
                ├── structure_module.py             // 结构模块
            ├── data
                ├── feature
                    ├── data_transforms.py              //msa和template数据处理
                    ├── feature_extraction.py           //msa和template特征提取
                ├── tools
                    ├── data_process.py                 // 搜索msa和template
                    ├── data_tools.py                   // 数据处理脚本
                    ├── mmcif_parsing.py                // mmcif解析脚本
                    ├── msa_search.sh                   // mmseqs2搜索msa的shell脚本
                    ├── parsers.py                      // 解析文件脚本
                    ├── templates.py                    // 模板搜索脚本
            ├── config
                ├── config.py                           //参数配置脚本
            ├── common
                ├── generate_pdb.py                     // 生成pdb
                ├── r3.py                               // 3D坐标转换
                ├── residue_constants.py                // 氨基酸残基常量
                ├── utils.py                            // 功能函数
```

### 脚本参数

```bash
用法：run.py [--input_fasta_path INPUT_PATH][--msa_result_path MSA_RESULT_PATH]
             [--database_dir DATABASE_PATH][--database_envdb_dir DATABASE_ENVDB_PATH]
             [--hhsearch_binary_path HHSEARCH_PATH][--pdb70_database_path PDB70_PATH]
             [--template_mmcif_dir TEMPLATE_PATH][--max_template_date TRMPLATE_DATE]
             [--kalign_binary_path KALIGN_PATH][--obsolete_pdbs_path OBSOLETE_PATH]


选项：
  --input_fasta_path       FASTA文件，用于预测蛋白质结构的蛋白质序列。
  --msa_result_path        保存mmseqs2检索得到的msa结果路径。
  --database_dir           搜索msa时的数据库。
  --database_envdb_dir     搜索msa时的扩展数据库。
  --hhsearch_binary_path   hhsearch可执行文件路径。
  --pdb70_database_path    供hhsearch使用的pdb70数据库路径。
  --template_mmcif_dir     具有mmcif结构模板的路径。
  --max_template_date      模板最新发布的时间
  --kalign_binary_path     kalign可执行文件路径。
  --obsolete_pdbs_path     PDB IDs的映射文件路径。
```

### 参数配置

### 推理过程

- 当前推理支持2048长度的蛋白质序列结构预测， 推理过程如下，可选择是否使用MindSpore serving，是否使用可扩展msa数据库。

```bash
#使用MindSpore serving
python

# 使用可扩展msa数据库
python run.py  --input_fasta_path=[INPUT_PATH] --database_dir=[DATABASE_PATH] --database_envdb_dir=[DATABASE_ENVDB_PATH] --hhsearch_binary_path=[HHSEARCH_PATH]
 --pdb70_database_path=[PDB70_PATH]  --template_mmcif_dir=[TEMPLATE_PATH] --kalign_binary_path=[KALIGN_PATH] --obsolete_pdbs_path=[OBSOLETE_PATH] > output.log 2>&1 &

# 使用不可扩展msa数据库
python run.py  --input_fasta_path=[INPUT_PATH] --database_dir=[DATABASE_PATH] --database_envdb_dir=“” --hhsearch_binary_path=[HHSEARCH_PATH]
 --pdb70_database_path=[PDB70_PATH]  --template_mmcif_dir=[TEMPLATE_PATH] --kalign_binary_path=[KALIGN_PATH] --obsolete_pdbs_path=[OBSOLETE_PATH] > output.log 2>&1 &

```

#### 结果

推理结果保存在 “./test_data/result” 中，共有两个文件， 其中的pdb文件即为蛋白质预测结构结果，timings文件保存了运行过程中的时间信息。

```bash
{'pre_process_time': 530.79, 'model_time': 432.21, 'pos_process_time': 0.22, 'all_time': 963.22}
```

## 推理性能

| 参数  | Fold(Ascend)                         |
| ------------------- | --------------------------- |
| 模型版本      | AlphaFold                       |
| 资源        | Ascend 910                  |
| 上传日期              | 2021-11-05                    |
| MindSpore版本   | 1.1.1-alpha                 |
| 数据集 | CASP14 T1079 |
| seq_length          |      505                     |
| confidence  | 94.62 |
| TM-score | 98.01%; |
|运行时间|345.26s|

### 预测结果对比示意图

<video src="./seq_64.mpg" width="800px" height="600px" control s="controls"></video>