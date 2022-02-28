# 目录

<!-- TOC -->

- [目录](#目录)
    - [模型描述](#模型描述)
    - [环境要求](#环境要求)
        - [硬件环境与框架](#硬件环境与框架)
        - [MMseqs2安装](#mmseqs2安装)
    - [数据准备](#数据准备)
        - [MSA所需数据库](#msa所需数据库)
        - [Template所需工具和数据](#template所需工具和数据)
            - [数据](#数据)
            - [工具](#工具)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [训练示例](#训练示例)
    - [引用](#引用)

<!-- /TOC -->

## 模型描述

蛋白质结构预测工具是利用计算机高效计算获取蛋白质空间结构的软件。该计算方法一直存在精度不足的缺陷，直至2020年谷歌DeepMind团队的[AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)【1】【2】取得CASP14比赛中蛋白质3D结构预测的榜首，才让这一缺陷得以弥补。本次开源的蛋白质结构预测推理工具模型部分与其相同，在多序列比对阶段，采用了[MMseqs2](https://www.biorxiv.org/content/10.1101/2021.08.15.456425v1.full.pdf)【3】进行序列检索，相比于原版算法端到端运算速度有2-3倍提升。

## 环境要求

### 硬件环境与框架

本代码运行基于Ascend处理器硬件环境与[MindSpore](https://www.mindspore.cn/) AI框架，当前版本需基于库上master代码（commit id: fecbb98b944f1798a99392075cbb90f2cea61fe1）[编译](https://www.mindspore.cn/install/detail?path=install/r1.5/mindspore_ascend_install_source.md&highlight=%E6%BA%90%E7%A0%81%E7%BC%96%E8%AF%91)，
MindSpore环境参见[MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)，环境安装后需要运行以下命令配置环境变量：

``` shell
export MS_DEV_ENABLE_FALLBACK=0
```

其余python依赖请参见[requirements.txt](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/mindsponge/fold/requirements.txt)。

### MMseqs2安装

MMseqs2用于生成多序列比对(multiple sequence alignments，MSA)，MMseqs2安装和使用可以参考[MMseqs2 User Guide](https://mmseqs.com/latest/userguide.pdf)，安装完成后需要运行以下命令配置环境变量：

``` shell
export PATH=$(pwd)/mmseqs/bin/:$PATH
```

## 数据准备

### MSA所需数据库

- [uniref30_2103](http://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2103.tar.gz)：375G（下载68G）
- [colabfold_envdb_202108](http://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz)：949G（下载110G）

数据处理参考[colabfold](http://colabfold.mmseqs.com)。

### Template所需工具和数据

#### 数据

- [pdb70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz)：56G(下载19G)
- [mmcif database](https://ftp.rcsb.org/pub/pdb/data/structures/divided/mmCIF/)： 206G（下载48G）
- [obsolete_pdbs](http://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat)：140K

#### 工具

- [HHsearch](https://github.com/soedinglab/hh-suite)
- [kalign](https://msa.sbc.su.se/downloads/kalign/current.tar.gz)

## 脚本说明

### 脚本及样例代码

```bash
├── mindscience
    ├── MindSPONGE
        ├── mindsponge
            ├── fold
                ├── train
                        ├── README_CN.md                            // fold 相关中文说明
                        ├── requirements.txt                        // 依赖包
                        ├── train.py                                // 主训练脚本
                        ├── model.py                                // 主模型
                        ├── src
                            ├── af_wrapcell.py                      // 自定义的训练配置
                            ├── learning_rate.py                    // 学习率策略
                        ├── module
                            ├── basic_module.py                     // 基础模块
                            ├── evoformer_module.py                 // evoformer模块
                            ├── structure_module.py                 // 结构模块
                            ├── all_atom.py                         // 原子坐标重建模块
                            ├── loss_module.py                      // 损失函数计算模块
                        ├── data
                            ├── feature
                                ├── data_transforms.py              // msa和template数据处理
                                ├── feature_extraction.py           // msa和template特征提取
                            ├── tools
                                ├── get_train_data.py               // 数据迭代器和训练label生成脚本
                                ├── data_tools.py                   // 数据处理脚本
                                ├── mmcif_parsing.py                // mmcif解析脚本
                                ├── parsers.py                      // 解析文件脚本
                                ├── templates.py                    // 模板搜索脚本
                                ├── quat_affine_np.py               // 四元数转换脚本
                                ├── r3_np.py                        // 刚体坐标转换脚本
                        ├── config
                            ├── config.py                           // 参数配置脚本
                            ├── global_config.py                    // 全局参数配置脚本
                        ├── common
                            ├── generate_pdb.py                     // 生成pdb
                            ├── r3.py                               // 3D坐标转换
                            ├── residue_constants.py                // 氨基酸残基常量
                            ├── utils.py                            // 功能函数
                            ├── stereo_chemical_props.txt           // bond常数文件
```

### 训练示例

```bash
用法：python train.py [--seq_length PADDING_SEQENCE_LENGTH]
             [--pdb_data_dir PDB_DATA_PATH] [--raw_feature_dir RAW_FEATURE_PATH]
             [--resolution_data RESOLUTION_PATH] [--extra_msa_length EXTRA_MSA_LENGTH]
             [--max_msa_clusters MAX_MSA_LENGTH] [--extra_msa_num EATRA_MSA_NUM]
             [--evo_num EVO_NUM] [--total_steps TOTAL_STEPS]
             [--loss_scale LOSS_SCALE_VALUE] [--gradient_clip GRADIENT_CLIP_VALUE]
             [--run_distribute FALSE]


选项：
  --seq_length          补零后序列长度，目前支持256/384
  --pdb_data_dir        pdb文件路径
  --raw_feature_dir     msa和template搜索得到的特征文件路径
  --resolution_data     pdb文件对应的分辨率数据路径
  --extra_msa_length    extra msa部分长度
  --max_msa_clusters    msa序列个数
  --extra_msa_num       extra msa block数目
  --evo_num             evoformer block 数目
  --total_steps         总训练步数
  --loss_scale          混合精度下loss scale系数
  --gradient_clip       梯度裁减值
  --run_distribute      是否使用分布式训练
```

> 对于分布式训练，需要提前创建JSON格式的HCCL配置文件。关于配置文件，可以参考[HCCL_TOOL](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

### 更多训练细节待补充

## 引用

[1] Jumper J, Evans R, Pritzel A, et al. Applying and improving AlphaFold at CASP14[J].  Proteins: Structure, Function, and Bioinformatics, 2021.

[2] Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold[J]. Nature, 2021, 596(7873): 583-589.

[3] Mirdita M, Ovchinnikov S, Steinegger M. ColabFold-Making protein folding accessible to all[J]. BioRxiv, 2021.
