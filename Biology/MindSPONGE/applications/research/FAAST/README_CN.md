[ENGLISH](README.md)|简体中文

# FAAST and RASP

已有的AI计算方法如MEGA-Fold/AlphaFold虽然极大地提高了预测静态蛋白质结构的准确性，但仍存在未解决的问题，例如生成动态构象和进行符合实验或先验信息的结构预测。为了解决这些问题我们在已有MEGA-Fold的基础上自研了RASP(Restraints Assisted Structure Predictor)模型，RASP模型能接受抽象或实验约束，使它能根据抽象或实验、稀疏或密集的约束生成结构。这使得RASP可用于多种应用，包括改进多结构域蛋白和msa较少的蛋白的结构预测。

核磁共振方法（NMR）是唯一一种以原子分辨率解析更贴近蛋白质在实际环境下的溶液态构象与动态结构的方法[1][2]，然而NMR实验数据获取与分析耗时长，平均单条蛋白需领域专家投入至少数月，其中大部分时间用于实验数据的解析和归属。现有NMR NOE谱峰数据解析方法如CARA，ARIA、CYANA等使用传统分子动力学模拟生成的结构迭代解析数据，解析速度慢，且从数据中解析出的约束信息和结构仍然需要大量专家知识，同时需要投入较长时间做进一步修正。为了提高 NMR 实验数据解析的速度和准确性，我们基于MindSpore+昇腾AI软硬件平台开发了NMR数据自动解析方法FAAST（iterative Folding Assisted peak ASsignmenT）。

方便用户快速上手,我们在 Google 的 Colab 布置了简单的测试用例：[FAAST_DEMO](https://colab.research.google.com/drive/1uaki0Ui1Y_gqVW7KSo838aOhXHSM3PTe?usp=sharing)。测试版本支持有限（序列长度，推理速度），完整功能请尝试MindSpore+Ascend平台。

更多信息请参考论文 ["Assisting and Accelerating NMR Assignment with Restrained Structure Prediction"](https://www.biorxiv.org/content/10.1101/2023.04.14.536890v1)。

<details><summary>引用我们</summary>

```bibtex
@article{Liu2023AssistingAA,
title={Assisting and Accelerating NMR Assignment with Restrainted Structure Prediction},
author={Sirui Liu and Haotian Chu and Yuantao Xie and Fangming Wu and Ningxi Ni and Chenghao Wang and Fangjing Mu and Jiachen Wei and Jun Zhang and Mengyun Chen and Junbin Li and F. Yu and Hui Fu and Shenlin Wang and Changlin Tian and Zidong Wang and Yi Qin Gao},
journal={bioRxiv},
year={2023}
}
```

</details>

<details><summary>目录</summary>

<!-- TOC -->

- [FAAST and RASP](#faast-and-rasp)
    - [环境配置](#%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE)
        - [硬件环境与框架](#%E7%A1%AC%E4%BB%B6%E7%8E%AF%E5%A2%83%E4%B8%8E%E6%A1%86%E6%9E%B6)
        - [安装依赖](#%E5%AE%89%E8%A3%85%E4%BE%9D%E8%B5%96)
    - [代码目录](#%E4%BB%A3%E7%A0%81%E7%9B%AE%E5%BD%95)
    - [运行示例](#%E8%BF%90%E8%A1%8C%E7%A4%BA%E4%BE%8B)
        - [约束信息结构预测模型运行示例](#%E7%BA%A6%E6%9D%9F%E4%BF%A1%E6%81%AF%E7%BB%93%E6%9E%84%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B%E8%BF%90%E8%A1%8C%E7%A4%BA%E4%BE%8B)
        - [FAAST-NMR数据自动解析方法运行示例](#faast-nmr%E6%95%B0%E6%8D%AE%E8%87%AA%E5%8A%A8%E8%A7%A3%E6%9E%90%E6%96%B9%E6%B3%95%E8%BF%90%E8%A1%8C%E7%A4%BA%E4%BE%8B)
            - [**运行命令**](#%E8%BF%90%E8%A1%8C%E5%91%BD%E4%BB%A4)
            - [**日志示例**](#%E6%97%A5%E5%BF%97%E7%A4%BA%E4%BE%8B)
            - [**结果对比**](#%E7%BB%93%E6%9E%9C%E5%AF%B9%E6%AF%94)
    - [引用](#%E5%BC%95%E7%94%A8)
    - [致谢](#%E8%87%B4%E8%B0%A2)

<!-- /TOC -->

</details>

## 环境配置

### 硬件环境与框架

本工具基于[MindSPONGE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE)生物计算库与[MindSpore](https://www.mindspore.cn/)AI框架开发，MindSpore 2.0及以后的版本均可运行，MindSpore安装和配置可以参考[MindSpore安装页面](https://www.mindspore.cn/install)。本工具可以在Ascend910或16G以上内存的GPU上运行，基于Ascend运行时默认调用混合精度，基于GPU运行时使用全精度计算。

### 安装依赖

- 安装MindSpore:
    下载mindspore wheel包:

    平台 | 链接
    ----------|----------
    Ascend-910平台 ARM操作系统  | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindSpore/unified/aarch64/mindspore-2.0.0rc1-cp37-cp37m-linux_aarch64.whl>
    Ascend-910平台 x86操作系统 | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindSpore/unified/x86_64/mindspore-2.0.0rc1-cp37-cp37m-linux_x86_64.whl>
    GPU平台 x86操作系统  | <https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindSpore/unified/x86_64/mindspore-2.0.0rc1-cp37-cp37m-linux_x86_64.whl>

    该版本mindspore对应昇腾硬件驱动包版本为:Ascend Data Center Solution 23.0.RC1，详细安装链接参考:<<https://www.mindspore.cn/install>
    对应的英伟达cuda版本为11.1-11.8，安装链接可以参考：cuda安装链接(<https://developer.nvidia.com/cuda-toolkit-archive>)

    安装 wheel 包

    ``` shell
    pip install mindspore*.whl
    ```

- 安装MindSPONGE:
    下载 Mindscience仓，并编译 MindSPONGE包：

    ``` shell
    git clone https://gitee.com/mindspore/mindscience.git
    cd ./mindscience/MindSPONGE/
    ```

    若在Ascend910平台

    ``` shell
    bash build.sh -e ascend -j 8
    ```

    若在GPU平台

    ``` shell
    bash build.sh -e gpu -j 8
    ```

    安装 wheel 包

    ``` shell
    pip install ./output/mindsponge*.whl
    ```

- 安装其它依赖包：
    本工具依赖hhsearch 与 kalign 等搜索工具，可通过一键安装脚本自动配置（注意该脚本需要在FAAST目录下运行）

    ``` shell
    cd ./mindscience/MindSPONGE/applications/research/FAAST
    sh ./install.sh
    ```

## 代码目录

<details><summary><font size=4 color="blue">代码目录</font></summary>

```bash
├── FAAST
    ├── main.py                            // FAAST主脚本
    ├── run_rasp.py                        // RASP主脚本
    ├── README.md                          // FAAST相关英文说明
    ├── README_CN.md                       // FAAST相关中文说明
    ├── extract_restraints.py              // 从pdb提取约束样例文件
    ├── search.py                          // ColabFold的mmseqs在线搜索
    ├── install.sh                         // 安装相关依赖的shell脚本
    ├── assign
        ├── assign.py                      //迭代指认脚本
        ├── init_assign.py                 //初始指认脚本
    ├── commons
        ├── analysis.py                    //结果分析工具
        ├── nmr_hydrogen_equivariance.txt  //氢原子简并性列表
        ├── res_constants.py               //氢原子简并性解析字典
    ├── config
        ├── data.yaml                      //数据处理参数配置
        ├── model.yaml                     //模型参数配置
    ├── data
        ├── dataset.py                     // 异步数据读取脚本
        ├── hhsearch.py                    // python封装的HHsearch工具
        ├── kalign.py                      // python封装的Kalign工具
        ├── msa_query.py                   // python封装的MSA处理工具
        ├── parsers.py                     // mmcif文件读取脚本
        ├── preprocess.py                  // 数据预处理脚本
        ├── protein_feature.py             // MSA与template特征搜索与整合脚本
        ├── templates.py                   // 模板处理脚本
        ├── utils.py                       // 数据处理所需功能函数
    ├── model
        ├── fold.py                        // RASP主模型脚本
    ├── module
        ├── evoformer.py                   // evoformer特征提取模块
        ├── fold_wrapcell.py               // 训练迭代封装模块
        ├── head.py                        // FAAST附加输出模块
        ├── structure.py                   // 3D结构生成模块
        ├── template_embedding.py          // 模板信息提取模块
    ├── nmr_relax
        ├── model  
            ├── structure_violation.py     //计算结构是否存在严重违约
            ├── utils.py                   //运行relax时的通用工具
        ├── relax
            ├── amber_minimize.py          //运行openmm relax的主教本
            ├── cleanup.py                 //清除相关进程的脚本
            ├── relax.py                   //运行relax的主脚本
            ├── utils.py                   //运行openmm relax的通用工具
```

</details>

## 运行示例

### 约束信息结构预测模型运行示例

下载RASP模型训练好的权重:[RASP.ckpt](<https://download.mindspore.cn/mindscience/mindsponge/FAAST/checkpoint/RASP.ckpt>)，相关运行示例文件可以在[样例文件](<https://download.mindspore.cn/mindscience/mindsponge/FAAST/example/>)下载，运行以下命令启动推理。

```bash
用法：python run_rasp.py --run_platform PLATFORM --use_pkl False --restraints_path RESTRAINTS_PATH
            --input_path INPUT_FILE_PATH --checkpoint_file CHECKPOINT_FILE --use_template True --use_custom False
            --a3m_path A3M_PATH --template_path TEMPLATE_PATH

选项：
--restraints_path    约束信息文件夹位置，其中单个约束信息文件以txt形式保存
--run_platform       运行平台，可选Ascend或GPU
--input_path         输入文件夹目录，可包含多个.fasta/.pkl文件
--checkpoint_file    模型权重文件路径
--use_pkl            使用pkl数据作为输入，默认False
--use_template       是否使用template信息， 默认True
--use_custom         是否使用搜索好的msa信息与template信息, 默认False
--a3m_path           搜索后保存的的a3m文件夹位置，或者直接提供的a3m文件路径位置
--template_path      搜索后保存的cif文件位夹置，或者直接提供的cif文件路径位置
```

RASP模型支持三种模式的输入:

1. 输入原始fasta序列，通过在线mmseqs检索得到MSA和template，需要将use_pkl与use_custom设为False，同时输入a3m_path与template_path作为保存搜索结果的路径;
2. 输入用户提供的MSA与template文件，其中MSA为a3m格式，template为cif格式，可以由用户自行检索或者由经验知识提供；需要将use_pkl设为False 与use_custom设为True,同时输入用户提供的MSA和template路径a3m_path 与 template_path;
3. 输入提前预处理好得到的pkl文件，需要将use_pkl设为True，不需要额外输入a3m_path与template path。

    pkl文件的预处理可以参考`./data/protein_feature.py:monomer_feature_generate`函数，该函数主要处理输入序列的特征信息，搜索到的msa信息以及template信息。为了方便使用，每次运行完一次RASP模型会在 ./pkl_file/保存对应的pkl文件。也可以参考[样例pkl文件](https://download.mindspore.cn/mindscience/mindsponge/FAAST/example/pkl/2L33.pkl)。

**约束信息**

该模型额外需要restraints信息作为输入，约束信息是指形如`[[1,2],...,[2,10]]`等多维二进制序列代表氨基酸对的空间位置信息，为了方便用户使用，这里输入的约束信息需要以.txt后缀形式输入。同时约束信息的来源多样，包括核磁共振波谱法、质谱交联等等，这里提供了一个从pdb提取约束信息的样例脚本，用法如下。

```bash
用法 python extract_restraints.py --pdb_path PDB_PATH --output_file OUTPUT_FILE
选项：
--pdb_path         提供约束信息的pdb文件
--output_file      输出约束信息的文件位置
```

以下是约束信息样例文件，每一行即是一对氨基酸的空间位置信息，每个位置信息间用一个空格隔开。

``` log
51 74
46 60
36 44
.. ..
70 46
18 68
```

推理结果保存在 `./result/`。

```log
{confidence of predicted structrue :89.23, time :95.86，restraint recall :1.0}
```

<div align=center>
<img src="./A.png" alt="FAASTresult" width="300"/>
</div>

图A分别是原始PDB、AlphaFold、MEGA-Fold、RASP 的结果，可以看出在多域蛋白6XMV上RASP模型推理得到结果更接近真实结构。

### FAAST-NMR数据自动解析方法运行示例

#### **运行命令**

下载RASP模型训练好的权重:[RASP.ckpt](<https://download.mindspore.cn/mindscience/mindsponge/FAAST/checkpoint/RASP.ckpt>)，相关运行示例文件可以在[样例文件](<https://download.mindspore.cn/mindscience/mindsponge/FAAST/example/>)下载，运行以下命令启动推理。调整迭代配置可通过修改`assign_settings.py`中相关参数实现。

```bash
用法：python main.py --run_platform PLATFORM --use_pkl True --peak_and_cs_path PEAKLIST_PATH
            --input_path INPUT_FILE_PATH --checkpoint_file CHECKPOINT_FILE --use_template True --use_custom False
            --a3m_path A3M_PATH --template_path TEMPLATE_PATH

选项：
--peak_and_cs_path   化学位移表和NOESY谱峰列表所在路径
--run_platform       运行平台，可选Ascend或GPU
--input_path         输入文件目录，可包含多个.fasta/.pkl文件
--checkpoint_file    模型权重文件路径
--use_pkl            使用pkl数据作为输入，默认False
--use_template       是否使用template信息， 默认True
--use_custom         是否使用搜索好的msa信息与template信息, 默认False
--a3m_path           搜索后保存的的a3m文件夹位置，或者直接提供的a3m文件路径位置
--template_path      搜索后保存的cif文件位夹置，或者直接提供的cif文件路径位置
```

该方法支持的输入形式与RASP模型类似，区别在于不需要约束信息，但需要化学位移表与NOESY谱峰数据，每条蛋白质序列的化学位移表与NOESY谱峰数据需存放在独立的文件夹中，文件组织形式请参考样例文件。

**NOESY谱数据**：文件名必须是以`noelist_`开头的`.txt`文件，包含四列数据，以空格符分隔，其中第一列为重原子的共振频率，第三列为与重原子相连的氢原子的共振频率，第二列为另一个氢原子的共振频率，第四列为峰强度（volume），若存在多个NOESY谱，需分为多个`.txt`文件独立存储，当前仅支持3D-NOESY谱数据。文件示例如下：

``` log
w1 w3 w2 volume
119.73 4.584 8.102 7689.0
119.73 3.058 8.102 1084.0
119.73 3.057 8.102 1084.0
119.73 7.005 8.102 317.0
120.405 8.102 7.857 945.0
......
```

**化学位移表**：文件名以`chemical_shift_aligned.txt`命名，包含以空格符分隔的五列数据，按顺序分别为原子名称，原子类型，化学位移，原子所属残基编号，原子所属残基类型，其中原子所属残基编号必须与input_path中的序列对齐。文件示例如下：

``` log
atom_name atom_type chem_shift res_idx res_type
HA H 4.584 10 HIS
HB2 H 3.058 10 HIS
HB3 H 3.057 10 HIS
HD2 H 7.005 10 HIS
CA C 56.144 10 HIS
......
```

#### **日志示例**

以下为运行日志示例，FAAST会运行多轮迭代（iteration），每次迭代会运行多次RASP模型（repeat）使用随机采样的部分约束信息计算蛋白质结构。第0次迭代仅重复一次，所得结构用于过滤初始指认所得的约束信息中较差的约束信息。第1次迭代开始每次迭代重复多次推理，结构用于NOESY峰指认（assignment），同时输出指认结构的评估（Evaluation of assignment），方法详情请参考论文方法部分。

```log
# Initial structure prediction without restraint
>>>>>>>>>>>>>>>>>>>>>>Protein name: 5W9F, iteration: 0, repeat: 0, number of input restraint pair: 0, confidence: 84.58, input restraint recall: 1.0,
Violation of structure after relaxation:  0.0

# Initial assignment
Initial assignment:
C       2L33 noelist_17169_spectral_peak_list_2.txt 4644 4626
N       2L33 noelist_17169_spectral_peak_list_1.txt 1366 1210
Filtering restraint with given structure.

......

# Structure prediction with RASP
>>>>>>>>>>>>>>>>>>>>>>Protein name: 5W9F, iteration: 8, repeat: 0, number of input restraint pair: 62, confidence: 75.21, input restraint recall: 1.0,
Violation of structure after relaxation:  0.0

>>>>>>>>>>>>>>>>>>>>>>Protein name: 5W9F, iteration: 9, repeat: 1, number of input restraint pair: 56, confidence: 65.50, input restraint recall: 1.0,
Violation of structure after relaxation:  0.0

......

# Assignment
1st calibration and calculation of new distance-bounds done (calibration factor: 6.546974e+06)
Time: 0.019391536712646484s
Violation analysis done: 664 / 4447 restraints (14.9 %) violated.
Time: 14.645306587219238s
Final calibration and calculation of new distance-bounds done (calibration factor: 5.004552e+06).
Time: 0.015628814697265625s
Partial assignment done.
Time: 15.671599626541138s

......

# Evaluation of assignment
Iteration 1:
protein name:  2L33
restraints number per residue:  31.48
long restraints number per residue:  7.67
restraints-structure coincidence rate:  0.977
long restraints structure coincidence rate:  0.9642

......
```

Protein name是该蛋白的名字。，number of input restraint pair是有效的输入的约束信息数量，confidence 是所得结构的可信度，0为完全不可信，100为非常可信，可信度与结构质量正相关（相关系数>0.65），input restraint recall是指推理所得结构与输入约束信息的符合率。long restraints是指蛋白质一级序列中残基编号距离大于等于4的残基对约束信息。

#### **结果对比**

<div align=center>
<img src="./B.png" alt="FAAST-TIME" width="600"/>
</div>

上图是FAAST方法和传统方法的解析时间及精度的对比，以ARM+Ascend910平台为例，在一台硬件驱动包已经安装好的环境，单条序列NOESY峰指认平均耗时半个小时，且解析出的约束数量与约束-结构符合率持平人工解析。

## 引用

[1] Jumper J, Evans R, Pritzel A, et al. Applying and improving AlphaFold at CASP14[J].  Proteins: Structure, Function, and Bioinformatics, 2021.

[2] Liu S, Zhang J, Chu H, et al. PSP: million-level protein sequence dataset for protein structure prediction[J]. arXiv preprint arXiv:2206.12240, 2022.

[3] Terwilliger T C, Poon B K, Afonine P V, et al. Improved AlphaFold modeling with implicit experimental information[J]. Nat Methods, 2022.

## 致谢

FAAST使用或参考了以下开源工具：

- [ARIA](<http://aria.pasteur.fr/documentation>)
- [ColabFold](https://github.com/sokrypton/ColabFold)
- [AlphaFold2](https://github.com/deepmind/alphafold)
- [Biopython](https://biopython.org)
- [HH Suite](https://github.com/soedinglab/hh-suite)
- [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)
- [ML Collections](https://github.com/google/ml_collections)
- [NumPy](https://numpy.org)
- [OpenMM](<https://github.com/openmm/openmm>)

我们感谢这些开源工具所有的贡献者和维护者！
