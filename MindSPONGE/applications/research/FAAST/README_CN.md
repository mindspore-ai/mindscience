[ENGLISH](README.md)|简体中文

<details><summary>README目录</summary>

<!-- TOC -->

- [约束信息结构预测模型](#rasp)
- [NMR数据自动解析方法](#faast)
- [环境配置](#环境配置)
  - [硬件环境与框架](#硬件环境与框架)
  - [安装依赖](#安装依赖)
- [代码目录](#代码目录)
- [运行示例](#运行示例)
  - [RASP模型运行示例](#约束信息结构预测模型运行示例)
  - [FAAST方法运行示例](#FAAST-NMR数据自动解析方法运行示例)
- [运行时间](#运行时间)
- [引用](#引用)
- [致谢](#致谢)

<!-- /TOC -->

</details>

# RASP

已有的AI计算方法如MEGA-Fold/AlphaFold虽然极大地提高了预测静态蛋白质结构的准确性，但仍存在未解决的问题，例如生成动态构象和进行符合实验或先验信息的结构预测。为了解决这些问题我们在已有MEGA-Fold的基础上自研了RASP(Restraints Assisted Structure Predictor)模型，RASP模型能接受抽象或实验约束，使它能根据抽象或实验、稀疏或密集的约束生成结构。这使得RASP可用于多种应用，包括改进多结构域蛋白和msa较少的蛋白的结构预测。

## FAAST

核磁共振方法（NMR）是唯一一种以原子分辨率解析更贴近蛋白质在实际环境下的溶液态构象与动态结构的方法[1][2]，然而NMR实验数据获取与分析耗时长，平均单条蛋白需领域专家投入至少数月，其中大部分时间用于实验数据的解析和归属。现有NMR NOE谱峰数据解析方法如CARA，ARIA、CYANA等使用传统分子动力学模拟生成的结构迭代解析数据，解析速度慢，且从数据中解析出的约束信息和结构仍然需要大量专家知识，同时需要投入较长时间做进一步修正。为了提高 NMR 实验数据解析的速度和准确性，我们基于MindSpore+昇腾AI软硬件平台开发了NMR数据自动解析方法FAAST（iterative Folding Assisted peak ASsignmenT）。

RASP模型和FAAST方法是基于 MindSpore + Ascend 平台开发的，同时为了方便用户使用我们在 Google 的 Colab 布置了简单的测试用例：[FAAST_DEMO](https://colab.research.google.com/drive/1uaki0Ui1Y_gqVW7KSo838aOhXHSM3PTe?usp=sharing)。它们不仅可以利用实验限制来改进模型预测，而且还可以通过其集成功能促进和加快实验数据分析。

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
    对应的英伟达cuda版本为11.1-11.6，安装链接可以参考：cuda安装链接(<https://developer.nvidia.com/cuda-toolkit-archive>)

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
    本工具依赖hhsearch 与 kalign 等搜索工具为方便使用配置了一键安装脚本（注意该脚本需要在FAAST目录下运行）

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

该模型支持三种模式的输入：
第一种输入原始fasta序列，通过在线mmseqs检索得到MSA和template，需要将use_pkl与use_custom设为False，同时输入a3m_path 与 template_path作为保存搜索结果的路径;
第二种是输入用户提供的MSA与template文件，其中MSA为a3m格式，template为cif格式，可以由用户自行检索或者由经验知识提供；需要将use_pkl设为False 与use_custom设为True,同时输入用户提供的MSA和template路径a3m_path 与 template_path;
第三种是用提前预处理好得到的pkl文件，需要将use_pkl设为True，不需要额外输入a3m_path与template path。
pkl文件的预处理可以参考

```log
./data/protein_feature.py:monomer_feature_generate 函数
```

其主要是处理输入序列的特征信息，搜索到的msa信息以及template信息，同时为了方便使用，每次运行完一次rasp模型会在 ./pkl_file/保存对应的pkl文件。
也可以参考下载好的样例pkl文件:

```log
example/pkl/prot_name.pkl
```

**约束信息**
该模型额外需要restraints信息并通过restraints_path 传给模型。约束信息是指形如`[[1,2],...,[2,10]]`等多维二进制序列代表氨基酸对的空间位置信息，为了方便用户使用，这里输入的约束信息需要以.txt后缀形式输入。同时约束信息的来源多样，包括nmr核磁共振信息、质谱交联、荧光共振能量转移等等，这里提供了一个从pdb提取约束信息的样例脚本，用法如下。

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

推理结果保存在 `./result/`, 每条序列生成相应的 prot_name.pdb

```log
{confidence of predicted structrue :89.23, time :95.86，restraint recall :1.0}
```

![输入图片说明](A.PNG)
图A分别是原始pdb、alphafold、megaprotein、rasp 的结果，可以看出在multi-domain的蛋白上rasp模型推理得到结果更接近原始结果。

### FAAST-NMR数据自动解析方法运行示例

下载RASP模型训练好的权重:[RASP.ckpt](<https://download.mindspore.cn/mindscience/mindsponge/FAAST/checkpoint/RASP.ckpt>)，相关运行示例文件可以在[样例文件](<https://download.mindspore.cn/mindscience/mindsponge/FAAST/example/>)下载，运行以下命令启动推理。

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

该方法支持的输入形式与rasp模型相同，相较于rasp其不需要约束信息，需要化学位移表与NOESY谱峰数据。
**NOESY谱数据**：文件名以`noelist_`开头，数据需转换为numpy.ndarray形式并以.pkl文件格式存储，ndarray的shape是`[num_peak, 4]`，示例如下：

``` log
array([[3.257400e+01, 8.500000e+00, 4.878000e+00, 8.945950e+05],
       [4.557600e+01, 9.230000e-01, 3.922000e+00, 7.646940e+05],
       [3.148300e+01, 8.377000e+00, 5.340000e+00, 5.575200e+05],
       ...,
       [3.297700e+01, 3.166000e+00, 4.397000e+00, 1.694484e+06],
       [5.373800e+01, 4.392000e+00, 3.064000e+00, 1.335124e+06],
       [5.372800e+01, 4.393000e+00, 3.177000e+00, 1.281226e+06]])
```

第一列为重原子的，第二列为与重原子相连的氢原子的，第三列为与N相连的氢原子的H，第四列为峰强度（volume），若存在多个NOESY谱，需分为多个pkl文件独立存储，当前仅支持3D-NOESY谱数据。

**化学位移表**：文件名以`chemical_shift_aligned.pkl`命名，数据需转换为5个numpy.ndarray的数组并以.pkl文件格式存储，示例如下：

``` log
['HA' 'HB1' 'HB2' 'HB3' 'C' 'CA' 'CB' 'H' 'HA2' 'HA3']
['H' 'H' 'H' 'H' 'C' 'C' 'C' 'H' 'H' 'H']
[  4.09   1.49   1.49   1.49 174.1   52.3   19.2    8.62   3.95   3.95]
[2 2 2 2 2 2 2 3 3 3]
['ALA' 'ALA' 'ALA' 'ALA' 'ALA' 'ALA' 'ALA' 'GLY' 'GLY' 'GLY']
```

按顺序分别为原子名称，原子类型，化学位移，原子所属残基编号，原子所属残基类型，原子所属残基编号必须与input_path中的序列对齐。

```log
>>>>>>>>>>>>>>>>>>>>>>repeat_idx 0, contact_info_input, 42.0, confidence 84.72035603116198, contact_pred_rate_input 0.0, prot_name 5W9F,

input_file_path:  ./megaassign/iter_1/structure/5W9F_0.pdb
 2023-04-28 10:09:59.461741
final_violations:  0.0
output_file_path:  ./megaassign/iter_1/structure_relaxed/5W9F_0.pdb

[[14, 29], [22, 23], [11, 15], [20, 31], [37, 38]]
....
```

此处为log的一部分日志，FAAST会跑多轮迭代，每次迭代会运行20次rasp模型得到20个不同的pdb，repeat_idx 是指该轮迭代中第几个pdb，contact_info_input 即是有效的输入contact信息位置，confidence 是该轮迭代的pdb可信度，contact_pred_rate_input是指推理后生效的contact信息，prot name是该蛋白的名字，input_file_path 是指运行蛋白质relax的输入。

## 运行时间

![输入图片说明](B.PNG)

上图是FAAST模型和传统模型的解析时间及精度的对比，以Ascend 910 aarch64系统为例，在一台硬件驱动包已经安装好的环境，部署时间大约半个小时，单条序列FAAST模型平均运行半个小时。也就是说哪怕是零基础的小白仅用最长的一天时间就能将原本耗时几个月甚至更长的nmr数据解析完成。

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

我们感谢这些开源工具所有的贡献者和维护者
