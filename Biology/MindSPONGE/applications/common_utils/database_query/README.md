
# 配置数据库检索

蛋白质结构预测工具MEGA-Fold依赖多序列比对(MSA，multiple sequence alignments)与模板检索生成等传统数据库搜索工具提供的共进化与模板信息，本工具提供数据库及相关检索工具的安装教程。

配置数据库搜索需**2.5T硬盘**（推荐SSD）和与Kunpeng920性能持平的CPU。

## 配置MSA检索

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

## 配置MSA检索加速(可选)

下载MSA加速缓存工具：

- [FoldMSA.tar.gz](https://download.mindspore.cn/mindscience/mindsponge/msa_tools/Fold_MSA.tar.gz)：按照工具内说明操作进行MSA搜索加速。

## 配置模板检索

首先安装模板搜索工具[**HHsearch**](https://github.com/soedinglab/hh-suite)
与[**kalign**](https://msa.sbc.su.se/downloads/kalign/current.tar.gz)，参考`install.sh`，然后下载模板检索所需数据库：

- [pdb70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz)：压缩包19G，解压后56G
- [mmcif database](https://ftp.pdbj.org/pub/pdb/data/structures/divided/mmCIF/)： 零散压缩文件～50G，解压后～200G，需使用爬虫脚本下载，下载后需解压所有mmcif文件放在同一个文件夹内。
- [obsolete_pdbs](http://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat)：140K

*数据库下载网站均为国外网站，下载速度可能较慢，需要自行配置VPN*。

## 配置数据库检索config

根据数据库安装情况配置数据库搜索的相关配置`database_query_config.yaml`，相关参数含义如下：

``` bash
# configuration for template search
hhsearch_binary_path   HHsearch可执行文件路径
kalign_binary_path     kalign可执行文件路径
pdb70_database_path    {pdb70文件夹}/pdb70
mmcif_dir              mmcif文件夹
obsolete_pdbs_path     PDB IDs的映射文件路径
max_template_date      模板搜索截止时间，该时间点之后的模板会被过滤掉，默认值"2100-01-01"
# configuration for Multiple Sequence Alignment
mmseqs_binary          MMseqs2可执行文件路径
uniref30_path          {uniref30文件夹}/uniref30_2103_db
database_envdb_dir     {colabfold_envdb文件夹}/colabfold_envdb_202108_db
a3m_result_path        mmseqs2检索结果(msa)的保存路径，默认值"./a3m_result/"
```

## 使用示例

安装完成后可以运行`test_database_query.py`测试，该脚本包含以下测试用例，注意**使用时需将common_utils目录添加到环境变量中**，参考代码：

``` python
import sys
sys.path.append("../../common_utils/")  # 将common_utils工具路径添加到环境变量中
from database_query.protein_feature import RawFeatureGenerator

input_path = "../../model_cards/examples/MEGA-Protein/fasta/T1082-D1.fasta"
feature_generator = RawFeatureGenerator()
features = feature_generator.monomer_feature_generate(input_path, save_file_path="./test.pkl")
for k, v in features.items():
    print(k, v.shape)

```

预期输出：

```log
Tue Mar 26 17:14:29 CST 2024
./a3m_result/../../model_cards/examples/MEGA-Protein/fasta/T1082-D1/qdb exists and will be overwritten
createdb ../../model_cards/examples/MEGA-Protein/fasta/T1082-D1.fasta ./a3m_result/../../model_cards/examples/MEGA-Protein/fasta/T1082-D1/qdb

Converting sequences

Time for merging to qdb_h: 0h 0m 0s 1ms
Time for merging to qdb: 0h 0m 0s 1ms
Database type: Aminoacid
Time for processing: 0h 0m 0s 6ms
search ./a3m_result/../../model_cards/examples/MEGA-Protein/fasta/T1082-D1/qdb /mnt/nvme1/uniref/uniref30_2103_db ./a3m_result/../../model_cards/examples/MEGA-Protein/fasta/T1082-D1/res ./a3m_result/../../model_cards/examples/MEGA-Protein/fasta/T1082-D1/tmp --num-iterations 3 --db-load-mode 2 -a -s 8 -e 0.1 --max-seqs 10000

...
...
...



Time for processing: 0h 0m 0s 0ms
rmdb ./a3m_result/../../model_cards/examples/MEGA-Protein/fasta/T1082-D1/res_env_exp_realign

Time for processing: 0h 0m 0s 0ms
Tue Mar 26 17:14:50 CST 2024
aatype (75, 21)
between_segment_residues (75,)
domain_name (1,)
residue_index (75,)
seq_length (75,)
sequence (1,)
deletion_matrix_int (25, 75)
deletion_matrix_int_all_seq (25, 75)
msa (25, 75)
msa_all_seq (25, 75)
num_alignments (75,)
msa_species_identifiers_all_seq (25,)
template_aatype (6, 75, 22)
template_all_atom_masks (6, 75, 37)
template_all_atom_positions (6, 75, 37, 3)
template_domain_names (6,)
template_e_value (6, 1)
template_neff (6, 1)
template_prob_true (6, 1)
template_release_date (6,)
template_score (6, 1)
template_similarity (6, 1)
template_sequence (6,)
template_sum_probs (6, 1)
template_confidence_scores (6, 75)
```

`features`可使用`pickle.dump()`存为.pkl格式文件，可参考`application/model_cards/examples/MEGA-Protein/pkl/T1082-D1.pkl`

## 致谢

本工具使用或参考了以下开源工具：

- [AlphaFold2](https://github.com/deepmind/alphafold)
- [Biopython](https://biopython.org)
- [ColabFold](https://github.com/sokrypton/ColabFold)
- [HH Suite](https://github.com/soedinglab/hh-suite)
- [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)
- [ML Collections](https://github.com/google/ml_collections)
- [NumPy](https://numpy.org)
- [OpenMM](https://github.com/openmm/openmm)

我们感谢这些开源工具所有的贡献者和维护者！
