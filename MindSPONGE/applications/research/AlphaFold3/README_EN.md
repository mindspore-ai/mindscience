# AlphaFold3-MindSpore

[**MindSpore Implementation of AlphaFold3**] A MindSpore-based deep learning framework implementation of AlphaFold3 inference network architecture.

> 📖 **Language**: [中文](README.md) | [English](README_EN.md)

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Reference](#reference)

## Project Overview

**Project Background**:
AlphaFold3 is a revolutionary biomolecular structure prediction model released by DeepMind in 2024, capable of predicting the three-dimensional structures of proteins, DNA, RNA, and other biological macromolecules. This project implements AlphaFold3's inference functionality based on Ascend NPU and MindSpore framework.

Model Architecture is shown below:

![AlphaFold3 Model Structure](image/af3_structure.jpg)

- **Inference Pipeline**：The workflow begins with the provision of sequence information for proteins, DNA, RNA, and ligands. This data undergoes preprocessing steps, including template search and multiple sequence alignment, before being fed into the model. Next, an embedding module encodes the input information. Subsequently, the Pairformer cycles analyze the relationships between the sequences and their structures. Following this, a diffusion module generates the 3D structures. Finally, a confidence module assigns a confidence score to the predictions, providing a measure of their reliability.
- **Biomolecular Structure Prediction**: A biomolecular structure prediction model based on the AlphaFold3 algorithm, supporting various input forms including proteins, DNA, RNA, and small molecules; enabling multi-chain inputs and predicting interactions and relative positions.
- **MindSpore Support**: Model Inference adaptation based on MindSpore.

### Hardware Requirements

- Atlas 800T A2

### Software Requirements

- Python >= 3.11
- MindSpore >= 2.5.0
- CANN = 8.0.0
- cmake >= 3.28.1

## Installation

### 1. Clone Repository

```bash
git clone https://gitee.com/mindspore/mindscience
cd mindsience/MindSPONGE/application/research/AlphaFold3
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
#`{PATH}` is the current path
export PYTHONPATH={PATH}/mindscience/MindSPONGE/src
export PYTHONPATH={PATH}/mindscience/MindChemistry
```

### 3. Installing the Software Package

Download the installation package from the link [hmmer](http://eddylab.org/software/hmmer/) , such as hmmer-3.4.tar.gz, and place it in the current directory.

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

If the file `/path/to/hmmer/bin/jackhmmer` appears, the installation is successful.

### 4. Compile

```bash
cd {PATH}/mindscience/MindSPONGE/applications/research/AlphaFold3
mkdir build
cd build
cmake ..
make
cp ./cpp.cpython-311-aarch64-linux-gnu.so ../src/alphafold
cd ..
```

Then, we need to generate data file：

```bash
python ./src/alphafold3/build_data.py
```

if you see the error 'counld not find components.cif', download the file from [wwpdb](https://files.wwpdb.org/pub/pdb/data/monomers/components.cif)，then put this file in your conda environment, `{CONDA_ENV_DIR}/lib/python3.11/site-packages/share/libcifpp`. If there is no `share/libcifpp` direction, create the direction by yourself.

### 5. Download DataBase

You can download a small test database from DeepMind [miniature_databases](https://github.com/google-deepmind/alphafold3/tree/main/src/alphafold3/test_data/miniature_databases)(Only for test，have influence to inference result!)
Download and put all the files in the same direction (No need to set `--db_dir=/PATH/TO/DB_DIR` if all the database are put in `/mindscience/MindSPONGE/applications/research/AlphaFold3/public_databases`) and rename the file like the example below:

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

If you want to seearch the full database, download the following database, and put them in the same direction(No need to set `--db_dir=/PATH/TO/DB_DIR` if all the database are put in `/mindscience/MindSPONGE/applications/research/AlphaFold3/public_databases`):

- [mmcif](https://storage.googleapis.com/alphafold-databases/v3.0/pdb_2022_09_28_mmcif_files.tar.zst)
- [BFD small](https://storage.googleapis.com/alphafold-databases/v3.0/bfd-first_non_consensus_sequences.fasta.zst)
- [MGnify](https://storage.googleapis.com/alphafold-databases/v3.0/mgy_clusters_2022_05.fa.zst)
- [PDB seqres](https://storage.googleapis.com/alphafold-databases/v3.0/pdb_seqres_2022_09_28.fasta.zst)
- [UniProt](https://storage.googleapis.com/alphafold-databases/v3.0/uniprot_all_2021_04.fa.zst)
- [uniref90](https://storage.googleapis.com/alphafold-databases/v3.0/uniref90_2022_05.fa.zst)
- [NT](https://storage.googleapis.com/alphafold-databases/v3.0/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst)
- [RFam](https://storage.googleapis.com/alphafold-databases/v3.0/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst)
- [RNACentral](https://storage.googleapis.com/alphafold-databases/v3.0/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst)

Make sure having enough space on disk:

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

Uncompressing the following database file：

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

If all the files are put under`/mindscience/MindSPONGE/applications/research/AlphaFold3/public_databases`, the setting `--db_dir=/PATH/TO/DB_DIR` can be ignored.

## Quick Start

### Input Structure

Example Input JSON:

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

### Running Pipeline

AlphaFold3 can be run with the following command（Precision: float32）.

```bash
source set_path.sh
python run_alphafold.py \
  --json_path=example_input.json \
  --output_dir=output \
  --run_data_pipeline=true \
  --run_inference=true \
  --db_dir=/PATH/TO/DB_DIR \
  --model_dir=/PATH/TO/MODEL_DIR \
  --buckets=256
```

### Parameter Introduction

- `--json_path`: Name of input json
- `--output_dir`: Output direction
- `--run_data_pipeline`: run data-pipeline or not
- `--run_inference`: run inference or not
- `--db_dir`: path to database, default `{HOME}/public_databases`
- `--model_dir`: Path to ckpt, default `{HOME}/ckpt`
- `--buckets`: setting the sequence length，Default：padding to N * 256

### Input & Output

- **JSON Input**: Contains sequence information of proteins and other molecules. Support the following types of input (same as DeepMind version): Protein, DNA, RNA, Ligand, etc. Currently, only single NPU version and the max sequence length should be smaller than 1000.

- **CIF Output**: 5 Standard protein structure files and confidence info.

```txt
└─name_in_your_json
    └─ seed-{random_seed}_sample-0      # First Sample
      │  confidence.json                # Confidence of the first sample
      │  model.cif                      # Predicted structure of the first sample
      │  summary_confidence.json        # Summary confidence of the first sample
    └─ seed-{random_seed}_sample-1      # Second Sample
    └─ seed-{random_seed}_sample-2      # Third Sample
    └─ seed-{random_seed}_sample-3      # Forth Sample
    └─ seed-{random_seed}_sample-4      # Fifth Sample
    │  {name}_confidences.json          # Confidence of the best sample
    │  {name}_data.json                 # Data json file after data-processing
    │  {name}_model.cif                 # Predicted structure of the best sample
    │  {name}_summary_confidence.json   # Summary confidence of the best sample
    │  ranking_scores.csv               # Ranking Score of all five samples, the higher of the ranking score, the higher of the confidence of the sample
```

### End of Inference

When you see the following log，the inference finished correctly：

```text
=======write output to /PATH/TO/OUTPUT/DIR/name_of_your_input==========
Done processing fold input name_of_your_input.
Done processing 1 fold inputs.
```

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The implementation of Modules including: data，structure，common, constant refers to [DeepMind](https://github.com/google-deepmind/alphafold3).
- The implementation of Modules including: model，utils are based on [MindScience](https://gitee.com/mindspore/mindscience/)

## 联系我们

If you encounter any issues or have any suggestions during use, please contact us through the following methods:

- **Gitee Repository**: [AlphaFold3](https://gitee.com/mindspore/mindscience/tree/main/MindSPONGE/applications/research/AlphaFold3)
- **Issue Tracking**: [Issue Tracking](https://gitee.com/mindspore/mindscience/issues)

## Reference

- Abramson J, Adler J, Dunger J, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3[J]. Nature, 2024, 630(8016): 493-500.
