ENGLISH|[简体中文](README_CN.md)

# ESM-IF1

The ESM-IF1 inverse folding model is built for predicting protein sequences from their backbone atom coordinates. We provide scripts here 1) to sample sequence designs for a given structure; 2) to score sequences for a given structure; 3) to train the model. The ESM-IF1 model consists of invariant geometric input processing layers followed by a sequence-to-sequence transformer. The model is also trained with span masking to tolerate missing backbone coordinates and therefore can predict sequences for partially masked structures.

![Illustration](illustration.png)

**Model and Data Availability**

| Name              | Size  | Description                                     | Model URL                                                    |
| ----------------- | ----- | ----------------------------------------------- | ------------------------------------------------------------ |
| `chain_set.jsonl` | 512MB | Backbone coordinates and sequences of CATH v4.3 | [download](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/chain_set.jsonl) |
| `splits.json`     | 197kB | CATH v4.3 dataset division                      | [download](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json) |

<details><summary>Contents</summary>

<!-- TOC -->

- [ESM-IF1](#ESM-IF1)
  - [Environment](#environment)
    - [[Hardware & Framework](#hardware--framework)]
    - [Conda environment configuration](#Conda-environment-configuration)
  - [Code Contents](#code-contents)
  - [Example](#example)
    - [ESM-IF1 training](#ESM-IF1-training)
    - [Sample sequence designs for a given structure](#Sample-sequence-designs-for-a-given-structure)
    - [Scoring sequences](#Scoring-sequences)
  - [Acknowledgement](#acknowledgement)

<!-- /TOC -->

## Environment

### Hardware & Framework

This project adopts Mindspore framework and is run on Nvidia RTX3090 and Ascend 910. This project can deploy in other hardware environments by configuring its own environment.

The version of the environment used in this project is:

mindspore-gpu 1.8.0 or mindspore-ascend 1.8.1；

python 3.7；

## Conda environment configuration

It is highly recommended to start a new conda environment.

Set up a new conda environment with required packages.

```text
conda create -n inverse python=3.7
conda activate inverse
conda install mindspore-gpu=1.8.1 cudatoolkit=11.1 -c mindspore -c conda-forge
conda install pip
pip install biotite
```

## Code and Running Examples

<details><summary><font size=4 color="blue">Code Contents</font></summary>

```bash
├── esm
    ├── illustration.png                // Model structure diagram
    ├── README_EN.md                    // ESM-IF1 README English version
    ├── README_CN.md                    // ESM-IF1 README Chinese version
    ├── src
        ├── args.json                   // Model parameter configuration
        ├── data.py                     // Data processing
        ├── features.py                 // Feature extraction scripts
        ├── gvp_encoder.py              // GVP encoder script
        ├── gvp_modules.py              // GVP module
        ├── gvp_transformer.py          // gvp_transformer module
        ├── gvp_transformer_encoder.py  // gvp_transformer encoder module
        ├── gvp_utils.py                // GVP func scripts
        ├── inspector.py                // inspector module
        ├── message_passing.py          // Message passing module
        ├── modules.py                  // Modules required for the model
        ├── multihead_attention.py      // Multi-head attention module
        ├── pretrained.py               // Pre-training scripts
        ├── transformer_decoder.py      // gvp_transformer decoder module
        ├── transformer_layer.py        // transformer layer
        ├── util.py                     // common func scripts
    ├── sample_sequences.py             // sample sequences
    ├── score_log_likelihoods.py        // Sequence scoring
    ├── train.py                        // Training script
```

</details>

### ESM-IF1 training

We can train a model with the following command:

```bash
Usage:python train.py --epochs 100

option:
--epochs        Model training algebra
```

### Sample sequence designs for a given structure

To sample sequences for a given structure in PDB or mmCIF format, use the
`sample_sequences.py` script. The input file can have either `.pdb` or
`.cif` as suffix.

For example, to sample 3 sequence designs for the golgi casein kinase structure
(PDB [5YH2](https://www.rcsb.org/structure/5yh2); [PDB Molecule of the Month
from January 2022](https://pdb101.rcsb.org/motm/265)), we can run the following
command from the `esm` directory:

```bash
Usage:python sample_sequences.py data/5YH2.pdb
    --chain C --temperature 1 --num-samples 3
    --outpath output/sampled_sequences.fasta

option:
--chain        Protein chain type
--temperature  Sample temperature
--num-samples  Number of samples
--outpath      Output path
```

The sampled sequences will be saved in a fasta format to the specified output file.

The temperature parameter controls the sharpness of the probability distribution for sequence sampling. Higher sampling temperatures yield more diverse sequences but likely with lower native sequence recovery. The default sampling temperature is 1. To optimize for native sequence recovery, we recommend sampling with low temperature such as 1e-6.

### Scoring sequences

To score the conditional log-likelihoods for sequences conditioned on a given
structure, use the `score_log_likelihoods.py` script.

For example, to score the sequences in `data/5YH2_mutated_seqs.fasta`
according to the structure in `data/5YH2.pdb`, we can run the following command from the `esm` directory:

```bash
Usage:python score_log_likelihoods.py data/5YH2.pdb \
    data/5YH2_mutated_seqs.fasta --chain C \
    --outpath output/5YH2_mutated_seqs_scores.csv \
    --pdbfile src/data/5YH2.pdb --seqfile src/data/5YH2_mutated_seqs.fasta

option:
--chain        Protein chain type
--outpath      Output path
--pdbfile      pdb file path
--seqfile      Sequence data file path
```

The conditional log-likelihoods are saved in a csv format in the specified output path.
The output values are the average log-likelihoods averaged over all amino acids in a sequence.

## Acknowledgement

ESM-IF1 referred or used following tools:

- [AlphaFold2](https://github.com/deepmind/alphafold)
- [Biotite](https://www.biotite-python.org/install.html)

We thank all the contributors and maintainers of these open source tools！
