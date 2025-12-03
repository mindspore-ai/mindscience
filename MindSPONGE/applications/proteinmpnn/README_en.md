# ProteinMPNN

<p align="center">
  <img src="./img/github_fig.png" alt="alt text" width="1100px" align="middle"/>
</p>

## Description

ProteinMPNN is an open-source deep-learning method for sequence design on given protein backbones. It rapidly generates high-quality amino-acid sequences that fold into the desired 3-D structure, enabling a wide range of applications from monomer design to complex interface engineering as described in [the ProteinMPNN paper](https://www.science.org/doi/10.1126/science.add2187).

This repository provides a MindSpore implementation of ProteinMPNN, adapted from the original [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) repository and integrated with antibody-specific utilities from [RFantibody](https://github.com/RosettaCommons/RFantibody).

---

## Getting started / installation

Basic requirements：

```text
python >= 3.11
mindspore >= 2.7.1
CANN >= 8.2.RC1
```

To get started using ProteinMPNN, clone the MindScience repository:

```bash
git clone https://gitee.com/mindspore/mindscience.git
```

Then download model weights using the provided script into the ProteinMPNN directory:

```bash
cd mindscience/MindSPONGE/applications/proteinmpnn
bash scripts/download_models.sh
```

### Configure Python Environment

Install the requirement python packages:

```bash
pip install -r requirements.txt
```

---

### Get Example PDBs

To run the examples, we have provided some example pdb.
You'll need to unzip this:

```bash
unzip examples/example_inputs.zip -d examples/
```

---

## Usage

### Basic usages for ProteinMPNN

The following examples can be run from the repository root:

- Monomer design over a folder:

  ```bash
  bash examples/submit_example_1.sh
  ```

  Parses monomer PDBs and designs 2 sequences per target into `examples/example_outputs/example_1_outputs`.

- Complex design with selected designed chains:

  ```bash
  bash examples/submit_example_2.sh
  ```

  Parses complexes and assigns fixed chains; designs only chains `A B`.

- Single complex PDB design:

  ```bash
  bash examples/submit_example_3.sh
  ```

  Designs sequences for chains `A B` in a single PDB.

- Score-only on a PDB:

  ```bash
  bash examples/submit_example_3_score_only.sh
  ```

  Computes model scores for the native sequence without generating new sequences.

- Score-only from FASTA against a PDB:

  ```bash
  bash examples/submit_example_3_score_only_from_fasta.sh
  ```

  Evaluates how well sequences from a FASTA file match the 3-D structure by scoring them.

- Fixed/non-fixed positions:

  ```bash
  bash examples/submit_example_4.sh
  bash examples/submit_example_4_non_fixed.sh
  ```

  Fixes residues (not designed) or designs only specific residues using `helper_scripts/make_fixed_positions_dict.py`.

- Tied positions across chains:

  ```bash
  bash examples/submit_example_5.sh
  ```

  Samples the same residue identities at specified positions across multiple chains using `helper_scripts/make_tied_positions_dict.py`.

- Homooligomer design with chain tying:

  ```bash
  bash examples/submit_example_6.sh
  ```

  Uses `--homooligomer 1` to tie equivalent positions across identical chains.

- Output unconditional probabilities only:

  ```bash
  bash examples/submit_example_7.sh
  ```

  Produces per-position unconditional log-probabilities (outputs under `unconditional_probs_only`).

- Global amino-acid bias (e.g., polar bias):

  ```bash
  bash examples/submit_example_8.sh
  ```

  Generates a bias dictionary and designs with `--bias_AA_jsonl`.

- PSSM-guided design:

  ```bash
  bash examples/submit_example_pssm.sh
  ```

  Combines ProteinMPNN logits with PSSM-derived probabilities. Control the global mixture with `--pssm_multi` (0=no PSSM, 1=only PSSM) and per-residue coefficients via `helper_scripts/make_pssm_input_dict.py`. Enable bias distribution with `--pssm_bias_flag`.

### Antibody CDR Design from RFantibody

To perform CDR loop design from RFantibody, ProteinMPNN may be run on a directory of HLT-formatted .pdb files using the following command:

```bash
python proteinmpnn_interface_design.py \
    -pdbdir /path/to/inputdir \
    -outpdbdir /path/to/outputdir
```

This will design all CDR loops and will provide one sequence per input structure. There are many more arguments that may be experimented with and are explained by running:

```bash
python proteinmpnn_interface_design.py --help
```

We provide an example command with example inputs which can be run as follows:

```bash
bash examples/ab_pdb_example.sh
```

> Modified from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)  
> Original license: MIT License
