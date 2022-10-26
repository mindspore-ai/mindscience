# Inverse folding with ESM-IF1

The ESM-IF1 inverse folding model is built for predicting protein sequences
from their backbone atom coordinates. We provide scripts here 1) to sample sequence
designs for a given structure; 2) to score sequences for a given structure; 3) to train the model. The ESM-IF1 model consists of invariant geometric input processing layers followed by a sequence-to-sequence transformer. The model is also trained with span masking to tolerate missing backbone coordinates and therefore can predict sequences for partially masked structures.

![Illustration](illustration.png)

## Environment

This project is run on Nvidia RTX3090 and adopts Mindspore framework. This project can deploy in other hardware environments by configuring its own environment.

The version of the environment used in this project is:

mindspore-gpu 1.8.0；

python 3.7；

## Code organization

- src：Data processing and model scripts;
- score_log_likelihoods：score sequences for a given structure;
- sample_sequences：sample sequence design scripts for a given structure;
- train：training script for esm.

## Conda environment configuration

It is highly recommended to start a new conda environment.

To set up a new conda environment with required packages,

```text
conda create -n inverse python=3.7
conda activate inverse
conda install mindspore-gpu=1.8.1 cudatoolkit=11.1 -c mindspore -c conda-forge
conda install pip
pip install biotite
```

## Quickstart

### Sample sequence designs for a given structure

To sample sequences for a given structure in PDB or mmCIF format, use the
`sample_sequences.py` script. The input file can have either `.pdb` or
`.cif` as suffix.

For example, to sample 3 sequence designs for the golgi casein kinase structure
(PDB [5YH2](https://www.rcsb.org/structure/5yh2); [PDB Molecule of the Month
from January 2022](https://pdb101.rcsb.org/motm/265)), we can run the following
command from the `esm` directory:

```text
python sample_sequences.py data/5YH2.pdb \
    --chain C --temperature 1 --num-samples 3 \
    --outpath output/sampled_sequences.fasta
```

The sampled sequences will be saved in a fasta format to the specified output file.

The temperature parameter controls the sharpness of the probability distribution for sequence sampling. Higher sampling temperatures yield more diverse sequences but likely with lower native sequence recovery. The default sampling temperature is 1. To optimize for native sequence recovery, we recommend sampling with low temperature such as 1e-6.

### Scoring sequences

To score the conditional log-likelihoods for sequences conditioned on a given
structure, use the `score_log_likelihoods.py` script.

For example, to score the sequences in `data/5YH2_mutated_seqs.fasta`
according to the structure in `data/5YH2.pdb`, we can run the following command from the `esm` directory:

```text
python score_log_likelihoods.py data/5YH2.pdb \
    data/5YH2_mutated_seqs.fasta --chain C \
    --outpath output/5YH2_mutated_seqs_scores.csv
```

The conditional log-likelihoods are saved in a csv format in the specified output path.
The output values are the average log-likelihoods averaged over all amino acids in a sequence.

## Data split

The training data of this project is CATH v4.3 data, which can be obtained through the following link:

- [Backbone coordinates and sequences](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/chain_set.jsonl)
- [Split](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json)
