#!/bin/bash

path_to_PDB="PDB_complexes/pdbs/3HTN.pdb"

output_dir="PDB_complexes/example_3_outputs"


chains_to_design="A B"

python eval.py \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --batch_size 1
