#!/bin/bash

path_to_PDB="examples/example_inputs/PDB_complexes/3HTN.pdb"

output_dir="examples/example_outputs/example_3_score_only_outputs"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design="A B"

python proteinmpnn_run.py \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
        --num_seq_per_target 10 \
        --sampling_temp "0.1" \
        --score_only 1 \
        --seed 37 \
        --batch_size 1
