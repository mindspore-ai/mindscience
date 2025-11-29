#!/bin/bash

folder_with_pdbs="examples/example_inputs/PDB_monomers/"

output_dir="examples/example_outputs/example_7_outputs"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"

python helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python proteinmpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --num_seq_per_target 1 \
        --sampling_temp "0.1" \
        --unconditional_probs_only 1 \
        --seed 37 \
        --batch_size 1
