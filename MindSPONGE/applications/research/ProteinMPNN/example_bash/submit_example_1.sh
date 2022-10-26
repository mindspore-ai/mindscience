#!/bin/bash



output_dir="PDB_monomers/example_1_outputs"

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"


python eval.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --batch_size 1
