#!/bin/bash


output_dir="PDB_monomers/example_8_outputs"

path_for_bias=$output_dir"/bias_pdbs.jsonl"
#Adding global polar amino acid bias (Doug Tischer)

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"

python eval.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --bias_AA_jsonl $path_for_bias \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --batch_size 1
