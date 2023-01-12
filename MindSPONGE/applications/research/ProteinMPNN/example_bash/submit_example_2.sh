#!/bin/bash


output_dir="PDB_complexes/example_2_outputs"

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$output_dir"/assigned_pdbs.jsonl"



python eval.py \
        --model_name "pretrained_model_020" \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --out_folder $output_dir \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --batch_size 1
