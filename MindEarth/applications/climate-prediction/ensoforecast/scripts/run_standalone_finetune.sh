#!/bin/bash

python main.py --config_file_path ./configs/finetune.yaml --run_mode train ;
python main.py --config_file_path ./configs/finetune.yaml --run_mode test