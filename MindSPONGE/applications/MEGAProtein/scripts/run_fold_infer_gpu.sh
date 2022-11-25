#!/usr/bin/env bash

if [ ! -e filename ]; then
echo downloading checkpoint
wget --no-check-certificate https://download.mindspore.cn/mindscience/mindsponge/ckpts/MEGAFold/MEGA_Fold_1.ckpt
fi

# Luanch inference
echo inference start
python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --run_platform GPU --use_pkl True --input_path ./examples/pkl/ --checkpoint_path ./MEGA_Fold_1.ckpt