#!/usr/bin/env bash

mkdir train_data
cd train_data

# Download output pdb data
echo downloading output pdb data
mkdir pdb
wget http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/pdb/pdb_0.tar.gz
tar xzmf pdb_0.tar.gz
mv pdb_0/* pdb/

rename _renum.pdb .pdb  pdb/*
rm -f pdb_0.tar.gz

# Download input pkl data
echo downloading input pkl data
mkdir pkl
wget http://ftp.cbi.pku.edu.cn/psp/true_structure_dataset/pkl/pkl_0.tar.gz
tar xzmf pkl_0.tar.gz
mv pkl_0/* pkl/
rm -f pkl_0.tar.gz

cd ..

# Start training
echo start training
python main.py --data_config ./config/data.yaml --model_config ./config/model.yaml --is_training True --raw_feature_dir ./train_data/pkl/ --pdb_data_dir ./train_data/pdb/