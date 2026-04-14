#!/bin/bash

# Default file paths
default_bert_config_path="../config/run_geneformer_args.yaml"
default_dataset_path="../config/geneformer_config.yaml"

default_RANK_TABLE_FILE="../config/rank_table.json"
default_RANK_SIZE=8

# Function to display help message
display_help() {
    echo "Usage: bash run_8p.sh <bert_config_path> <dataset_path> <rank_table_file> <rank_size>"
    echo "Usage: bash run_8p.sh $default_bert_config_path $default_dataset_path $default_RANK_TABLE_FILE $default_RANK_SIZE"
    echo
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  <bert_config_path>        Path to the BERT configuration file (default: $default_bert_config_path)"
    echo "  <dataset_path>            Path to the dataset configuration file (default: $default_dataset_path)"
    echo "  <rank_table_file>         Path to the rank table file (default: $default_RANK_TABLE_FILE)"
    echo "  <rank_size>               Number of devices for distributed training (default: $default_RANK_SIZE)"
    echo
    exit 0
}


# Check if command-line arguments are provided for file paths, otherwise use defaults
bert_config_path="${1:-$default_bert_config_path}"
dataset_path="${2:-$default_dataset_path}"
RANK_TABLE_FILE="${3:-$default_RANK_TABLE_FILE}"
RANK_SIZE="${4:-$default_RANK_SIZE}"

# Check if help option is provided
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi


# Check if the BERT config file exists
if [ ! -f "$bert_config_path" ]; then
    echo "Error: BERT config file does not exist: $bert_config_path"
    exit 1
fi

# Check if the dataset config file exists
if [ ! -f "$dataset_path" ]; then
    echo "Error: Dataset config file does not exist: $dataset_path"
    exit 1
fi

# Check if the dataset config file exists
if [ ! -f "$RANK_TABLE_FILE" ]; then
    echo "Error: Dataset config file does not exist: $RANK_TABLE_FILE"
    exit 1
fi

export RANK_TABLE_FILE=$default_RANK_TABLE_FILE
export RANK_SIZE=$default_RANK_SIZE

DIR_NAME="runlog"
if [ -d "$DIR_NAME" ]; then
  echo "Directory $DIR_NAME already exists"
else
  mkdir "$DIR_NAME"
  echo "Directory $DIR_NAME has been created"
fi
for((i=0;i<${RANK_SIZE};i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python3 ./main.py --bert_config_path "$bert_config_path"  --dataset_path "$dataset_path"  --nproc 16 --data_parallel True > $DIR_NAME/train_device$i.log 2>&1 &
done
