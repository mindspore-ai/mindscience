#!/bin/bash
pwd
# Default file paths
default_bert_config_path="../config/run_geneformer_args.yaml"
default_dataset_path="../config/geneformer_config.yaml"

# Function to display help message
show_help() {
    echo "Usage: bash run.sh <bert_config_path> <dataset_path>"
    echo "Usage: bash run.sh $default_bert_config_path $default_dataset_path"
    echo
    echo "Options:"
    echo "  --bert_config_path <path>    Path to the BERT configuration file (default: $default_bert_config_path)"
    echo "  --dataset_path <path>        Path to the dataset configuration file (default: $default_dataset_path)"
    echo "  -h, --help                   Show this help message"
    echo
    echo "This script runs the Python program 'main.py' with the specified configuration files."
    echo "If no paths are provided, default values will be used."
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bert_config_path)
            bert_config_path="$2"
            shift 2
            ;;
        --dataset_path)
            dataset_path="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if command-line arguments are provided for file paths, otherwise use defaults
bert_config_path="${1:-$default_bert_config_path}"
dataset_path="${2:-$default_dataset_path}"

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

# If files exist, proceed with running the Python script
python main.py --bert_config_path "$bert_config_path" --dataset_path "$dataset_path"
