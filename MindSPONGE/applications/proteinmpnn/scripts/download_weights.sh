#!/bin/bash

set -e

DOWNLOAD_DIR="./weights"
EXAMPLE_DIR="./examples"

if ! command -v wget &> /dev/null
then
    echo "Error: wget could not be found. Please install wget (sudo apt-get install wget)"
    exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"
wget -P "${DOWNLOAD_DIR}/ca_model_weights/" \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/ca_model_weights/v_48_002.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/ca_model_weights/v_48_010.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/ca_model_weights/v_48_020.ckpt
wget -P "${DOWNLOAD_DIR}/soluble_model_weights/" \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/soluble_model_weights/v_48_002.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/soluble_model_weights/v_48_010.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/soluble_model_weights/v_48_020.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/soluble_model_weights/v_48_030.ckpt
wget -P "${DOWNLOAD_DIR}/vanilla_model_weights/" \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/vanilla_model_weights/v_48_002.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/vanilla_model_weights/v_48_010.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/vanilla_model_weights/v_48_020.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/vanilla_model_weights/v_48_030.ckpt

wget -P "${EXAMPLE_DIR}" \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/ProteinMPNN/example_inputs.zip