#!/bin/bash

# Usage: bash download_models.sh /path/to/download/directory
set -e

DOWNLOAD_DIR="./models"
EXAMPLE_DIR="./examples"

if ! command -v wget &> /dev/null
then
    echo "Error: wget could not be found. Please install wget (sudo apt-get install wget)"
    exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"
wget -P "${DOWNLOAD_DIR}" \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/ActiveSite_ckpt.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/Base_ckpt.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/Base_epoch8_ckpt.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/Complex_Fold_base_ckpt.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/Complex_base_ckpt.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/InpaintSeq_Fold_ckpt.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/InpaintSeq_ckpt.ckpt \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/RFdiffusion_Ab.ckpt

wget -P "${EXAMPLE_DIR}" \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/antibody_pdbs.tar.gz \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/input_pdbs.tar.gz \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/ppi_scaffolds_subset.tar.gz \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/target_folds.tar.gz \
    https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/RFdiffusion/tim_barrel_scaffold.tar.gz