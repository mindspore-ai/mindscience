#!/bin/bash

# Download script for Task1/Earth datasets and checkpoints
# Skips onebox.huawei.com URLs as requested

set -e

BASE_DIR="datasets"

# Wget options with SSL verification disabled and resume support
WGET_OPTS="--no-check-certificate"

echo "Starting dataset downloads..."

# 1. WeatherBench_1.4_69 - for fourcastnet, vitkno, skno, graphcast
# echo "Downloading WeatherBench_1.4_69 dataset..."
# wget -r -np -nH --cut-dirs=4 $WGET_OPTS -P WeatherBench_1.4_69 \
#   https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/WeatherBench_1.4_69/ || true

# Copy to each application
# for app in fourcastnet koopman_vit skno graphcast; do
#   mkdir -p "$BASE_DIR/medium-range/$app/dataset"
#   cp -r WeatherBench_1.4_69/* "$BASE_DIR/medium-range/$app/dataset/"
#   echo "Copied WeatherBench to $app"
# done

# 2. ERA5_0_25_tiny400 - for fuxi
echo "Downloading ERA5_0_25_tiny400 dataset..."
mkdir -p "$BASE_DIR/ERA5_0_25_tiny400"
wget -r -np -nH --cut-dirs=4 $WGET_OPTS -P "$BASE_DIR/ERA5_0_25_tiny400" \
  https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/ERA5_0_25_tiny400/ || true

# 3. medium_precipitation tiny_datasets - for graphcastTp
echo "Downloading medium_precipitation dataset..."
mkdir -p "$BASE_DIR/tiny_datasets"
wget -r -np -nH --cut-dirs=5 $WGET_OPTS -P "$BASE_DIR/tiny_datasets" \
  https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/medium_precipitation/tiny_datasets/ || true

# 5. G-TEAM dataset
echo "Downloading G-TEAM dataset..."
mkdir -p "$BASE_DIR/G-TEAM"
wget -r -np -nH --cut-dirs=4 $WGET_OPTS -P "$BASE_DIR/G-TEAM" \
  https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/G-TEAM/ || true

# 6. PreDiff checkpoint
echo "Downloading PreDiff checkpoint..."
mkdir -p "$BASE_DIR/PreDiff"
wget -r -np -nH --cut-dirs=4 $WGET_OPTS -P "$BASE_DIR/PreDiff" \
  https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/PreDiff/ || true

# # 4. sevir_lr.zip - for PreDiff
# echo "Downloading sevir_lr dataset..."
# wget $WGET_OPTS -O sevir_lr.zip https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip || true
# mkdir -p "$BASE_DIR/nowcasting/PreDiff/dataset"
# unzip -o sevir_lr.zip -d "$BASE_DIR/nowcasting/PreDiff/dataset/"

echo "Download complete!"
echo "Note: onebox.huawei.com checkpoints were skipped as requested"
