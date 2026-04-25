# 气象套件

## 概述

气象套件基于国产深度学习框架MindSpore开发，提供了一系列气象预测的AI解决方案。套件涵盖中期气象预测、短临降水预测、地震预警等多种应用，适用于气象预报和灾害预警场景。

## 应用案例

### 1. FourCastNet

**路径**: `Earth/MindEarth/applications/medium-range/fourcastnet/`

**功能描述**: 基于FourCastNet架构的全球气象中期预测模型，使用ERA5数据集进行训练和预测。采用自适应傅里叶神经算子(AFNO)实现高效的全球气象场预测。

**主要文件**:
- `main.py` - 主程序入口，支持训练和测试模式
- `configs/FourCastNet.yaml` - 配置文件（网格分辨率1.4°，69个特征维度）
- `src/callback.py` - 回调函数
- `src/utils.py` - 工具函数

**关键特性**:
- 自适应傅里叶神经算子
- 全球气象场预测
- 1.4°分辨率

---

### 2. ViT-KNO中期预测

**路径**: `Earth/MindEarth/applications/medium-range/koopman_vit/`

**功能描述**: 基于Vision Transformer和Koopman算子的中期气象预测模型，结合深度学习与动力系统理论。使用Koopman算子理论建模气象系统的动态演化。

**主要文件**:
- `main.py` - 主程序入口
- `configs/vit_kno_1.4.yaml` - 配置文件（backbone: ViTKNO，16层编码器）
- `src/callback.py` - 回调函数
- `src/utils.py` - 工具函数

**关键特性**:
- Vision Transformer架构
- Koopman算子理论
- 动力系统建模

---

### 3. SKNO中期预测

**路径**: `Earth/MindEarth/applications/medium-range/skno/`

**功能描述**: 基于Spherical KNO（球面Koopman神经算子）的中期气象预测，适用于球面坐标系。专门针对地球球面几何设计的神经算子。

**主要文件**:
- `main.py` - 主程序入口
- `configs/skno.yaml` - 配置文件（网格分辨率1.4°，patch_size: 4）
- `src/skno.py` - SKNO模型实现
- `src/skno_block.py` - SKNO网络块

**关键特性**:
- 球面Koopman神经算子
- 球面几何适配
- 全球气象预测

---

### 4. FuXi中期预测

**路径**: `Earth/MindEarth/applications/medium-range/fuxi/`

**功能描述**: 基于FuXi架构的高分辨率中期气象预测模型，支持0.25°高精度网格。采用Swarm Transformer架构实现高分辨率气象预测。

**主要文件**:
- `main.py` - 主程序入口
- `configs/FuXi.yaml` - 配置文件（网格分辨率0.25°，深度18层）
- `src/fuxi.py` - FuXi模型实现
- `src/fuxi_net.py` - FuXi网络结构

**关键特性**:
- 0.25°高分辨率
- Swarm Transformer架构
- 高精度预测

---

### 5. GraphCast中期预测

**路径**: `Earth/MindEarth/applications/medium-range/graphcast/`

**功能描述**: 基于图神经网络的全球中期气象预测模型，使用网格-图结构进行预测。支持降水预测（tp模式）。

**主要文件**:
- `main.py` - 主程序入口，支持降水预测
- `configs/GraphCast_1.4.yaml` - 配置文件（latent_dims: 512，processing_steps: 16）
- `src/net_with_clip.py` - 带梯度裁剪的网络
- `src/precip.py` - 降水预测模块
- `src/precip_dataset.py` - 降水数据集

**关键特性**:
- 图神经网络架构
- 网格-图结构
- 支持降水预测

---

### 6. GraphCastTp中期预测

**路径**: `Earth/MindEarth/applications/medium-range/graphcastTp/`

**功能描述**: GraphCast的降水预测变体，专门针对降水进行优化。支持内存卸载和高分辨率预测。

**主要文件**:
- `main.py` - 主程序入口，支持内存卸载
- `configs/GraphCastTp.yaml` - 配置文件（网格分辨率0.5°，tp: True）
- `src/net_with_clip.py` - 带梯度裁剪的网络
- `src/precip.py` - 降水预测模块

**关键特性**:
- 降水预测优化
- 0.5°分辨率
- 内存卸载支持

---

### 7. PreDiff短临预测

**路径**: `Earth/MindEarth/applications/nowcasting/PreDiff/`

**功能描述**: 基于扩散模型的短临降水预测系统，使用知识对齐技术提升预测准确性。结合扩散模型和物理约束进行短临降水预测。

**主要文件**:
- `main.py` - 主程序入口，支持训练和推理
- `configs/diffusion.yaml` - 配置文件（时间步1000，SEVIR数据集）
- `src/diffusion/latent_diffusion.py` - 潜在扩散模型
- `src/diffusion/cuboid_transformer.py` - Cuboid Transformer
- `src/knowledge_alignment/alignment.py` - 知识对齐算法
- `src/vae/autoencoder_kl.py` - VAE模型

**关键特性**:
- 扩散模型
- 知识对齐技术
- 短临降水预测
- SEVIR数据集

---

### 8. 地震预警（G-TEAM）

**路径**: `Earth/MindEarth/applications/earthquake/G-TEAM/`

**功能描述**: 基于Transformer的地震预警系统，支持PGA（峰值地面加速度）预测。通过地震波形数据预测地震强度。

**主要文件**:
- `main.py` - 主程序入口
- `config/GTEAM.yaml` - 配置文件（transformer_layers: 6，n_heads: 10）
- `src/models.py` - 模型定义
- `src/forcast.py` - 预测模块
- `src/data.py` - 数据处理
- `src/visual.py` - 可视化工具

**关键特性**:
- Transformer架构
- PGA预测
- 地震波形分析
- 灾害预警

---

## 技术特点

1. **架构统一**: 所有应用都采用相似的目录结构（configs/、src/、main.py）
2. **数据集**: 气象应用主要使用ERA5数据集，短临预测使用SEVIR数据集
3. **分辨率范围**: 从0.25°（高精度）到1.4°（低精度）不等
4. **训练框架**: 基于MindSpore框架，支持Ascend和GPU设备
5. **预测时长**: 中期预测支持6-120小时，短临预测支持分钟级预测

## 数据集与模型权重

### 数据集下载

| 应用 | 数据集下载地址 | 存放路径 |
|------|---------------|---------|
| FourCastNet, ViT-KNO, SKNO, GraphCast | https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/WeatherBench_1.4_69/ | 各自文件夹下的`dataset/` |
| FuXi | https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/ERA5_0_25_tiny400/ | `fuxi/dataset/` |
| GraphCastTp | https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/medium_precipitation/tiny_datasets/ | `graphcastTp/dataset/` |
| PreDiff | https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip | PreDiff目录下 |
| G-TEAM | https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/G-TEAM/ | `G-TEAM/` |

### 模型权重下载

| 应用 | ckpt下载地址 |
|------|-------------|
| FourCastNet | https://onebox.huawei.com/p/65dece19918a7e57727a7707df54471a |
| ViT-KNO | https://onebox.huawei.com/p/b4d5ba19245de3403e2ac196a849d569 |
| SKNO | https://onebox.huawei.com/p/80d631582bcaf39a7570b775c1b0384e |
| FuXi | https://onebox.huawei.com/p/5377c7ea61b6c0b797e13c5fcc0c4409 |
| GraphCast | https://onebox.huawei.com/p/1ab209b38371d5a672862d63c00b8df6 |
| GraphCastTp | https://onebox.huawei.com/p/98ef8bd3881717c2135d0bc383a39506 |
| PreDiff | https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/PreDiff/ |
| G-TEAM | https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/G-TEAM/ |

## 运行环境

- MindSpore >= 2.0
- MindEarth
- Python >= 3.7
- xarray, netCDF4

## 使用方式

```bash
# FourCastNet训练
cd Earth/MindEarth/applications/medium-range/fourcastnet
python main.py --mode train --config configs/FourCastNet.yaml

# ViT-KNO预测
cd Earth/MindEarth/applications/medium-range/koopman_vit
python main.py --mode test --config configs/vit_kno_1.4.yaml

# PreDiff短临预测
cd Earth/MindEarth/applications/nowcasting/PreDiff
python main.py --mode train --config configs/diffusion.yaml

# 地震预警
cd Earth/MindEarth/applications/earthquake/G-TEAM
python main.py --config config/GTEAM.yaml
```

## 目录结构

```
Earth/
├── README.md
├── download_datasets.sh
└── MindEarth/
    ├── applications/
    │   ├── medium-range/      # 中期预测
    │   │   ├── fourcastnet/   # FourCastNet
    │   │   ├── koopman_vit/   # ViT-KNO
    │   │   ├── skno/          # SKNO
    │   │   ├── fuxi/          # FuXi
    │   │   ├── graphcast/     # GraphCast
    │   │   └── graphcastTp/   # GraphCastTp
    │   ├── nowcasting/        # 短临预测
    │   │   └── PreDiff/       # PreDiff
    │   └── earthquake/        # 地震预警
    │       └── G-TEAM/        # G-TEAM
    └── mindearth/             # 核心模块
```
