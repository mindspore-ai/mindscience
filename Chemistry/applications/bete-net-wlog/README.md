# BETE-NET 完整训练指南

## 🚀 快速启动

### 方法1: 交互式启动器（推荐）
```bash
python start_training_with_batch.py
```

### 方法2: 直接运行完整训练
```bash
python run_full_training.py
```

## 📊 训练进度显示功能

### 🔄 实时进度条
- **每个epoch显示**: 训练和验证的实时进度条
- **损失更新**: 实时显示当前样本损失和平均损失
- **时间估算**: 每个epoch的运行时间

## ⚙️ 配置选项

### 🔧 训练参数
- **epochs**: 训练轮数（默认100）
- **display_interval**: 详细报告间隔（默认5）
- **plot_interval**: 绘图间隔（默认10）
- **learning_rate**: 学习率（CSO/CPD: 0.001, FPD: 0.0005）
- **patience**: 早停耐心值（默认20）

## 📁 输出文件

### 🏆 模型文件
- `best_cpd_model_ms.ckpt` - 最佳模型权重
- `fpd_training_state.json` - 训练状态（可恢复）

### 📊 可视化文件
- `fpd_training_progress_epoch_10.png` - 中间进度图
- `fpd_training_progress_epoch_20.png` - 中间进度图
- `fpd_final_training_results.png` - 最终结果图

### 📋 日志文件
- 终端输出包含完整训练日志
- JSON状态文件包含损失历史和配置

## 📊 结果解读

### 🎯 训练指标
- **Training Loss**: 训练集损失，应该持续下降
- **Validation Loss**: 验证集损失，用于早停和模型选择
- **Loss Ratio**: 验证/训练损失比，监控过拟合

### 📈 趋势分析
- **📉 Decreasing**: 损失在下降，训练正常
- **📈 Increasing**: 损失在上升，可能过拟合
- **➡️ Stable**: 损失稳定，可能收敛

### 🏆 最终评估
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数（越接近1越好）

# BETE-NET MindSpore 推理指南

## 📋 概述

`enhanced_inference_with_plots.py` 是一个完整的BETE-NET推理脚本，支持：

- ✅ **权重加载**：支持加载预训练模型权重
- ✅ **结果可视化**：生成与原版PyTorch相同的散点图
- ✅ **评估指标**：计算MAE、RMSE、R²指标
- ✅ **多模型支持**：支持CSO、CPD、FPD三种配置
- ✅ **完整输出**：生成图片、CSV数据和汇总报告

## 🚀 使用方法

### 基本用法

```bash
# 使用随机权重进行推理
python enhanced_inference_with_plots.py --model_type FPD

# 使用预训练权重进行推理
python enhanced_inference_with_plots.py --model_type FPD --weight_path path/to/model.ckpt

# 指定输出目录
python enhanced_inference_with_plots.py --model_type FPD --weight_path model_weights.ckpt --output_dir results_fpd
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_type` | str | FPD | 模型类型：FPD |
| `--weight_path` | str | None | 预训练权重文件路径(.ckpt) |
| `--output_dir` | str | inference_results | 结果输出目录 |

## 📈 输出文件说明

运行后会在指定目录生成以下文件：

### 1. 可视化图片
- **文件名**: `{MODEL_TYPE}_inference_results.png`
- **内容**: 三个散点图 (λ, ω_log, ω_2)
- **格式**: 高分辨率PNG (300 DPI)

### 2. 详细结果数据
- **文件名**: `{MODEL_TYPE}_detailed_results.csv`
- **内容**: 每个样本的预测值和真实值
- **用途**: 进一步分析和后处理

### 3. 汇总指标报告
- **文件名**: `{MODEL_TYPE}_summary_metrics.txt`
- **内容**: 整体和分项评估指标
- **格式**: 文本格式，易于阅读
```