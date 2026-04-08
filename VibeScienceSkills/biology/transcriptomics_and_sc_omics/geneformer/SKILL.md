---
name: geneformer
description: geneformer is a context-aware, attention-based deep learning model pre-trained on large-scale human single-cell transcriptome data (Genecorpus-30M). It uses rank-based encoding to represent transcriptomes and transformer architecture for gene network dynamics prediction. Use this model when you need to perform gene classification, network biology predictions, or analyze gene expression patterns in single-cell data.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# Geneformer

## Overview

Geneformer is a transformer-based deep learning model designed for gene network mapping and dynamics prediction. It was pre-trained on approximately 30 million single-cell transcriptomes (Genecorpus-30M) and uses a novel rank-based encoding scheme where genes are ranked by their expression levels within each cell. This encoding approach prioritizes genes that distinguish cell states while down-weighting housekeeping genes, and provides robustness against technical artifacts that may systematically affect absolute transcript counts.

The model architecture consists of 6 transformer encoder layers with 256 hidden dimensions, 4 attention heads, and a feedforward size of 512. It processes transcriptomes with an input size of 2048 genes (representing 93% of rank-encoded genes in Geneformer-30M).

This skill provides inference capabilities adapted for Ascend NPU using MindSpore framework, enabling users to run gene classification tasks on Huawei Ascend NPUs.

---

## When to Use

- **Gene classification**: Classify genes based on transcriptomic context, such as identifying dosage-sensitive transcription factors
- **Network biology prediction**: Predict gene network dynamics and identify key regulatory factors
- **Disease target discovery**: Discover potential therapeutic targets using limited patient data through transfer learning
- **Single-cell analysis**: Analyze single-cell transcriptomes to understand cell-type specific gene expression patterns
- **Cardiomyopathy prediction**: Apply pre-trained models for disease-specific classification tasks

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | Pickle files (.pkl) containing gene expression data; PyTorch model weights (.bin) |
| Data Size   | Genecorpus-30M sample (50k cells for training); Model weights ~100MB |
| Data Source | HuggingFace: ctheodoris/Genecorpus-30M and ctheodoris/Geneformer |

#### Data Acquisition Methods

1. **Genecorpus-30M Dataset**: Download from https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/gene_classification/dosage_sensitive_tfs
   - Required files: `gc-30M_sample50k.dataset`, `dosage_sensitivity_TFs.pickle`
2. **Gene Vocabulary**: Download `gc104M.pkl` from https://huggingface.co/ctheodoris/Geneformer/tree/main/geneformer
3. **Pre-trained Model**: Download `pytorch_model.bin` from https://huggingface.co/ctheodoris/Geneformer/tree/pr146_branch/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224

#### Data Preprocessing

- Download required files to the Geneformer working directory
- Use the provided `convert_weight.py` script to convert PyTorch weights to MindSpore format (.ckpt)
- Ensure the gene vocabulary file (gc104M.pkl) is placed in the src directory

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend Stack (reference)

| Component   | Version      |
| ----------- | ------------ |
| CANN        | >= 8.0.rc1   |
| Python      | >= 3.9       |
| MindSpore   | 2.3.0rc4     |
| MindNLP     | 0.3.1        |
| MindFormers | 1.1.0rc1     |

Install CANN on the host per Huawei documentation before the Python steps. Ensure MindSpore and MindNLP are installed with NPU support.

#### Clone Repository

```bash
git clone -b r0.7 https://gitee.com/mindspore/mindscience.git
cd mindscience/MindSPONGE/applications/research/Geneformer/
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Hardware    | Ascend AI Processor (NPU)                     |
| Memory      | At least 16GB RAM                              |
| Disk Space  | At least 5GB for model weights and datasets   |

#### Installation Steps

1. **Install CANN**: Follow Huawei documentation to install CANN >= 8.0.rc1
2. **Create Python environment**: Ensure Python >= 3.9
3. **Install MindSpore**: `pip install mindspore==2.3.0rc4`
4. **Install MindNLP**: `pip install mindnlp==0.3.1`
5. **Install MindFormers**: `pip install mindformers==1.1.0rc1`
6. **Prepare data**: Download and place required dataset files as described above

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type      | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Input Size          | Maximum 2048 genes per sample; genes beyond this are truncated             |
| Model Size          | 6L-30M-i2048 configuration (6 layers, 30M pre-training corpus, 2048 input) |
| Framework           | Currently optimized for MindSpore; PyTorch version available separately    |
| Training Data       | Pre-trained on non-cancer transcriptomes; cancer-specific model available  |

#### Notes

- **Model Conversion**: Use `convert_weight.py` to convert PyTorch weights to MindSpore format before inference
- **Multi-card Support**: For multi-card training, configure `rank_table.json` in the configs directory
- **Task-specific Fine-tuning**: The model can be fine-tuned for specific downstream tasks with limited data

---

### 4. Quick Start

#### Step 1: Convert Model Weights

```bash
python3 scripts/convert_weight.py --layers 6 --torch_path pytorch_model.bin --mindspore_path ./out_model/geneformer_mindspore.ckpt
```

#### Step 2: Configure and Run

Modify `config/geneformer_config.yaml` to set the `model_output` directory to the converted weights path, then run:

```bash
# Multi-card execution
cd scripts && bash run.sh

# Single-card execution
cd scripts && bash run_8p.sh
```

---

### 5. Performance Metrics

| Metric              | Value                                      |
| ------------------- | ------------------------------------------ |
| Hardware            | Ascend AI Processor                        |
| Framework Version   | MindSpore 2.3.0rc4                         |
| Dataset             | Genecorpus-30M                             |
| Model Parameters    | 6L-30M-i2048                               |
| Training Parameters | batch_size=12, steps_per_epoch=835, epochs=1 |
| Test Parameters     | batch_size=16                              |
| Optimizer           | AdamW                                      |
| Train steps/s       | 12.34                                      |
| Train runtimes      | 9.60                                       |
| Eval Accuracy       | 0.70                                       |
| Eval F1             | 0.80                                       |

---

## Reference Resources

- **Model Paper**: Transfer learning enables predictions in network biology (Nature, May 2023)
- **GitHub**: https://github.com/ctheodoris/Geneformer
- **HuggingFace Models**: https://huggingface.co/ctheodoris/Geneformer
- **Genecorpus-30M**: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M