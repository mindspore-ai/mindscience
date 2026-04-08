---
name: pyehr
description: pyehr is a deep learning framework for Electronic Health Records (EHR) data. Use this model when you need to predict clinical outcomes, analyze patient trajectories, or perform downstream tasks on structured EHR data such as mortality prediction, length of stay estimation, or diagnosis code prediction.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# PyEHR

## Overview

PyEHR (Python for Electronic Health Records) is a deep learning framework designed for processing and analyzing structured Electronic Health Records data. It provides neural network architectures specifically designed for temporal EHR data, enabling tasks such as clinical outcome prediction, patient trajectory modeling, and healthcare analytics.

The framework typically uses recurrent architectures (LSTM, GRU) or attention-based models to process sequences of patient events, diagnoses, medications, and other clinical variables over time.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: Clinical outcome prediction - Suitable for predicting mortality, readmission, or disease progression from patient EHR data
- **Scenario 2**: Length of stay estimation - Suitable for predicting hospital length of stay based on patient admission data
- **Scenario 3**: Diagnosis code prediction - Suitable for predicting future diagnosis codes or disease categories from patient history
- **Scenario 4**: Healthcare analytics - Suitable for patient clustering, trend analysis, and population health management

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | CSV, JSON, or MIMIC-III/IV format (patient ID, timestamps, codes, values) |
| Data Size | Varies by task; typically hundreds to thousands of patient records |
| Data Source | MIMIC-III, MIMIC-IV, eICU, or custom hospital EHR systems |

#### Data Acquisition Methods

1. **MIMIC Database** - Download from https://physionet.org/content/mimiciv/ - Requires PhysioNet credentialing
2. **eICU Database** - Download from https://physionet.org/content/eicu-crd/ - Requires PhysioNet credentialing
3. **Custom EHR Data** - Prepare CSV files with patient ID, timestamp, event type, and value columns

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure data contains patient identifiers and timestamps
- **Step 2**: Map diagnosis codes to standard vocabularies (ICD-9, ICD-10, SNOMED-CT)
- **Step 3**: Handle missing values appropriately (forward fill, interpolation, or masking)
- **Step 4**: Encode categorical variables (diagnosis codes, medication codes)
- **Step 5**: Split data into train/validation/test sets while preserving patient-level splits

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

**Note**: The component table lists **Python 3.10**, while some conda examples use **python=3.11**. Match **torch_npu 2.6.0** to the Python minor version you actually use.

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 24.1.RC3 |
| CANN      | 8.2.RC1 |
| Python    | 3.10 |
| torch     | 2.6.0 |
| torch-npu | 2.6.0 |

#### Clone repository (GitCode — primary)

```bash
git clone https://gitcode.com/AI4Science/pyehr.git
cd PyEHR
```

#### Conda environment and Python dependencies

```bash
conda create -n pyehr-test python==3.11
conda activate pyehr-test
pip install -r requirements.txt
pip install torch_npu==2.6.0 decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado pyyaml numpy==1.26.4 lightning==2.5.6 lightning-utilities==0.15.2
# Apply Lightning Fabric patches described in the official README
```

**Ascend constraints:**

- Prefer **torch** and **torch_npu** builds from the README / CANN guidance rather than generic PyTorch index installs that are not paired with **torch-npu**.
- Install HDK and CANN per vendor documentation before Python dependency steps.

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU (per GitCode README) |
| Memory | 8GB+ RAM recommended for typical EHR datasets |
| Disk Space | ~500MB for code and dependencies, additional space for datasets |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Clone `https://gitcode.com/AI4Science/PyEHR.git` |
| 2 | Install HDK/CANN per README; create conda env; install dependencies |
| 3 | Prepare EHR data in required format (CSV/JSON) |
| 4 | Run inference: `python run_inference.py --input_data <path> --output <path>` |

**Optional NPU availability check** (when Ascend applies):

```bash
python -c "import torch; print(torch.__version__); print(getattr(torch, 'npu', None) and torch.npu.is_available())"
```

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Requires structured EHR data; may not work with unstructured clinical notes |
| Performance Limitations | Performance depends on data quality and preprocessing |
| Scale Limitations | Very large patient cohorts may require batch processing |
| Input Format | Requires specific data format (patient ID, timestamps, codes) |

#### Notes

- **Note 1**: PyEHR supports various model architectures (LSTM, GRU, Transformer). Check the model configuration for details.
- **Note 2 (NPU)**: Treat **https://gitcode.com/AI4Science/PyEHR** README as authoritative for the Ascend stack.
- **Note 3**: Data privacy is critical when working with EHR data. Ensure compliance with HIPAA, GDPR, and institutional policies.
- **Note 4**: Model performance may vary based on the specific EHR dataset and task. Validation on local data is recommended.

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| Default model | LSTM-based EHR predictor |
| Checkpoint | ./checkpoints/pyehr_model.pt |
| Config | ./configs/default_config.yaml |

#### Running examples (recommended path)

**Shell:**

```bash
cd /path/to/pyehr

# Run inference with pre-trained model
python run_inference.py \
    --input_data "./data/test_patients.csv" \
    --model_path "./checkpoints/pyehr_model.pt" \
    --output "./outputs/predictions.csv"

# Or use the main entry point
python main.py --mode inference --data_path "./data/test_patients.csv"
```

**Python (optional):** Import and use the model programmatically - see `run_inference.py` or `main.py` for detailed API usage.

#### Result Post-processing

- **Output files**: Generated predictions in CSV/JSON format
  - Predicted outcomes (mortality, readmission, etc.)
  - Probability scores for each prediction
  - Confidence intervals (if available)
- **Evaluation metrics**: Depending on task, may include AUC-ROC, AUC-PR, accuracy, F1-score

---

## Reference Resources

- **GitCode (primary)**: https://gitcode.com/AI4Science/PyEHR
- **Official README**: https://gitcode.com/AI4Science/PyEHR/blob/main/README.md
- **Additional reference**: https://github.com/yhzhu99/pyehr/tree/main
- **MIMIC Database**: `https://physionet.org/content/mimiciv/` - Standard EHR benchmark dataset
- **PyEHR Paper**: See GitHub repository for relevant publications