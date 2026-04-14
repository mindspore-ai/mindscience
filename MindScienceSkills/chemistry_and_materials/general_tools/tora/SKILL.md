---
name: tora
description: tora (Tool-Integrated Reasoning Agent) is a Microsoft LLM agent series for math via tool use (e.g. Python). Ascend deployment uses conda Python 3.10, vLLM 0.9.1 with VLLM_TARGET_DEVICE=empty, vllm-ascend 0.9.1rc2, CANN 8.2.RC1 + nnal, and local weights under src/outputs for scripts/infer.sh.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# ToRA

## Overview

ToRA (Tool-Integrated Reasoning Agent) is a series of Large Language Models developed by Microsoft for mathematical problem solving through interaction with external tools. The model interleaves natural language reasoning with Python code execution, using a built-in Python executor to perform calculations, symbolic computations, and verify solutions. ToRA achieves state-of-the-art performance on mathematical reasoning benchmarks by leveraging tool integration rather than relying solely on internal knowledge.

The model is available in multiple sizes (7B, 13B, 34B, 70B) and variants (ToRA and ToRA-Code), with the ToRA-Code variants being fine-tuned on code-specific data for enhanced programming and mathematical reasoning capabilities.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: Mathematical problem solving - Suitable for solving complex math problems from datasets like MATH, GSM8K, GSM-Hard, SVAMP, TabMWP, ASDiv, and MAWPS
- **Scenario 2**: Step-by-step reasoning with tool execution - Suitable for problems requiring intermediate calculations, symbolic manipulation, or verification steps
- **Scenario 3**: Code-assisted reasoning - Suitable for problems where Python code execution can verify or accelerate solution finding

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | JSON/JSONL files with problem statements and ground truth answers |
| Data Size | Single problems or batch evaluation datasets |
| Data Source | MATH, GSM8K, GSM-Hard, SVAMP, TabMWP, ASDiv, MAWPS benchmarks |

#### Data Acquisition Methods

1. **Built-in Datasets** - Use sample datasets included in the repository under `src/data/`
2. **HuggingFace Datasets** - Download standard benchmarks from HuggingFace datasets
3. **Custom JSONL** - Prepare custom JSONL files with the following format:
   ```json
   {"problem": "Solve for x: 2x + 5 = 15", "answer": "5"}
   ```

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure data is in JSONL format with "problem" and "answer" fields
- **Step 2**: Place data file in accessible directory path
- **Step 3**: Set `TOKENIZERS_PARALLELISM=false` to avoid warnings
- **Step 4**: Configure model path and dataset name in inference script

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

**Note**: The official README lists **CANN 8.2.RC1** packages (kernels / toolkit / nnal) and **torch 2.6.0 / torch_npu 2.6.0**. **HDK** is not spelled out in the table; use the driver bundle that matches CANN 8.2.RC1.

| Component | Version |
| --------- | ----------------------------- |
| HDK       | Match CANN 8.2.RC1 (per README / vendor guidance) |
| CANN      | 8.2.RC1 |
| Python    | 3.10 |
| torch     | 2.6.0 |
| torch-npu | 2.6.0 |

```bash
conda create -n tora python=3.10
conda activate tora
```

#### Clone repository

```bash
git clone https://atomgit.com/AI4Science/ToRA.git
cd ToRA
```

#### Install Python requirements (ToRA tree)

```bash
pip install packaging==22.0
pip install -r requirements.txt
```

#### Install vLLM and vllm-ascend

Build **vLLM** without a CUDA target, then install the **Ascend** plugin:

```bash
# vLLM
git clone --depth 1 --branch v0.9.1 https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vLLM Ascend
git clone --depth 1 --branch v0.9.1rc2 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -v -e .
cd ..
```

Run these from a directory **next to** `ToRA` (e.g. clone siblings `ToRA/`, `vllm/`, `vllm-ascend/`), or adjust paths.

#### CANN, PyTorch, and runtime environment

On **aarch64**, install the CANN packages from Huawei (exact file names vary by build):

- `Ascend-cann-kernels-*_8.2.RC1_linux-aarch64.run`
- `Ascend-cann-toolkit_*_8.2.RC1_linux-aarch64.run`
- `Ascend-cann-nnal_*_8.2.RC1_linux-aarch64.run` (**required** for this stack)

Then install **torch 2.6.0** and **torch_npu 2.6.0** using wheels or instructions that match your CANN build.

After installation:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# If your layout uses underscores:
# source /usr/local/Ascend/ascend_toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
```

#### Environment requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU with vllm-ascend support |
| Memory | Scales with model size (34B needs large device + host RAM) |
| Disk | Tens of GB for checkpoints (e.g. ToRA-Code-34B) |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | `conda create -n tora python=3.10` / `conda activate tora` |
| 2 | `git clone https://atomgit.com/AI4Science/ToRA.git` and `cd ToRA` |
| 3 | `pip install packaging==22.0` and `pip install -r requirements.txt` |
| 4 | Install **vLLM v0.9.1** (`VLLM_TARGET_DEVICE=empty`) and **vllm-ascend v0.9.1rc2** |
| 5 | Install **CANN 8.2.RC1** (kernels, toolkit, **nnal**), **torch** / **torch_npu** 2.6.0; `source` env scripts; set `ASCEND_RT_VISIBLE_DEVICES` |
| 6 | Download weights (below); set `MODEL_NAME_OR_PATH` in `infer.sh`; `bash scripts/infer.sh` |

---

### 3. Usage limitations and notes

#### Model limitations

| Topic | Notes |
| ----- | ----- |
| Ascend | Requires **vllm-ascend** and matching **CANN nnal** stack |
| Size | 34B / 70B may need multi-NPU or sharding |
| Context | Limited by model max length |
| Prompts | Use `--prompt_type tora` and `--use_train_prompt_format` where the script supports them |

#### Notes

- vLLM **v0.9.1** + vllm-ascend **v0.9.1rc2** as above.
- Set `TOKENIZERS_PARALLELISM=false` if you see tokenizer warnings.
- Self-consistency: use `--n_sampling` with temperature greater than 0 when supported.

---

### 4. Inference

#### Download weights

HuggingFace repo: [llm-agents/tora-code-34b-v1.0](https://huggingface.co/llm-agents/tora-code-34b-v1.0/tree/main)

Download **all files** from that page into:

```text
/path/ToRA/src/outputs/llm-agents/tora-code-34b-v1.0
```

(Replace `/path/ToRA` with your clone path.)

#### Run `infer.sh`

1. Edit **`/path/ToRA/src/scripts/infer.sh`** and set **`MODEL_NAME_OR_PATH`** to the local directory above, e.g.:

   ```text
   /path/ToRA/src/outputs/llm-agents/tora-code-34b-v1.0
   ```

2. From **`ToRA/src`**, run:

```bash
bash scripts/infer.sh
```

Configure other variables in the same script (dataset name, prompt type, etc.) as needed.

#### Alternative: HuggingFace hub id

If your stack supports loading directly from the Hub, you can point `MODEL_NAME_OR_PATH` at `llm-agents/tora-code-34b-v1.0` instead of a local folder (depends on vLLM / cache setup).

**API-style entry (if present in your tree):**

```bash
cd /path/ToRA/src
python -m infer.inference_api \
    --model_name_or_path llm-agents/tora-code-34b-v1.0 \
    --data_name math \
    --prompt_type tora
```

#### Result Post-processing

- **Output Location**: Results written to `src/outputs/` directory
- **Output Format**: JSONL files with problem, predicted answer, ground truth, and reasoning trace
- **Evaluation**: Use `scripts/eval.sh` to compute accuracy metrics

---

## Reference resources

- **AtomGit (clone URL)**: https://atomgit.com/AI4Science/ToRA
- **GitCode mirror**: https://gitcode.com/AI4Science/ToRA
- **Upstream (Microsoft)**: https://github.com/microsoft/ToRA
- **vLLM**: https://github.com/vllm-project/vllm (branch `v0.9.1`)
- **vllm-ascend**: https://github.com/vllm-project/vllm-ascend (branch `v0.9.1rc2`)
- **Weights (ToRA-Code 34B)**: https://huggingface.co/llm-agents/tora-code-34b-v1.0
- **Paper**: [ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving](https://arxiv.org/abs/2309.17452) (ICLR 2024)