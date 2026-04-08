# VibeScienceAgent

**[简体中文](README.md)** | English

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

VibeScienceAgent is an AI-driven scientific research agent system designed for biomedicine and scientific discovery. It combines multi-agent collaboration, intelligent tool retrieval, and automated workflows to help researchers accelerate discovery, data analysis, and experimental design.

## Key features

### Multi-agent system
- **Prepare Agent**: Derives a problem description and research domain from the user task
- **Survey Agent**: Literature and background survey (runs sequentially with Prepare in the `prepare_survey` node)
- **Plan Agent**: Task planning and decomposition (the workflow parses model output tags such as `</think>` and `<solution>`)
- **Critic Agent**: Evaluation and quality control (only in the graph when `self_critic: true`)
- **Execute Agent**: Code Executing

### Tooling ecosystem
- **Science data (sciencedata)**: Resources such as DepMap, GTEx, GWAS Catalog, and more
- **Python / R / CLI**: Common analysis libraries and command-line tools
- **Retrievable tool descriptions**: Dynamically filter tools and datasets by task
- **Tool retriever**: LLM-based tool selection

### Analysis capabilities
- Single-cell RNA-seq analysis
- Genomics data processing
- Proteomics analysis
- Drug discovery and docking
- Network analysis and visualization

### Model backends
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Azure OpenAI
- Google Gemini
- Groq
- AWS Bedrock
- Ollama (local)
- Custom endpoints (OpenAI-compatible API)

## Requirements

- **OS**: Linux, macOS, Windows
- **Python**: 3.11
- **Conda**: Miniconda or Anaconda (optional but recommended)
- **RAM**: 8GB+ recommended
- **Disk**: 20GB+ (including datasets)

## Quick start

### 1. Environment

Install dependencies using the conda and pip files in this repository:

```bash
# Clone
git clone https://github.com/yourusername/MindScienceAgent.git
cd MindScienceAgent

# Create environment (adjust YAML filename to match your setup)
conda env create -f environment-your_conda_environment_name.yml
conda activate your_conda_environment_name

# Python dependencies
python -m pip install -r requirements.txt
```

Replace `your_conda_environment_name` with your actual conda environment name.

### 2. YAML configuration (models and runtime)

The project uses YAML as the main configuration entry. A template **[`vibescience.yaml`](vibescience.yaml)** lives at the repo root. Edit **`model_defaults`** (the usual YAML key; it is normalized into internal `agents_defaults`) and per-agent **`agents.*`** overrides (e.g. `api_key`, `base_url`, `model_name`, `source`). **Do not commit real secrets.**

If you omit `--config`, `VibeScienceConfig()` loads `vibescience.yaml` from the repo root when present (see [`vibescience_agent/config/config.py`](vibescience_agent/config/config.py)); otherwise built-in defaults apply.

**Survey / Semantic Scholar**: Set **`tools.paper_survey.semantic_scholar_key`** or the **`S2_API_KEY`** environment variable. Optional **`max_results`** caps per-source results for each multi-source keyword search (PubMed, arXiv, Semantic Scholar). See the `paper_survey` block in the template.

**Real runs**: From the **`MindScienceAgent` root**:

```bash
python main.py --prompt "Your task here"
```

Without `--prompt`, a built-in FASTA integration demo runs.

#### RunLogger (two files, not stdlib logging)

The app uses **RunLogger** (`vibescience_agent.utils.run_logger`) to append UTF-8 text to two files. It does **not** attach file handlers to the root logger, so third-party `logging` output does not go into these files by default.

| Key (`logging` section) | Purpose |
|-------------------------|---------|
| `run_enabled` | `true` writes files; `false` disables file logging (good for pytest / CI) |
| `agent_log_file` | Agent boundaries: per-phase input summaries and outputs |
| `debug_log_file` | Full debug: prompts, raw responses, tools, internal steps |
| `console` | Brief messages to the terminal (stderr, not stdlib logging) |
| `max_bytes` / `backup_count` | Rotate by size (similar to classic Rotating behavior) |

Default paths: `logs/vibescience_run.log` (agent), `logs/vibescience_debug.log` (debug). Legacy keys like `logging.file` may still exist in YAML for compatibility; **runtime logging** uses `agent_log_file` and `debug_log_file`.

Example with the test config:

```bash
python main.py --config tests/agent_config.yaml --prompt "Your task here"
```

Minimal model- and survey-related snippet for `vibescience.yaml`:

```yaml
global:
  path: "./data"
  timeout_seconds: 600
  self_critic: false

model_defaults:
  model_name: "qwen3-235b-a22b-instruct-2507"
  api_key: "your_api_key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  source: "Custom"

tools:
  paper_survey:
    semantic_scholar_key: ""   # or set S2_API_KEY
    max_results: 10

agents:
  plan:
    temperature: 0.7
    use_tool_retriever: true
  execute:
    temperature: 0.3
    use_tool_retriever: true
```

### 3. Basic usage

#### CLI

The entry script **[`main.py`](main.py)** sits next to **[`vibescience.yaml`](vibescience.yaml)** at the **`MindScienceAgent` repo root**. Run from that directory (with `vibescience_agent` importable):

```bash
python main.py --help
python main.py --self-critic
python main.py --prompt "Your task description"
python main.py --config tests/agent_config.yaml
```

Without `--prompt`, the built-in FASTA demo runs (same idea as `python tests/test.py`). `tests/test.py` forwards to root `main.py`.

#### Python

```python
import asyncio
from vibescience_agent.config.config import VibeScienceConfig
from vibescience_agent.workflow import ExperimentWorkflow

async def main():
    cfg = VibeScienceConfig(config_path="vibescience.yaml")
    agent = ExperimentWorkflow(config=cfg, self_critic=True, timeout_seconds=600)

    log, result = await agent.run("""
    Analyze DepMap data to find genes highly expressed in lung cancer cell lines,
    and assess their feasibility as potential drug targets.
    """)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Jupyter

Top-level `await` requires an **`async def` cell** or `nest_asyncio`. Below uses `asyncio.run` to avoid top-level `await`:

```python
import asyncio
from vibescience_agent.config.config import VibeScienceConfig
from vibescience_agent.workflow import ExperimentWorkflow

cfg = VibeScienceConfig(config_path="vibescience.yaml")
agent = ExperimentWorkflow(config=cfg, self_critic=True)

async def run_task():
    log, result = await agent.run("""
Design a CRISPR screen to identify genes regulating T cell exhaustion.
Output 32 genes that maximize perturbation effect.
""")
    print(result)

asyncio.run(run_task())
```

## Detailed guide

### Configuration

Put model settings in YAML; constructor arguments override at runtime:

```python
from vibescience_agent.config.config import VibeScienceConfig
from vibescience_agent.workflow import ExperimentWorkflow

cfg = VibeScienceConfig(config_path="vibescience.yaml")

agent = ExperimentWorkflow(
    config=cfg,
    path="./data",               # overrides global.path
    timeout_seconds=900,         # overrides global.timeout_seconds
    self_critic=True,            # overrides global.self_critic
    test_time_scale_round=1,     # overrides global.test_time_scale_round
)
```

### Models

Configure models in `vibescience.yaml` instead of passing `llm` / `source` / `api_key` / `base_url` only in Python.

#### OpenAI-compatible (example)

```yaml
model_defaults:
  model_name: "gpt-4o"
  api_key: "your_openai_key"
  base_url: "https://api.openai.com/v1"
  source: "Custom"
```

#### Custom OpenAI-compatible (example)

```yaml
model_defaults:
  model_name: "qwen3-235b-a22b-instruct-2507"
  api_key: "your_api_key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  source: "Custom"
```

### Workflow modes

#### Standard (no self-critic)

```python
cfg = VibeScienceConfig(config_path="vibescience.yaml")
agent = ExperimentWorkflow(config=cfg, self_critic=False)
```

Graph outline (see [`experiment_workflow.py`](vibescience_agent/workflow/experiment_workflow.py)):

1. **`prepare_survey`**: Prepare Agent, then Survey Agent (once at the start).
2. **`plan`**: Plan Agent uses messages and survey results; if `<solution>` is parsed, the run ends; if a `</think>` block is parsed, flow goes to Execute; otherwise Plan may retry.
3. **`execute`**: Execute Agent executes code and it is executed; execution loops back to **`plan`** until completion.

#### Self-critic mode

```python
cfg = VibeScienceConfig(config_path="vibescience.yaml")
agent = ExperimentWorkflow(config=cfg, self_critic=True)
```

On top of standard mode, when `</think>` appears, Plan may route to **`self_critic`**; after Critic passes, flow goes to **Execute**, otherwise back to **Plan**. The Code–Execute loop is similar to standard mode.

## Example tasks

### 1. Gene expression

```python
log, result = await agent.run("""
Using GTEx data, find genes specifically highly expressed in brain tissue.
Run differential expression with DESeq2 and produce a volcano plot.
""")
```

### 2. Drug targets

```python
log, result = await agent.run("""
Combine DepMap CRISPR and BindingDB to find small-molecule inhibitors
for high-dependency genes and discuss drug-development potential.
""")
```

### 3. Networks

```python
log, result = await agent.run("""
Build a protein–protein interaction network, identify hub proteins,
and run WGCNA module analysis.
""")
```

### 4. Literature survey

```python
log, result = await agent.run("""
Survey recent T cell exhaustion research: mechanisms and therapeutic targets.
""")
```

### 5. Experimental design

```python
log, result = await agent.run("""
Design a single-cell RNA-seq experiment comparing immune composition
in healthy vs disease samples: prep, sequencing depth, and analysis.
""")
```

### 6. CRISPR screen design

```python
log, result = await agent.run("""
Plan a CRISPR screen to identify genes that regulate T cell exhaustion,
measured by the change in T cell receptor (TCR) signaling between acute
(interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions.
Generate 32 genes that maximize the perturbation effect.
""")
```

## Repository layout

```
MindScienceAgent/
├── main.py                 # CLI entry (same folder as vibescience.yaml)
├── vibescience.yaml            # Default config template (optional file)
├── vibescience_agent/
│   ├── agents/                 # prepare / survey / plan / critic / execute
│   ├── workflow/               # ExperimentWorkflow, BaseWorkflow, ...
│   ├── config/                 # VibeScienceConfig
│   ├── model/                  # BaseModel & ModelFactory
│   ├── tools/                  # Tools, registry, retriever
│   └── utils/                  # RunLogger, env_desc, literature, ...
├── tests/                      # pytest; tests/test.py forwards to main.py
├── requirements.txt
├── environment-your_conda_environment_name.yml
├── README.md                   # Chinese
└── README.en.md                # This file (English)
```

## Advanced

### Execution timeout

```python
cfg = VibeScienceConfig(config_path="vibescience.yaml")
agent = ExperimentWorkflow(config=cfg, timeout_seconds=1200)  # 20 minutes
```

### Code execution modes

#### Python

```python
log, result = await agent.run("""
Use pandas to load data and run statistical analysis.
""")
```

#### R

```python
log, result = await agent.run("""
#!R
library(DESeq2)
# R code...
""")
```

#### Bash

```python
log, result = await agent.run("""
#!BASH
#!/bin/bash
# Bash script...
""")
```

#### CLI

```python
log, result = await agent.run("""
#!CLI
samtools view input.bam | head -n 10
""")
```

## Troubleshooting

### Common issues

1. **Conda missing**  
   Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

2. **Model misconfiguration**  
   Check YAML: `model_defaults` (normalized to `agents_defaults`) and `agents.<prepare|survey|plan|execute>` for `model_name`, `api_key`, `base_url`, `source`.

3. **Survey / Semantic Scholar errors**  
   Set `tools.paper_survey.semantic_scholar_key` or the **`S2_API_KEY`** environment variable.

4. **Out of memory**  
   Lower `timeout_seconds` or use a smaller model.

5. **Import errors**  
   ```bash
   conda deactivate
   conda activate your_conda_environment_name
   python -m pip install -r requirements.txt
   ```

6. **Science data paths**  
   Set `global.use_sciencedata: true` to sync from S3 (large download, needs network); default is `false`.
   ```python
   import os
   print(os.path.exists("./data/sciencedata"))
   ```

### Debugging and logs

Use **RunLogger** (table above): keep `run_enabled: true` and inspect **`debug_log_file`** for full prompts and responses; **`agent_log_file`** for agent I/O boundaries.

Application code does **not** use stdlib `logging` for these files. If you call `logging.basicConfig` or attach root handlers, third-party libraries may still log elsewhere or to stderr.

## Datasets

With `global.use_sciencedata: true`, the workflow checks `sciencedata` under your configured `global.path` and downloads missing objects from the public S3 bucket (requires network). Default is off to save disk and bandwidth.

The project documents 76+ biomedical datasets. Examples:

### Cancer (DepMap)
- `DepMap_CRISPRGeneDependency.csv` — gene dependency scores
- `DepMap_CRISPRGeneEffect.csv` — CRISPR effect estimates
- `DepMap_Model.csv` — model / cell line metadata
- `DepMap_OmicsExpressionProteinCodingGenesTPMLogp1.csv` — expression (TPM)

### Expression
- `gtex_tissue_gene_tpm.parquet` — GTEx tissue TPM
- `czi_census_datasets_v4.parquet` — CZI cell census

### Protein interactions
- `affinity_capture-ms.parquet`, `co-fractionation.parquet`, `two-hybrid.parquet`, `proximity_label-ms.parquet`

### Drugs
- `BindingDB_All_202409.tsv`
- `broad_repurposing_hub_molecule_with_smiles.parquet`
- `broad_repurposing_hub_phase_moa_target_info.parquet`

### Gene–disease
- `DisGeNET.parquet`, `omim.parquet`, `kg.csv`

### Genomics
- `gwas_catalog.pkl`, `genebass_*`, `variant_table.parquet`

### Regulation
- `miRDB_v6.0_results.parquet`, `miRTarBase_microRNA_target_interaction.parquet`

### MSigDB gene sets
- `msigdb_human_c1_positional_geneset.parquet` through `msigdb_human_h_hallmark_geneset.parquet`

### Immunology & virology
- `McPAS-TCR.parquet`
- `Virus-Host_PPI_P-HIPSTER_2020.parquet`

Full list: [`vibescience_agent/utils/env_desc.py`](vibescience_agent/utils/env_desc.py).

## Tools and libraries (reference)

### Python (examples)
- Bioinformatics: `biopython`, `scanpy`, `scikit-bio`, `anndata`, `mudata`
- Genomics: `pysam`, `pybedtools`, `pyranges`, `pyfaidx`, `gget`
- Drug discovery: `rdkit`, `deeppurpose`, `pytdc`
- Data science: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`
- Single-cell: `scvelo`, `scrublet`, `cellxgene-census`
- Networks: `networkx`, `igraph` (install `python-igraph`), WGCNA (R)
- Viz: `plotly`

### R (via `Rscript` / subprocess)
- `DESeq2`, `edgeR`, `limma`, `clusterProfiler`, `WGCNA`, `ggplot2`, `dplyr`, `tidyr`, `readr`

### CLI
- `samtools`, `bowtie2`, `bwa`, `bedtools`, `fastqc`, `mafft`, `plink`, `gcta64`, `iqtree2`, `vina`, ...

See [`vibescience_agent/utils/env_desc.py`](vibescience_agent/utils/env_desc.py) for the full catalog.

## Tests

```bash
conda activate your_conda_environment_name
pytest tests/ -q
```

Run from the `MindScienceAgent` root when possible.

## Runtime logging (summary)

**RunLogger** writes two UTF-8 logs (see `logging` keys above). Legacy `logging.level` / `logging.file` may remain in YAML; effective behavior uses `agent_log_file`, `debug_log_file`, and `run_enabled`. For tests, set `logging.run_enabled: false` in YAML or fixtures to avoid creating `logs/` in the repo root.

## Contributing

1. Fork the repository  
2. Branch (`git checkout -b feature/AmazingFeature`)  
3. Commit (`git commit -m 'Add some AmazingFeature'`)  
4. Push (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

### Dev setup

```bash
git clone https://github.com/yourusername/MindScienceAgent.git
cd MindScienceAgent
conda env create -f environment-your_conda_environment_name.yml
conda activate your_conda_environment_name
python -m pip install -r requirements.txt
pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Open-source bioinformatics tooling authors  
- LangChain and LangGraph communities  
- Contributors and users

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/MindScienceAgent/issues)  
- **Ideas**: [GitHub Discussions](https://github.com/yourusername/MindScienceAgent/discussions)

## Roadmap

- [ ] More local model backends  
- [ ] Richer visualization  
- [ ] More dataset connectors  
- [ ] Better tool retrieval  
- [ ] Distributed execution  
- [ ] Web UI

---

**Disclaimer**: For research use. Respect dataset licenses and API terms of service.

Made with love for the scientific research community.
