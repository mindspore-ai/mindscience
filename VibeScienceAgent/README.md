# VibeScienceAgent

**简体中文** | **[English](README.en.md)**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

VibeScienceAgent是一个AI驱动的科学研究代理系统，专门为生物医学和科学研究领域设计。它通过多代理协作、智能工具检索和自动化工作流，帮助研究人员加速科学发现、数据分析和实验设计。

## 🌟 主要特性

### 🤖 多代理协作系统
- **Prepare Agent**: 从用户任务中提炼问题描述与研究领域
- **Survey Agent**: 文献与背景调研（与 Prepare 在同一编排节点 `prepare_survey` 中顺序执行）
- **Plan Agent**: 任务规划与分解（工作流解析模型输出中的 `</think>`、`<solution>` 等标签）
- **Critic Agent**: 结果评估与质量控制（仅在 `self_critic: true` 时参与图）
- **Execute Agent**: 代码生成与多轮修正

### 🔧 强大的工具生态
- **生物医学数据**: 覆盖DepMap、GTEx、GWAS Catalog等常用资源
- **Python/R/CLI运行环境**: 支持科研分析常见库与命令行工具
- **可检索工具描述**: 按任务动态筛选相关工具与数据集
- **智能工具检索**: 基于 LLM 的自动工具选择

### 📊 数据分析能力
- 单细胞RNA-seq分析
- 基因组学数据处理
- 蛋白质组学分析
- 药物发现和分子对接
- 网络分析和可视化

### 🎯 灵活的模型支持
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Azure OpenAI
- Google Gemini
- Groq
- AWS Bedrock
- Ollama (本地模型)
- 自定义模型（通过兼容OpenAI API的接口）

## 📋 系统要求

- **操作系统**: Linux, macOS, Windows
- **Python**: 3.11
- **Conda**: Miniconda或Anaconda
- **内存**: 建议8GB+
- **存储**: 20GB+（包括数据集）

## 🚀 快速开始

### 1. 环境安装

请使用本仓库内的 conda + pip 文件安装依赖：

```bash
# 克隆仓库
git clone https://github.com/yourusername/MindScienceAgent.git
cd MindScienceAgent

# 创建环境
conda env create -f environment-your_conda_environment_name.yml
conda activate your_conda_environment_name

# 安装 Python 依赖
python -m pip install -r requirements.txt
```

安装完成后环境名称为 `your_conda_environment_name`。

### 2. 配置 YAML（模型与运行参数）

项目当前使用 YAML 作为主配置入口。仓库根目录提供模板 **[`vibescience.yaml`](vibescience.yaml)**：请编辑 **`model_defaults`**（YAML 中常用写法，加载时会归并为内部的 `agents_defaults`）以及各 **`agents.*`** 覆盖项（如 `api_key`、`base_url`、`model_name`、`source`），**不要将含真实密钥的配置提交到版本库**。未传 `--config` 时，`VibeScienceConfig()` 会默认加载仓库根目录下的 `vibescience.yaml`（解析逻辑见 [`vibescience_agent/config/config.py`](vibescience_agent/config/config.py)；若该文件不存在则仅使用内置默认）。

**文献调研（Survey）与 Semantic Scholar**：在 **`tools.paper_survey`** 中填写 **`semantic_scholar_key`**（或设置环境变量 **`S2_API_KEY`**）；可选 **`max_results`** 控制每次多源检索（PubMed / arXiv / Semantic Scholar）单源返回条数上限。详见模板中的 `paper_survey` 段。

**真实任务**：在 YAML 中填好模型相关字段后，在 **`MindScienceAgent` 根目录**执行 `python main.py --prompt "你的任务"`。未传 `--prompt` 时会运行内置 FASTA 集成示例。

#### RunLogger（双文件，非 stdlib logging）

应用侧使用自研 **RunLogger**（`vibescience_agent.utils.run_logger`），直接写入两个 UTF-8 文本文件，**不会**给 Python 的 root logger 添加文件 handler，因此第三方库的 `logging` 默认不会进入你的业务日志文件。

| 配置键（`logging` 段） | 作用 |
|------------------------|------|
| `run_enabled` | `true` 时写文件；`false` 时不写（适合 pytest、CI） |
| `agent_log_file` | Agent 边界：各阶段输入摘要与输出正文 |
| `debug_log_file` | 全量调试：含完整 prompt、原始响应、工具与内部步骤等 |
| `console` | 是否在终端输出简要信息（经 stderr，不经 stdlib logging） |
| `max_bytes` / `backup_count` | 单文件超限时按序号轮转（与旧 Rotating 行为类似） |

默认路径示例：`logs/vibescience_run.log`（agent）、`logs/vibescience_debug.log`（debug）。`logging.file` 等历史键仍存在于默认配置中；**实际运行**由 RunLogger 使用 `agent_log_file` 与 `debug_log_file`。

本地跑集成测试时也可使用 [`tests/agent_config.yaml`](tests/agent_config.yaml)：

```bash
python main.py --config tests/agent_config.yaml --prompt "你的任务"
```

最小可运行示例（`vibescience.yaml` 中与模型、文献检索相关的字段）：

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
    semantic_scholar_key: ""   # 或设环境变量 S2_API_KEY
    max_results: 10

agents:
  plan:
    temperature: 0.7
    use_tool_retriever: true
  execute:
    temperature: 0.3
    use_tool_retriever: true
```

### 3. 基本使用

#### 命令行入口

入口脚本 **[`main.py`](main.py)** 与 **[`vibescience.yaml`](vibescience.yaml)** 同在 **`MindScienceAgent` 仓库根目录**。在该目录下执行（需已安装依赖且能 `import vibescience_agent`）：

```bash
python main.py --help
python main.py --self-critic
python main.py --prompt "你的任务描述"
python main.py --config tests/agent_config.yaml
```

不加 `--prompt` 时，会运行内置 FASTA 集成示例（与原先 `python tests/test.py` 行为一致）。`tests/test.py` 会转发到根目录 `main.py`。

#### Python脚本

```python
import asyncio
from vibescience_agent.config.config import VibeScienceConfig
from vibescience_agent.workflow import ExperimentWorkflow

async def main():
    cfg = VibeScienceConfig(config_path="vibescience.yaml")
    agent = ExperimentWorkflow(config=cfg, self_critic=True, timeout_seconds=600)

    log, result = await agent.run("""
    分析DepMap数据，识别在肺癌细胞系中高表达的基因，
    并预测这些基因作为潜在药物靶点的可行性。
    """)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Jupyter Notebook

Notebook 中若顶层使用 `await`，需放在 **`async def` 单元**或使用 `nest_asyncio`。下面示例用 `asyncio.run` 避免顶层 `await` 问题：

```python
import asyncio
from vibescience_agent.config.config import VibeScienceConfig
from vibescience_agent.workflow import ExperimentWorkflow

cfg = VibeScienceConfig(config_path="vibescience.yaml")
agent = ExperimentWorkflow(config=cfg, self_critic=True)

async def run_task():
    log, result = await agent.run("""
设计一个CRISPR筛选实验，识别调节T细胞耗竭的基因。
生成32个最大化扰动效应的基因列表。
""")
    print(result)

asyncio.run(run_task())
```

## 📖 详细使用指南

### 配置选项

当前接口中，模型参数建议写入 YAML；构造参数主要用于运行期覆盖：

```python
from vibescience_agent.config.config import VibeScienceConfig
from vibescience_agent.workflow import ExperimentWorkflow

cfg = VibeScienceConfig(config_path="vibescience.yaml")

agent = ExperimentWorkflow(
    config=cfg,
    path="./data",               # 覆盖 YAML 的 global.path
    timeout_seconds=900,         # 覆盖 YAML 的 global.timeout_seconds
    self_critic=True,            # 覆盖 YAML 的 global.self_critic
    test_time_scale_round=1,     # 覆盖 YAML 的 global.test_time_scale_round
)
```

### 使用不同模型

通过 `vibescience.yaml` 配置模型，而不是在构造函数直接传 `llm/source/api_key/base_url`。

#### OpenAI兼容接口（示例）

```yaml
model_defaults:
  model_name: "gpt-4o"
  api_key: "your_openai_key"
  base_url: "https://api.openai.com/v1"
  source: "Custom"
```

#### 自定义 OpenAI 兼容接口（示例）

```yaml
model_defaults:
  model_name: "qwen3-235b-a22b-instruct-2507"
  api_key: "your_api_key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  source: "Custom"
```

### 工作流模式

#### 标准模式（无自我批评）

```python
cfg = VibeScienceConfig(config_path="vibescience.yaml")
agent = ExperimentWorkflow(config=cfg, self_critic=False)
```

图结构概要（见 [`experiment_workflow.py`](vibescience_agent/workflow/experiment_workflow.py)）：

1. **`prepare_survey`**：依次运行 Prepare Agent，再运行 Survey Agent（仅流程开始时执行一次）。
2. **`plan`**：Plan Agent 根据消息与调研结果输出；若解析到 `<solution>` 则结束；若解析到 `</think>` 则进入 Code；否则可能重试 Plan。
3. **`execute`**：Execute Agent 生成代码后执行；执行结束后边回到 **`plan`**，形成 Plan–Execute 循环直至结束。

#### 自我批评模式

```python
cfg = VibeScienceConfig(config_path="vibescience.yaml")
agent = ExperimentWorkflow(config=cfg, self_critic=True)
```

在标准模式基础上，Plan 在出现 `</think>` 时可路由到 **`self_critic`**；Critic 通过后进入 **Execute**，否则回到 **Plan**。Code 与 Execute 的循环与标准模式类似。

## 🧪 示例任务

### 1. 基因表达分析

```python
log, result = await agent.run("""
分析GTEx数据，找出在脑组织中特异性高表达的基因。
使用DESeq2进行差异表达分析，并生成火山图。
""")
```

### 2. 药物靶点发现

```python
log, result = await agent.run("""
结合DepMap CRISPR数据和BindingDB数据，
识别高依赖性基因的小分子抑制剂，
并评估其药物开发潜力。
""")
```

### 3. 网络分析

```python
log, result = await agent.run("""
构建蛋白质-蛋白质相互作用网络，
识别关键枢纽蛋白，
并使用WGCNA进行模块分析。
""")
```

### 4. 文献调研

```python
log, result = await agent.run("""
调研T细胞耗竭的最新研究进展，
总结关键调控机制和潜在治疗靶点。
""")
```

### 5. 实验设计

```python
log, result = await agent.run("""
设计一个单细胞RNA-seq实验，
比较健康和疾病样本的免疫细胞组成，
包括样本制备、测序深度和分析流程。
""")
```

### 6. CRISPR筛选设计

```python
log, result = await agent.run("""
Plan a CRISPR screen to identify genes that regulate T cell exhaustion,
measured by the change in T cell receptor (TCR) signaling between acute
(interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions.
Generate 32 genes that maximize the perturbation effect.
""")
```

## 📁 项目结构

```
MindScienceAgent/
├── main.py                 # 命令行入口（与 vibescience.yaml 同目录）
├── vibescience.yaml            # 默认配置模板（可不存在，则仅用内置默认）
├── vibescience_agent/
│   ├── agents/                 # prepare / survey / plan / critic / execute
│   ├── workflow/               # ExperimentWorkflow、BaseWorkflow 等
│   ├── config/                 # VibeScienceConfig
│   ├── model/                  # BaseModel 与 ModelFactory
│   ├── tools/                  # 工具实现、注册与检索
│   └── utils/                  # RunLogger、env 描述、文献调研等
├── tests/                      # pytest 与 tests/test.py CLI 转发
├── requirements.txt
├── environment-your_conda_environment_name.yml
├── README.md
└── README.en.md
```

## 🔧 高级功能

### 执行超时控制

```python
# 设置超时时间（默认600秒=10分钟）
cfg = VibeScienceConfig(config_path="vibescience.yaml")
agent = ExperimentWorkflow(config=cfg, timeout_seconds=1200)  # 20分钟
```

### 代码执行

代理支持多种代码执行方式：

#### Python代码

```python
# 直接执行 Python 代码
log, result = await agent.run("""
使用 pandas 读取数据，进行统计分析。
""")
```

#### R代码

```python
# 使用#!R标记执行R代码
log, result = await agent.run("""
#!R
library(DESeq2)
# R代码...
""")
```

#### Bash脚本

```python
# 使用#!BASH标记执行Bash脚本
log, result = await agent.run("""
#!BASH
#!/bin/bash
# Bash脚本...
""")
```

#### CLI命令

```python
# 使用#!CLI标记执行CLI命令
log, result = await agent.run("""
#!CLI
samtools view input.bam | head -n 10
""")
```

## 🐛 故障排除

### 常见问题

1. **Conda未安装**
   ```bash
   # 安装Miniconda
   # 访问: https://docs.conda.io/en/latest/miniconda.html
   ```

2. **模型配置错误**
   ```bash
   # 检查 YAML：model_defaults（或等价合并后的 agents_defaults）与各 agents.* 下的
   # model_name / api_key / base_url / source
   ```

3. **Survey 阶段 Semantic Scholar 报错**
   - 在 `tools.paper_survey.semantic_scholar_key` 填写密钥，或设置环境变量 **`S2_API_KEY`**。

4. **内存不足**
   ```python
   # 减少超时时间或使用更小的模型
   cfg = VibeScienceConfig(config_path="vibescience.yaml")
   agent = ExperimentWorkflow(config=cfg, timeout_seconds=300)
   ```

5. **包导入错误**
   ```bash
   # 重新激活环境
   conda deactivate
   conda activate your_conda_environment_name
   
   # 重新安装依赖
   python -m pip install -r requirements.txt
   ```

6. **数据访问问题**
   ```python
   # 检查数据路径（需在 YAML 的 global 中设置 use_sciencedata: true 才会从 S3 同步到本地；
   # 首次同步体积大、需网络，默认 false）
   import os
   print(os.path.exists("./data/sciencedata"))
   ```

### 调试与日志

应用侧调试请使用 **RunLogger**（见上文「RunLogger」表）：在 `vibescience.yaml` 的 `logging` 段保持 `run_enabled: true`，查看 **`debug_log_file`** 即可得到完整 prompt、原始模型响应与内部步骤；**`agent_log_file`** 侧重各 agent 输入/输出边界。

说明：业务代码**不再**通过 stdlib `logging` 写上述文件；若你对第三方库调用 `logging.basicConfig` 或给 root 加 handler，仍可能把**第三方**日志写到别处或 stderr，与 RunLogger 无关。

## 📚 可用数据集

启用 `global.use_sciencedata: true` 后，工作流会在 `sciencedata` 下检查并下载缺失文件（需可访问 S3 源站）。默认关闭 sciencedata 同步，避免误占磁盘与流量。

VibeScienceAgent包含76个生物医学数据集：

### 癌症研究（DepMap）
- `DepMap_CRISPRGeneDependency.csv` - 基因依赖概率估计
- `DepMap_CRISPRGeneEffect.csv` - CRISPR基因效应估计
- `DepMap_Model.csv` - 癌症模型/细胞系元数据
- `DepMap_OmicsExpressionProteinCodingGenesTPMLogp1.csv` - 基因表达数据

### 基因表达
- `gtex_tissue_gene_tpm.parquet` - GTEx组织表达谱
- `czi_census_datasets_v4.parquet` - CZI细胞普查数据集

### 蛋白质相互作用
- `affinity_capture-ms.parquet` - 亲和捕获质谱PPI
- `co-fractionation.parquet` - 共分级PPI
- `two-hybrid.parquet` - 酵母双杂交PPI
- `proximity_label-ms.parquet` - 邻近标记质谱PPI

### 药物数据
- `BindingDB_All_202409.tsv` - 蛋白质-小分子结合亲和力
- `broad_repurposing_hub_molecule_with_smiles.parquet` - 药物重定位中心分子
- `broad_repurposing_hub_phase_moa_target_info.parquet` - 药物阶段和机制信息

### 基因-疾病关联
- `DisGeNET.parquet` - 基因-疾病关联
- `omim.parquet` - 遗传疾病和相关基因
- `kg.csv` - 精准医学知识图谱

### 基因组学
- `gwas_catalog.pkl` - 全基因组关联研究结果
- `genebass_missense_LC_filtered.pkl` - 错义变异
- `genebass_pLoF_filtered.pkl` - 预测功能缺失变异
- `variant_table.parquet` - 注释的基因变异表

### 转录调控
- `miRDB_v6.0_results.parquet` - 预测的microRNA靶点
- `miRTarBase_microRNA_target_interaction.parquet` - 实验验证的miRNA-靶点相互作用

### 基因集和通路（MSigDB）
- `msigdb_human_c1_positional_geneset.parquet` - 位置基因集
- `msigdb_human_c2_curated_geneset.parquet` - 策展基因集
- `msigdb_human_c3_regulatory_target_geneset.parquet` - 调控靶点基因集
- `msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet` - 转录因子靶点
- `msigdb_human_c4_computational_geneset.parquet` - 计算基因集
- `msigdb_human_c5_ontology_geneset.parquet` - 本体基因集
- `msigdb_human_c6_oncogenic_signature_geneset.parquet` - 致癌特征基因集
- `msigdb_human_c7_immunologic_signature_geneset.parquet` - 免疫学特征基因集
- `msigdb_human_c8_celltype_signature_geneset.parquet` - 细胞类型特征基因集
- `msigdb_human_h_hallmark_geneset.parquet` - 标志基因集

### 免疫学
- `McPAS-TCR.parquet` - T细胞受体序列和特异性数据

### 病毒-宿主相互作用
- `Virus-Host_PPI_P-HIPSTER_2020.parquet` - 病毒-宿主PPI

完整数据集列表请参考`vibescience_agent/utils/env_desc.py`。

## 🔌 可用工具和库

### Python包

#### 生物信息学核心
- `biopython` - 生物计算工具
- `scanpy` - 单细胞基因表达数据分析
- `scikit-bio` - 生物信息学数据结构和算法
- `anndata` - 注释数据矩阵处理
- `mudata` - 多模态数据存储

#### 基因组学
- `pysam` - SAM/BAM/VCF/BCF格式读取
- `pybedtools` - BEDTools Python包装器
- `pyranges` - 区间操作
- `pyfaidx` - FASTA文件高效访问
- `gget` - 基因组数据库访问

#### 药物发现
- `rdkit` - 化学信息学和机器学习
- `deeppurpose` - 药物-靶点相互作用预测
- `pytdc` - 治疗数据通用

#### 数据分析
- `pandas` - 数据分析和操作
- `numpy` - 科学计算
- `scipy` - 科学和技术计算
- `scikit-learn` - 机器学习
- `matplotlib` - 可视化
- `seaborn` - 统计数据可视化
- `statsmodels` - 统计建模

#### 单细胞分析
- `scanpy` - 单细胞分析
- `scvelo` - RNA速度分析
- `scrublet` - 双细胞检测
- `cellxgene-census` - CellxGene普查访问

#### 网络分析
- `networkx` - 网络分析
- `igraph` - 网络分析和可视化
- `WGCNA` - 加权相关网络分析

#### 可视化
- `matplotlib` - 静态可视化
- `seaborn` - 统计图形
- `plotly` - 交互式可视化

### R包

#### 差异表达分析
- `DESeq2` - 基于负二项分布的差异表达分析
- `edgeR` - 数字基因表达数据的经验分析
- `limma` - 微阵列数据的线性模型

#### 功能分析
- `clusterProfiler` - 基因和基因簇的功能分析和可视化

#### 网络分析
- `WGCNA` - 加权相关网络分析

#### 可视化
- `ggplot2` - 基于图形语法的声明式图形系统

#### 数据操作
- `dplyr` - 数据操作语法
- `tidyr` - 整洁数据创建
- `readr` - 快速友好的矩形数据读取

### CLI工具

#### 序列分析
- `samtools` - 高通量测序数据处理
- `bowtie2` - 超快内存高效的序列比对
- `bwa` - 低差异序列比对
- `bedtools` - 基因组算术工具集

#### 质量控制
- `fastqc` - 高通量序列数据质量控制

#### 多序列比对
- `mafft` - 多序列比对程序

#### 遗传分析
- `plink` / `plink2` - 全基因组关联分析工具包
- `gcta64` - 全基因组复杂性状分析

#### 系统发育
- `iqtree2` - 最大似然分析系统发育软件

#### 分子对接
- `vina` - 分子对接和虚拟筛选

完整工具列表请参考`vibescience_agent/utils/env_desc.py`。

## 🧪 测试

运行完整测试：

```bash
# 激活环境
conda activate your_conda_environment_name

# 运行测试（建议在 MindScienceAgent 根目录执行）
pytest tests/ -q
```

## 📝 运行日志（摘要）

业务运行轨迹由 **RunLogger** 写入两个 UTF-8 文本文件（路径与开关见 `logging` 配置段及前文表格）。配置里仍保留 `logging.level`、`logging.file` 等历史字段以便兼容旧 YAML，**当前实现以 `agent_log_file` / `debug_log_file` 与 `run_enabled` 为准**。测试建议在 YAML 或测试夹具中设置 `logging.run_enabled: false`，避免在仓库根目录产生 `logs/` 文件。

## 🤝 贡献指南

我们欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/MindScienceAgent.git
cd MindScienceAgent

# 安装开发环境
conda env create -f environment-your_conda_environment_name.yml
conda activate your_conda_environment_name
python -m pip install -r requirements.txt

# 运行测试（建议在 MindScienceAgent 根目录执行）
pytest tests/ -q
```

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

## 🙏 致谢

- 感谢所有开源生物信息学工具的开发者
- 感谢LangChain和LangGraph社区
- 感谢所有贡献者和用户

## 📧 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/yourusername/MindScienceAgent/issues)
- **功能请求**: [GitHub Discussions](https://github.com/yourusername/MindScienceAgent/discussions)

## 🗺️ 路线图

- [ ] 支持更多本地模型
- [ ] 增强可视化功能
- [ ] 添加更多数据集连接器
- [ ] 改进工具检索准确性
- [ ] 支持分布式计算
- [ ] Web界面

---

**注意**: 本项目仅用于研究目的。使用时请遵守相关数据集的使用许可和API服务条款。

Made with ❤️ for: scientific research community
