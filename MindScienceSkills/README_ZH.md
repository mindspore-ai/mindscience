# MindScienceSkills: 面向通用科学分析、仿真与实验的专业技能库

[![Skills](https://img.shields.io/badge/skills-300+-blue?style=flat-square)](skills/)
[![Hardware: GPU & NPU](https://img.shields.io/badge/Hardware-GPU%20%7C%20NPU-lightgrey)](#)
[![Agent Skills](https://img.shields.io/badge/Standard-Agent_Skills-blueviolet.svg)](https://agentskills.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#参与贡献)
[![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](LICENSE)

MindScienceSkills 是一个专为通用科学研究打造的开源智能体技能库，全面覆盖生物医药、化学材料、流体、地球科学、电磁等核心科学领域。

本仓库目前已内置了 300+ 专业科研技能（Skills），涵盖了参数配置繁琐的领域专用软件、各领域前沿 AI4S 模型、各学科专用的 Python 数据分析与处理工具库等，覆盖绝大部分日常科学分析、仿真与实验场景中的基础工具需求，降低 AI 跨学科调用的技术门槛。

MindScienceSkills 可以与 Hermes Agent、OpenClaw、Claude Code、JiuwenClaw等智能体无缝集成，也支持直接接入您专属的 AI 科研助手中。

## 📑 目录
  - [💡 主要特性](#-主要特性)
  - [👥 主要受众](#-主要受众)
  - [🗂️ 所有技能](#️-所有技能)
  - [⚙️ 快速开始与集成](#️-快速开始与集成)
    - [🔌 以 OpenClaw 为例](#-以-openclaw-为例)
      - [方式一：我是Agent](#方式一我是agent)
      - [方式二：我是Human](#方式二我是human)
  - [⚙️ 应用案例](#️-应用案例)
  - [⚙️ 致谢与开源参考](#️-致谢与开源参考)
  - [🤝 参与贡献](#-参与贡献)
---

## 💡 主要特性


### 1. 300+专业Skills：覆盖“分析-仿真-实验”科研全流程

覆盖100+生物、70+医学与药学、30+化学材料、10+地球科学、10+流体、5+电磁等领域Skills，其他领域（数学、天文、量子等）持续完善中：

| 主要领域 | Skills功能 | 应用场景 |
| ------ | ---------------------------------------------- | --------------------------------------------------------- |
| 🧬 **生物** | 多组学、基因组学与遗传学、蛋白质组学、转录组学、单细胞组学等 | 单细胞RNA测序分析、基因调控网络重建、蛋白质组学数据分析等 |
| 🎛️ **医学与药学** | 癌症基因组学、临床医学、药物发现、医学影像等 |靶向药物匹配、耐药机制分析、早期异常筛查等  |
| 🧪 **化学与材料** | 化学信息学、分子动力学、量子化学等 | 分子性质预测、分子动力学模拟等 |
| 🌦️ **地球科学** | 中期预报、降水预报、中尺度天气模拟等 | 短临、中期气象预报等 |
| 🌊 **流体力学** | 流体仿真 | 湍流分析、飞机翼型设计、风阻分析、海浪模拟等 |
| 📡 **电磁** | 电磁仿真 | 天线设计、电磁异常检测等 |
| ☢️ **能源** | 波形反演 | 地磁/地震成像等 |

每种学科基本包含6大类型Skills，覆盖绝大部分科学分析、仿真与实验场景中的基础工具需求，支持你的AI科研助手完成“分析-仿真-实验”的全栈工作流：

| Skill类型                | 代表性Skill           | 功能                                                         |
| ------------------------ | --------------------- | ------------------------------------------------------------ |
| **💻 领域专业软件类**         | hpc-vasp、hpc-gromacs                 | 深度集成 VASP、OpenFOAM等配置复杂的工业级HPC软件及FitDock等SOTA工具。内置输入构建、参数调整、计算结果分析等流程，构建20+人人可用的复杂领域软件skills，提升Agent软件调用成功率。 |
| **🐍 领域Python工具库** | deepchem、rdkit      | 覆盖各学科的专业数据处理、特征计算等工具库。             |
| **🧠 AI4S模型**           | alphafold3、proteinix | 原生适配Protenix、RFdiffusion等40+昇思、昇腾AI4S顶尖模型，极大扩展Agent知识边界；并支持用户分钟级将自有模型封装为智能体skill，将AI4S模型快速转换为即插即用的生产力工具。 |
| **💡 Know-How**        |    doped-perovskite-structure-analysis   | 沉淀了顶尖实验室“隐性知识”（e.g.,掺杂材料第一性原理计算skill），将复杂的长链路科研任务固化为Know-How类skill，提升Agent解决复杂科研任务效率。 |
| **🎛️ 实验室工作站**     | dual-station-electrochemical-workstation、centrifuge-workstation                      |  连打通干湿实验界限，提供湿实验设备操作skill（e.g.,电催化场景湿实验设备skill），为湿实验的设计和执行提供参考实现，缩短实验迭代周期，实现真正的闭环科研。                                                            |
| **📚 API**              | openalex-database、pubchem-database          | 集成业界常用文献、数据搜索API，涵盖文献搜索、数据获取等通用科研能力。         |


### 2. 高可靠调用：提升科学工具执行成功率
在真实科研应用场景中，Agent调用技能时常遇报错，导致执行受阻。原因主要有两点：一是技能本身说明模糊（如环境依赖、版本信息等），导致Agent在执行工具时容易报错；二是像 VASP 这类计算软件过于复杂，参数空间极大，Agent仅凭简单的模板，很难针对实际场景正确调参。

为提升技能的调用成功率，我们对模型类和软件类技能进行了优化：

* 🧠 **模型类技能的自动试错：** 我们开发了 [model-skill-creator](research_tools/general_tools/model-skill-creator/) 来规范技能的构建流程。在将各类模型转换为统一技能后，我们会先让 Agent 进行实际运行测试。如果遇到报错，Agent 会根据错误信息尝试修正代码。我们将最终测试通过的可用代码保留下来，不断迭代从而提升技能可靠性。

  

* 💻 **软件类技能的场景化指导：** 对于典型的计算软件（以 VASP 为例），我们不再只提供简单的输入模板。针对不同的使用场景和物质性质，我们在技能中内置了对应的查询方式与详细的参数调整指导。同时，我们还补充了运行结果的分析方法与后续的优化建议。这不仅能帮助大模型把软件成功运行起来，还能指导它根据结果做出合理的科学调整。


### 3. 多硬件生态兼容
我们在技能库中分别集成了适用于 GPU 生态和 NPU 生态的AI4S模型，您的科研助手可以在GPU/NPU上无缝调用相应的模型技能，实现科研计算任务的灵活、高效部署。

---

## 👥 主要受众

* 🤖 **AI4S 开发者：** 可直接将本库接入智能体框架，也可基于skill-creator类skill构建自研工具，大幅节省专业学科工具的封装成本。

* 🔬 **跨学科科研人员与领域专家：** 无需深究代码细节，即可通过智能体稳定调用各类复杂的计算仿真软件与 AI 预测模型。

* ⚙️ **实验室自动化工程师：** 可参考利构建实验室自动化设备skill来接入实验室设备。


---

## 🗂️ 所有技能



<details>
<summary><strong>生物  (110 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [abalign](biology/bioinformatics/abalign) | bioinformatics | know-how | CPU |
| [abcvista](biology/bioinformatics/abcvista) | bioinformatics | know-how | CPU |
| [abrsa](biology/bioinformatics/abrsa) | bioinformatics | know-how | CPU |
| [adaptyv](biology/protein_and_structure/adaptyv) | protein_and_structure | python package | CPU |
| [alphafold-database](biology/database_and_knowledge/alphafold-database) | database_and_knowledge | api | CPU |
| [anndata](biology/general_tools/anndata) | general_tools | python package | CPU |
| [arboreto](biology/multi_omics_integration/arboreto) | multi_omics_integration | python package | CPU |
| [biorxiv-database](biology/database_and_knowledge/biorxiv-database) | database_and_knowledge | api | CPU |
| [bioservices](biology/database_and_knowledge/bioservices) | database_and_knowledge | python package | CPU |
| [boltz-1](biology/protein_and_structure/boltz-1) | protein_and_structure | AI model | NPU |
| [cellxgene-census](biology/database_and_knowledge/cellxgene-census) | database_and_knowledge | api | CPU |
| [cobrapy](biology/protein_and_structure/cobrapy) | protein_and_structure | python package | CPU |
| [deepchem](biology/protein_and_structure/deepchem) | protein_and_structure | python package | GPU/CPU |
| [deeptools](biology/general_tools/deeptools) | general_tools | know-how | CPU |
| [diffdock](biology/protein_and_structure/diffdock) | protein_and_structure | python package | CPU/GPU |
| [diffdock-model](biology/protein_and_structure/diffdock-model) | protein_and_structure | AI model | NPU |
| [ena-database](biology/database_and_knowledge/ena-database) | database_and_knowledge | api | CPU |
| [ensembl-database](biology/database_and_knowledge/ensembl-database) | database_and_knowledge | api | CPU |
| [esm](biology/protein_and_structure/esm) | protein_and_structure | python package | CPU/GPU |
| [esm2](biology/protein_and_structure/esm2) | protein_and_structure | AI model | NPU |
| [esm3](biology/protein_and_structure/esm3) | protein_and_structure | AI model | NPU |
| [esmfold](biology/protein_and_structure/esmfold) | protein_and_structure | AI model | NPU |
| [etetoolkit](biology/genomics_and_genetics/etetoolkit) | genomics_and_genetics | python package | CPU |
| [fitdock](biology/bioinformatics/fitdock) | bioinformatics | know-how | CPU |
| [flowio](biology/general_tools/flowio) | general_tools | python package | CPU |
| [gene-database](biology/database_and_knowledge/gene-database) | database_and_knowledge | api | CPU |
| [geneformer](biology/transcriptomics_and_sc_omics/geneformer) | transcriptomics_and_sc_omics | AI model | NPU |
| [geniml](biology/general_tools/geniml) | general_tools | python package | GPU/CPU |
| [geo-database](biology/database_and_knowledge/geo-database) | database_and_knowledge | api | CPU |
| [gnomad-database](biology/database_and_knowledge/gnomad-database) | database_and_knowledge | api | CPU |
| [gtex-database](biology/database_and_knowledge/gtex-database) | database_and_knowledge | api | CPU |
| [gget](biology/database_and_knowledge/gget) | database_and_knowledge | python package | CPU |
| [gtars](biology/bioinformatics/gtars) | bioinformatics | python package | CPU |
| [hmdb-database](biology/database_and_knowledge/hmdb-database) | database_and_knowledge | api | CPU |
| [hpc-gromacs](biology/protein_and_structure/hpc-gromacs) | protein_and_structure | HPC software | CPU/GPU |
| [interpro-database](biology/database_and_knowledge/interpro-database) | database_and_knowledge | api | CPU |
| [jaspar-database](biology/database_and_knowledge/jaspar-database) | database_and_knowledge | api | CPU |
| [kegg-database](biology/database_and_knowledge/kegg-database) | database_and_knowledge | api | CPU |
| [lamindb](biology/general_tools/lamindb) | general_tools | python package | CPU |
| [ligandmpnn](biology/protein_and_structure/ligandmpnn) | protein_and_structure | AI model | NPU |
| [matchms](biology/protein_and_structure/matchms) | protein_and_structure | python package | CPU |
| [megaprotein](biology/protein_and_structure/megaprotein) | protein_and_structure | AI model | NPU |
| [metabolomics-workbench-database](biology/database_and_knowledge/metabolomics-workbench-database) | database_and_knowledge | api | CPU |
| [microbiome-research](biology/genomics_and_genetics/microbiome-research) | genomics_and_genetics | know-how | CPU |
| [monarch-database](biology/database_and_knowledge/monarch-database) | database_and_knowledge | api | CPU |
| [oligoformer](biology/transcriptomics_and_sc_omics/oligoformer) | transcriptomics_and_sc_omics | AI model | NPU |
| [openfold2](biology/protein_and_structure/openfold2) | protein_and_structure | AI model | NPU |
| [pdb-database](biology/database_and_knowledge/pdb-database) | database_and_knowledge | api | CPU |
| [peptidebert](biology/protein_and_structure/peptidebert) | protein_and_structure | AI model | NPU |
| [phylogenetics](biology/genomics_and_genetics/phylogenetics) | genomics_and_genetics | python package | CPU |
| [prott5](biology/protein_and_structure/prott5) | protein_and_structure | AI model | NPU |
| [pyehr](biology/database_and_knowledge/pyehr) | database_and_knowledge | AI model | NPU |
| [pydeseq2](biology/general_tools/pydeseq2) | general_tools | python package | CPU |
| [pyopenms](biology/protein_and_structure/pyopenms) | protein_and_structure | python package | CPU |
| [pysam](biology/general_tools/pysam) | general_tools | python package | CPU |
| [reactome-database](biology/database_and_knowledge/reactome-database) | database_and_knowledge | api | CPU |
| [rfantibody](biology/transcriptomics_and_sc_omics/rfantibody) | transcriptomics_and_sc_omics | AI model | NPU |
| [scanpy](biology/transcriptomics_and_sc_omics/scanpy) | transcriptomics_and_sc_omics | python package | CPU |
| [scikit-bio](biology/general_tools/scikit-bio) | general_tools | python package | CPU |
| [scvelo](biology/transcriptomics_and_sc_omics/scvelo) | transcriptomics_and_sc_omics | python package | CPU |
| [scvi-tools](biology/transcriptomics_and_sc_omics/scvi-tools) | transcriptomics_and_sc_omics | python package | GPU/CPU |
| [string-database](biology/database_and_knowledge/string-database) | database_and_knowledge | api | CPU |
| [tiledbvcf](biology/general_tools/tiledbvcf) | general_tools | python package | CPU |
| [tooluniverse-aging-senescence](biology/multi_omics_integration/tooluniverse-aging-senescence) | multi_omics_integration | know-how | CPU |
| [tooluniverse-antibody-engineering](biology/protein_and_structure/tooluniverse-antibody-engineering) | protein_and_structure | know-how | CPU |
| [tooluniverse-comparative-genomics](biology/genomics_and_genetics/tooluniverse-comparative-genomics) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-crispr-screen-analysis](biology/genomics_and_genetics/tooluniverse-crispr-screen-analysis) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-epigenomics](biology/genomics_and_genetics/tooluniverse-epigenomics) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-epigenomics-chromatin](biology/genomics_and_genetics/tooluniverse-epigenomics-chromatin) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-expression-data-retrieval](biology/database_and_knowledge/tooluniverse-expression-data-retrieval) | database_and_knowledge | know-how | CPU |
| [tooluniverse-functional-genomics-screens](biology/genomics_and_genetics/tooluniverse-functional-genomics-screens) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-gene-enrichment](biology/transcriptomics_and_sc_omics/tooluniverse-gene-enrichment) | transcriptomics_and_sc_omics | know-how | CPU |
| [tooluniverse-gene-regulatory-networks](biology/multi_omics_integration/tooluniverse-gene-regulatory-networks) | multi_omics_integration | know-how | CPU |
| [tooluniverse-gpcr-structural-pharmacology](biology/protein_and_structure/tooluniverse-gpcr-structural-pharmacology) | protein_and_structure | know-how | CPU |
| [tooluniverse-hla-immunogenomics](biology/multi_omics_integration/tooluniverse-hla-immunogenomics) | multi_omics_integration | know-how | CPU |
| [tooluniverse-lipidomics](biology/protein_and_structure/tooluniverse-lipidomics) | protein_and_structure | know-how | CPU |
| [tooluniverse-metabolomics](biology/protein_and_structure/tooluniverse-metabolomics) | protein_and_structure | know-how | CPU |
| [tooluniverse-metabolomics-analysis](biology/protein_and_structure/tooluniverse-metabolomics-analysis) | protein_and_structure | know-how | CPU |
| [tooluniverse-metabolomics-pathway](biology/protein_and_structure/tooluniverse-metabolomics-pathway) | protein_and_structure | know-how | CPU |
| [tooluniverse-metagenomics-analysis](biology/genomics_and_genetics/tooluniverse-metagenomics-analysis) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-model-organism-genetics](biology/genomics_and_genetics/tooluniverse-model-organism-genetics) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-multi-omics-integration](biology/multi_omics_integration/tooluniverse-multi-omics-integration) | multi_omics_integration | know-how | CPU |
| [tooluniverse-noncoding-rna](biology/transcriptomics_and_sc_omics/tooluniverse-noncoding-rna) | transcriptomics_and_sc_omics | know-how | CPU |
| [tooluniverse-phylogenetics](biology/genomics_and_genetics/tooluniverse-phylogenetics) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-plant-genomics](biology/genomics_and_genetics/tooluniverse-plant-genomics) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-population-genetics](biology/genomics_and_genetics/tooluniverse-population-genetics) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-population-genetics-1000genomes](biology/genomics_and_genetics/tooluniverse-population-genetics-1000genomes) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-protein-interactions](biology/multi_omics_integration/tooluniverse-protein-interactions) | multi_omics_integration | know-how | CPU |
| [tooluniverse-protein-modification-analysis](biology/protein_and_structure/tooluniverse-protein-modification-analysis) | protein_and_structure | know-how | CPU |
| [tooluniverse-protein-structure-prediction](biology/protein_and_structure/tooluniverse-protein-structure-prediction) | protein_and_structure | know-how | CPU |
| [tooluniverse-protein-structure-retrieval](biology/protein_and_structure/tooluniverse-protein-structure-retrieval) | protein_and_structure | know-how | CPU |
| [tooluniverse-protein-therapeutic-design](biology/protein_and_structure/tooluniverse-protein-therapeutic-design) | protein_and_structure | know-how | CPU |
| [tooluniverse-proteomics-analysis](biology/multi_omics_integration/tooluniverse-proteomics-analysis) | multi_omics_integration | know-how | CPU |
| [tooluniverse-proteomics-data-retrieval](biology/database_and_knowledge/tooluniverse-proteomics-data-retrieval) | database_and_knowledge | know-how | CPU |
| [tooluniverse-regulatory-genomics](biology/transcriptomics_and_sc_omics/tooluniverse-regulatory-genomics) | transcriptomics_and_sc_omics | know-how | CPU |
| [tooluniverse-regulatory-variant-analysis](biology/genomics_and_genetics/tooluniverse-regulatory-variant-analysis) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-rnaseq-deseq2](biology/transcriptomics_and_sc_omics/tooluniverse-rnaseq-deseq2) | transcriptomics_and_sc_omics | know-how | CPU |
| [tooluniverse-sequence-analysis](biology/genomics_and_genetics/tooluniverse-sequence-analysis) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-sequence-retrieval](biology/database_and_knowledge/tooluniverse-sequence-retrieval) | database_and_knowledge | know-how | CPU |
| [tooluniverse-single-cell](biology/transcriptomics_and_sc_omics/tooluniverse-single-cell) | transcriptomics_and_sc_omics | know-how | CPU |
| [tooluniverse-spatial-omics-analysis](biology/transcriptomics_and_sc_omics/tooluniverse-spatial-omics-analysis) | transcriptomics_and_sc_omics | know-how | CPU |
| [tooluniverse-spatial-transcriptomics](biology/transcriptomics_and_sc_omics/tooluniverse-spatial-transcriptomics) | transcriptomics_and_sc_omics | know-how | CPU |
| [tooluniverse-statistical-modeling](biology/general_tools/tooluniverse-statistical-modeling) | general_tools | know-how | CPU |
| [tooluniverse-stem-cell-organoid](biology/multi_omics_integration/tooluniverse-stem-cell-organoid) | multi_omics_integration | know-how | CPU |
| [tooluniverse-structural-proteomics](biology/protein_and_structure/tooluniverse-structural-proteomics) | protein_and_structure | know-how | CPU |
| [tooluniverse-structural-variant-analysis](biology/genomics_and_genetics/tooluniverse-structural-variant-analysis) | genomics_and_genetics | know-how | CPU |
| [tooluniverse-systems-biology](biology/multi_omics_integration/tooluniverse-systems-biology) | multi_omics_integration | know-how | CPU |
| [tooluniverse-variant-analysis](biology/genomics_and_genetics/tooluniverse-variant-analysis) | genomics_and_genetics | know-how | CPU |
| [torchfold](biology/protein_and_structure/torchfold) | protein_and_structure | AI model | NPU |
| [uniprot-database](biology/database_and_knowledge/uniprot-database) | database_and_knowledge | api | CPU |

</details>

<details>
<summary><strong>医学与药学  (77 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [bindingdb-database](medical_and_pharmacy/database_and_knowledge/bindingdb-database) | database_and_knowledge | api | CPU |
| [brenda-database](medical_and_pharmacy/database_and_knowledge/brenda-database) | database_and_knowledge | api | CPU |
| [cbioportal-database](medical_and_pharmacy/database_and_knowledge/cbioportal-database) | database_and_knowledge | api | CPU |
| [chembl-database](medical_and_pharmacy/database_and_knowledge/chembl-database) | database_and_knowledge | api | CPU |
| [clinical-decision-support](medical_and_pharmacy/clinical/clinical-decision-support) | clinical | know-how | CPU |
| [clinical-reports](medical_and_pharmacy/clinical/clinical-reports) | clinical | know-how | CPU |
| [clinicaltrials-database](medical_and_pharmacy/database_and_knowledge/clinicaltrials-database) | database_and_knowledge | api | CPU |
| [clinpgx-database](medical_and_pharmacy/database_and_knowledge/clinpgx-database) | database_and_knowledge | api | CPU |
| [clinvar-database](medical_and_pharmacy/database_and_knowledge/clinvar-database) | database_and_knowledge | api | CPU |
| [cosmic-database](medical_and_pharmacy/database_and_knowledge/cosmic-database) | database_and_knowledge | api | CPU |
| [depmap](medical_and_pharmacy/cancer_genomics/depmap) | cancer_genomics | python package | CPU |
| [diffsbdd](medical_and_pharmacy/drug_development/diffsbdd) | drug_development | AI model | NPU |
| [drugbank-database](medical_and_pharmacy/database_and_knowledge/drugbank-database) | database_and_knowledge | api | CPU |
| [fda-database](medical_and_pharmacy/database_and_knowledge/fda-database) | database_and_knowledge | api | CPU |
| [genmol](medical_and_pharmacy/drug_development/genmol) | drug_development | AI model | NPU |
| [gwas-database](medical_and_pharmacy/database_and_knowledge/gwas-database) | database_and_knowledge | api | CPU |
| [histolab](medical_and_pharmacy/medical_imaging_and__signal_processing/histolab) | medical_imaging_and__signal_processing | python package | CPU |
| [imaging-data-commons](medical_and_pharmacy/medical_imaging_and__signal_processing/imaging-data-commons) | medical_imaging_and__signal_processing | know-how | CPU |
| [medchem](medical_and_pharmacy/drug_development/medchem) | drug_development | python package | CPU |
| [neurokit2](medical_and_pharmacy/medical_imaging_and__signal_processing/neurokit2) | medical_imaging_and__signal_processing | python package | CPU |
| [neuropixels-analysis](medical_and_pharmacy/medical_imaging_and__signal_processing/neuropixels-analysis) | medical_imaging_and__signal_processing | know-how | CPU |
| [opentargets-database](medical_and_pharmacy/database_and_knowledge/opentargets-database) | database_and_knowledge | api | CPU |
| [pathml](medical_and_pharmacy/medical_imaging_and__signal_processing/pathml) | medical_imaging_and__signal_processing | python package | CPU/GPU |
| [pubmed-database](medical_and_pharmacy/database_and_knowledge/pubmed-database) | database_and_knowledge | api | CPU |
| [pydicom](medical_and_pharmacy/medical_imaging_and__signal_processing/pydicom) | medical_imaging_and__signal_processing | python package | CPU |
| [pyhealth](medical_and_pharmacy/clinical/pyhealth) | clinical | python package | GPU/CPU |
| [pytdc](medical_and_pharmacy/drug_development/pytdc) | drug_development | python package | CPU |
| [scikit-survival](medical_and_pharmacy/cancer_genomics/scikit-survival) | cancer_genomics | python package | CPU |
| [tooluniverse-acmg-variant-classification](medical_and_pharmacy/genomic_medicine/tooluniverse-acmg-variant-classification) | genomic_medicine | know-how | CPU |
| [tooluniverse-admet-prediction](medical_and_pharmacy/drug_development/tooluniverse-admet-prediction) | drug_development | know-how | CPU |
| [tooluniverse-adverse-event-detection](medical_and_pharmacy/drug_development/tooluniverse-adverse-event-detection) | drug_development | know-how | CPU |
| [tooluniverse-adverse-outcome-pathway](medical_and_pharmacy/drug_development/tooluniverse-adverse-outcome-pathway) | drug_development | know-how | CPU |
| [tooluniverse-binder-discovery](medical_and_pharmacy/drug_development/tooluniverse-binder-discovery) | drug_development | know-how | CPU |
| [tooluniverse-cancer-classification](medical_and_pharmacy/cancer_genomics/tooluniverse-cancer-classification) | cancer_genomics | know-how | CPU |
| [tooluniverse-cancer-genomics-tcga](medical_and_pharmacy/cancer_genomics/tooluniverse-cancer-genomics-tcga) | cancer_genomics | know-how | CPU |
| [tooluniverse-cancer-variant-interpretation](medical_and_pharmacy/cancer_genomics/tooluniverse-cancer-variant-interpretation) | cancer_genomics | know-how | CPU |
| [tooluniverse-cell-line-profiling](medical_and_pharmacy/cancer_genomics/tooluniverse-cell-line-profiling) | cancer_genomics | know-how | CPU |
| [tooluniverse-clinical-data-integration](medical_and_pharmacy/clinical/tooluniverse-clinical-data-integration) | clinical | know-how | CPU |
| [tooluniverse-clinical-guidelines](medical_and_pharmacy/clinical/tooluniverse-clinical-guidelines) | clinical | know-how | CPU |
| [tooluniverse-clinical-trial-design](medical_and_pharmacy/clinical/tooluniverse-clinical-trial-design) | clinical | know-how | CPU |
| [tooluniverse-clinical-trial-matching](medical_and_pharmacy/clinical/tooluniverse-clinical-trial-matching) | clinical | know-how | CPU |
| [tooluniverse-disease-research](medical_and_pharmacy/clinical/tooluniverse-disease-research) | clinical | know-how | CPU |
| [tooluniverse-drug-drug-interaction](medical_and_pharmacy/drug_development/tooluniverse-drug-drug-interaction) | drug_development | know-how | CPU |
| [tooluniverse-drug-mechanism-research](medical_and_pharmacy/drug_development/tooluniverse-drug-mechanism-research) | drug_development | know-how | CPU |
| [tooluniverse-drug-regulatory](medical_and_pharmacy/drug_development/tooluniverse-drug-regulatory) | drug_development | know-how | CPU |
| [tooluniverse-drug-repurposing](medical_and_pharmacy/drug_development/tooluniverse-drug-repurposing) | drug_development | know-how | CPU |
| [tooluniverse-drug-research](medical_and_pharmacy/drug_development/tooluniverse-drug-research) | drug_development | know-how | CPU |
| [tooluniverse-drug-target-validation](medical_and_pharmacy/drug_development/tooluniverse-drug-target-validation) | drug_development | know-how | CPU |
| [tooluniverse-gene-disease-association](medical_and_pharmacy/genomic_medicine/tooluniverse-gene-disease-association) | genomic_medicine | know-how | CPU |
| [tooluniverse-gwas-drug-discovery](medical_and_pharmacy/genomic_medicine/tooluniverse-gwas-drug-discovery) | genomic_medicine | know-how | CPU |
| [tooluniverse-gwas-finemapping](medical_and_pharmacy/genomic_medicine/tooluniverse-gwas-finemapping) | genomic_medicine | know-how | CPU |
| [tooluniverse-gwas-snp-interpretation](medical_and_pharmacy/genomic_medicine/tooluniverse-gwas-snp-interpretation) | genomic_medicine | know-how | CPU |
| [tooluniverse-gwas-study-explorer](medical_and_pharmacy/genomic_medicine/tooluniverse-gwas-study-explorer) | genomic_medicine | know-how | CPU |
| [tooluniverse-gwas-trait-to-gene](medical_and_pharmacy/genomic_medicine/tooluniverse-gwas-trait-to-gene) | genomic_medicine | know-how | CPU |
| [tooluniverse-image-analysis](medical_and_pharmacy/medical_imaging_and__signal_processing/tooluniverse-image-analysis) | medical_imaging_and__signal_processing | know-how | CPU |
| [tooluniverse-immune-repertoire-analysis](medical_and_pharmacy/drug_development/tooluniverse-immune-repertoire-analysis) | drug_development | know-how | CPU |
| [tooluniverse-immunology](medical_and_pharmacy/drug_development/tooluniverse-immunology) | drug_development | know-how | CPU |
| [tooluniverse-immunotherapy-response-prediction](medical_and_pharmacy/cancer_genomics/tooluniverse-immunotherapy-response-prediction) | cancer_genomics | know-how | CPU |
| [tooluniverse-infectious-disease](medical_and_pharmacy/clinical/tooluniverse-infectious-disease) | clinical | know-how | CPU |
| [tooluniverse-kegg-disease-drug](medical_and_pharmacy/drug_development/tooluniverse-kegg-disease-drug) | drug_development | know-how | CPU |
| [tooluniverse-multiomic-disease-characterization](medical_and_pharmacy/genomic_medicine/tooluniverse-multiomic-disease-characterization) | genomic_medicine | know-how | CPU |
| [tooluniverse-network-pharmacology](medical_and_pharmacy/drug_development/tooluniverse-network-pharmacology) | drug_development | know-how | CPU |
| [tooluniverse-pathway-disease-genetics](medical_and_pharmacy/genomic_medicine/tooluniverse-pathway-disease-genetics) | genomic_medicine | know-how | CPU |
| [tooluniverse-pharmacogenomics](medical_and_pharmacy/drug_development/tooluniverse-pharmacogenomics) | drug_development | know-how | CPU |
| [tooluniverse-pharmacovigilance](medical_and_pharmacy/drug_development/tooluniverse-pharmacovigilance) | drug_development | know-how | CPU |
| [tooluniverse-polygenic-risk-score](medical_and_pharmacy/genomic_medicine/tooluniverse-polygenic-risk-score) | genomic_medicine | know-how | CPU |
| [tooluniverse-precision-medicine-stratification](medical_and_pharmacy/clinical/tooluniverse-precision-medicine-stratification) | clinical | know-how | CPU |
| [tooluniverse-precision-oncology](medical_and_pharmacy/cancer_genomics/tooluniverse-precision-oncology) | cancer_genomics | know-how | CPU |
| [tooluniverse-rare-disease-diagnosis](medical_and_pharmacy/clinical/tooluniverse-rare-disease-diagnosis) | clinical | know-how | CPU |
| [tooluniverse-rare-disease-genomics](medical_and_pharmacy/genomic_medicine/tooluniverse-rare-disease-genomics) | genomic_medicine | know-how | CPU |
| [tooluniverse-target-research](medical_and_pharmacy/drug_development/tooluniverse-target-research) | drug_development | know-how | CPU |
| [tooluniverse-toxicology](medical_and_pharmacy/drug_development/tooluniverse-toxicology) | drug_development | know-how | CPU |
| [tooluniverse-vaccine-design](medical_and_pharmacy/drug_development/tooluniverse-vaccine-design) | drug_development | know-how | CPU |
| [tooluniverse-variant-functional-annotation](medical_and_pharmacy/genomic_medicine/tooluniverse-variant-functional-annotation) | genomic_medicine | know-how | CPU |
| [tooluniverse-variant-interpretation](medical_and_pharmacy/genomic_medicine/tooluniverse-variant-interpretation) | genomic_medicine | know-how | CPU |
| [tooluniverse-variant-to-mechanism](medical_and_pharmacy/genomic_medicine/tooluniverse-variant-to-mechanism) | genomic_medicine | know-how | CPU |
| [torchdrug](medical_and_pharmacy/drug_development/torchdrug) | drug_development | python package | GPU/CPU |

</details>

<details>
<summary><strong>化学与材料  (49 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [centrifuge-workstation](chemistry_and_materials/real_world_workstations/centrifuge-workstation) | real_world_workstations | laboratory automation | / |
| [chgnet](chemistry_and_materials/molecular_dynamics/chgnet) | molecular_dynamics | AI model | NPU |
| [chemistry-query](chemistry_and_materials/general_tools/chemistry-query/) | general_tools | know-how | CPU |
| [chemprop](chemistry_and_materials/cheminformatics/chemprop) | cheminformatics | AI model | NPU |
| [crystalflow](chemistry_and_materials/molecular_dynamics/crystalflow) | molecular_dynamics | AI model | NPU |
| [datamol](chemistry_and_materials/cheminformatics/datamol) | cheminformatics | python package | CPU |
| [deephe3nn](chemistry_and_materials/molecular_dynamics/deephe3nn) | molecular_dynamics | AI model | NPU |
| [diffcsp](chemistry_and_materials/molecular_dynamics/diffcsp) | molecular_dynamics | AI model | NPU |
| [doped-perovskite-structure-analysis](chemistry_and_materials/quantum_chemistry/doped-perovskite-structure-analysis) | quantum_chemistry | know-how | CPU |
| [dryer-workstation](chemistry_and_materials/real_world_workstations/dryer-workstation) | real_world_workstations |  laboratory automation | / |
| [dual-station-electrochemical-workstation](chemistry_and_materials/real_world_workstations/dual-station-electrochemical-workstation) | real_world_workstations | laboratory automation | / |
| [elec-chem-storage-workstation](chemistry_and_materials/real_world_workstations/elec-chem-storage-workstation) | real_world_workstations |  laboratory automation | / |
| [gptff](chemistry_and_materials/molecular_dynamics/gptff) | molecular_dynamics | AI model | NPU |
| [hpc-cp2k](chemistry_and_materials/quantum_chemistry/hpc-cp2k) | quantum_chemistry | HPC software | CPU/GPU |
| [hpc-feff](chemistry_and_materials/quantum_chemistry/hpc-feff) | quantum_chemistry | HPC software | CPU |
| [hpc-gaussian](chemistry_and_materials/quantum_chemistry/hpc-gaussian) | quantum_chemistry | HPC software | CPU |
| [hpc-lammps](chemistry_and_materials/molecular_dynamics/hpc-lammps) | molecular_dynamics | HPC software | CPU/GPU |
| [hpc-nwchem](chemistry_and_materials/quantum_chemistry/hpc-nwchem) | quantum_chemistry | HPC software | CPU |
| [hpc-orca](chemistry_and_materials/quantum_chemistry/hpc-orca) | quantum_chemistry | HPC software | CPU/GPU |
| [hpc-psi4](chemistry_and_materials/quantum_chemistry/hpc-psi4) | quantum_chemistry | HPC software | CPU |
| [hpc-pyscf](chemistry_and_materials/quantum_chemistry/hpc-pyscf) | quantum_chemistry | HPC software | CPU |
| [hpc-quantum-espresso](chemistry_and_materials/quantum_chemistry/hpc-quantum-espresso) | quantum_chemistry | HPC software | CPU/GPU |
| [hpc-vasp](chemistry_and_materials/quantum_chemistry/hpc-vasp) | quantum_chemistry | HPC software | CPU/GPU |
| [hpc-xtb](chemistry_and_materials/quantum_chemistry/hpc-xtb) | quantum_chemistry | HPC software | CPU |
| [liquid-dispensing-workstation](chemistry_and_materials/real_world_workstations/liquid-dispensing-workstation) | real_world_workstations |  laboratory automation | / |
| [magnetic-stirring-workstation](chemistry_and_materials/real_world_workstations/magnetic-stirring-workstation) | real_world_workstations |  laboratory automation | / |
| [material-workstation](chemistry_and_materials/real_world_workstations/pure-workstation)| real_world_workstations |  laboratory automation | / |
| [matformer](chemistry_and_materials/molecular_dynamics/matformer) | molecular_dynamics | AI model | NPU |
| [mattergen](chemistry_and_materials/molecular_dynamics/mattergen) | molecular_dynamics | AI model | NPU |
| [mattersim](chemistry_and_materials/molecular_dynamics/mattersim) | molecular_dynamics | AI model | NPU |
| [molfeat](chemistry_and_materials/cheminformatics/molfeat) | cheminformatics | python package | CPU/GPU |
| [molecular-dynamics](chemistry_and_materials/molecular_dynamics/molecular-dynamics) | molecular_dynamics | know-how | CPU |
| [nequip](chemistry_and_materials/molecular_dynamics/nequip) | molecular_dynamics | AI model | NPU |
| [orb](chemistry_and_materials/molecular_dynamics/orb) | molecular_dynamics | AI model | NPU |
| [pubchem-database](chemistry_and_materials/database_and_knowledge/pubchem-database) | database_and_knowledge | api | CPU |
| [pure-workstation](chemistry_and_materials/real_world_workstations/pure-workstation) | real_world_workstations  |   laboratory automation | / |
| [pymatgen](chemistry_and_materials/general_tools/pymatgen) | general_tools | python package | CPU |
| [quip](chemistry_and_materials/quantum_chemistry/quip) | quantum_chemistry | AI model | NPU |
| [rdkit](chemistry_and_materials/cheminformatics/rdkit) | cheminformatics | python package | CPU |
| [reann](chemistry_and_materials/molecular_dynamics/reann) | molecular_dynamics | AI model | NPU |
| [schnet](chemistry_and_materials/molecular_dynamics/schnet) | molecular_dynamics | AI model | NPU |
| [tora](chemistry_and_materials/general_tools/tora) | general_tools | AI model | NPU |
| [tooluniverse-chemical-compound-retrieval](chemistry_and_materials/cheminformatics/tooluniverse-chemical-compound-retrieval) | cheminformatics | know-how | CPU |
| [tooluniverse-chemical-safety](chemistry_and_materials/general_tools/tooluniverse-chemical-safety) | general_tools | know-how | CPU |
| [tooluniverse-chemical-sourcing](chemistry_and_materials/general_tools/tooluniverse-chemical-sourcing) | general_tools | know-how | CPU |
| [tooluniverse-electron-microscopy](chemistry_and_materials/general_tools/tooluniverse-electron-microscopy) | general_tools | know-how | CPU |
| [tooluniverse-small-molecule-discovery](chemistry_and_materials/cheminformatics/tooluniverse-small-molecule-discovery) | cheminformatics | know-how | CPU |
| [ultrasonic-cleaning-workstation](chemistry_and_materials/real_world_workstations/ultrasonic-cleaning-workstation) | real_world_workstations |  laboratory automation | / |
| [zinc-database](chemistry_and_materials/database_and_knowledge/zinc-database) | database_and_knowledge | api | CPU |

</details>

<details>
<summary><strong>地球科学  (14 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [cfgrib](earth_sciences/meteorology/cfgrib) | meteorology | python package | CPU |
| [corrdiff](earth_sciences/meteorology/corrdiff) | meteorology | AI model | GPU |
| [eccodes](earth_sciences/meteorology/eccodes) | meteorology | python package | CPU |
| [geomaster](earth_sciences/geospatial/geomaster) | geospatial | python package | CPU |
| [geopandas](earth_sciences/geospatial/geopandas) | geospatial | python package | CPU |
| [hpc-openfwi](earth_sciences/geophysics/hpc-openfwi) | geophysics | HPC software | CPU |
| [hpc-wrf](earth_sciences/meteorology/hpc-wrf) | meteorology | HPC software | CPU |
| [leadformer](earth_sciences/meteorology/leadformer) | meteorology | AI model | NPU |
| [metpy](earth_sciences/meteorology/metpy) | meteorology | python package | CPU |
| [py-art](earth_sciences/meteorology/py-art) | meteorology | python package | CPU |
| [satpy](earth_sciences/meteorology/satpy) | meteorology | python package | CPU |
| [siphon](earth_sciences/meteorology/siphon) | meteorology | python package | CPU |
| [wrf-python](earth_sciences/meteorology/wrf-python) | meteorology | python package | CPU |
| [xesmf](earth_sciences/meteorology/xesmf) | meteorology | python package | CPU |

</details>



<details>
<summary><strong>电磁学  (7 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [gprmax](electromagnetics/electromagnetics/gprmax) | electromagnetics | python package | CPU |
| [hpc-cst](electromagnetics/electromagnetics/hpc-cst) | electromagnetics | HPC software | CPU/GPU |
| [meep](electromagnetics/electromagnetics/meep) | electromagnetics | python package | CPU |
| [ngsolve](electromagnetics/electromagnetics/ngsolve) | electromagnetics | python package | CPU |
| [pyaedt](electromagnetics/electromagnetics/pyaedt) | electromagnetics | python package | CPU |
| [pyfemm](electromagnetics/electromagnetics/pyfemm) | electromagnetics | python package | CPU |
| [scikit-rf](electromagnetics/electromagnetics/scikit-rf) | electromagnetics | python package | CPU |

</details>

<details>
<summary><strong>流体力学  (18 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [cantera](fluid_dynamics/fluid_dynamics/cantera) | fluid_dynamics | python package | CPU |
| [coolprop](fluid_dynamics/fluid_dynamics/coolprop) | fluid_dynamics | python package | CPU |
| [dpot](fluid_dynamics/fluid_dynamics/dpot) | fluid_dynamics | AI model | GPU |
| [fenics](fluid_dynamics/fluid_dynamics/fenics) | fluid_dynamics | python package | CPU |
| [fipy](fluid_dynamics/fluid_dynamics/fipy) | fluid_dynamics | python package | CPU |
| [fno1d](fluid_dynamics/fluid_dynamics/fno1d) | fluid_dynamics | AI model | NPU |
| [fno2d](fluid_dynamics/fluid_dynamics/fno2d) | fluid_dynamics | AI model | NPU |
| [fno3d](fluid_dynamics/fluid_dynamics/fno3d) | fluid_dynamics | AI model | NPU |
| [fluidsim](fluid_dynamics/fluid_dynamics/fluidsim) | fluid_dynamics | python package | CPU |
| [hpc-fenics](fluid_dynamics/fluid_dynamics/hpc-fenics) | fluid_dynamics | HPC software | CPU |
| [hpc-openfoam](fluid_dynamics/fluid_dynamics/hpc-openfoam) | fluid_dynamics | HPC software | CPU |
| [hpc-su2](fluid_dynamics/fluid_dynamics/hpc-su2) | fluid_dynamics | HPC software | CPU/GPU |
| [p2c2net](fluid_dynamics/fluid_dynamics/p2c2net) | fluid_dynamics | AI model | NPU |
| [phiflow](fluid_dynamics/fluid_dynamics/phiflow) | fluid_dynamics | python package | CPU |
| [pyfoam](fluid_dynamics/fluid_dynamics/pyfoam) | fluid_dynamics | python package | CPU |
| [pysph](fluid_dynamics/fluid_dynamics/pysph) | fluid_dynamics | python package | CPU |
| [pyvista](fluid_dynamics/fluid_dynamics/pyvista) | fluid_dynamics | python package | CPU |
| [transolver](fluid_dynamics/fluid_dynamics/transolver) | fluid_dynamics | AI model | NPU |

</details>

<details>
<summary><strong>数学  (4 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [hpc-fftw](mathematics/numerical_computing/hpc-fftw) | numerical_computing | HPC software | CPU |
| [hpc-openblas](mathematics/numerical_computing/hpc-openblas) | numerical_computing | HPC software | CPU |
| [matlab](mathematics/numerical_computing/matlab) | numerical_computing | python package | CPU |
| [sympy](mathematics/numerical_computing/sympy) | numerical_computing | python package | CPU |

</details>



<details>
<summary><strong>经济学  (4 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [edgartools](economics/economic_analysis/edgartools) | economic_analysis | know-how | CPU |
| [fred-economic-data](economics/economic_analysis/fred-economic-data) | economic_analysis | api | CPU |
| [market-research-reports](economics/economic_analysis/market-research-reports) | economic_analysis | know-how | CPU |
| [usfiscaldata](economics/economic_analysis/usfiscaldata) | economic_analysis | api | CPU |

</details>

<details>
<summary><strong>量子计算  (4 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [cirq](quantum/quantum/cirq) | quantum | python package | CPU |
| [pennylane](quantum/quantum/pennylane) | quantum | python package | GPU/CPU |
| [qiskit](quantum/quantum/qiskit) | quantum | python package | CPU |
| [qutip](quantum/quantum/qutip) | quantum | python package | CPU |

</details>



<details>
<summary><strong>天文学  (1 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [astropy](astronomy/astronomical_analysis/astropy) | astronomical-analysis | python package | CPU |

</details>

<details>
<summary><strong>能源  (2 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [cbssolver](energy/energy/cbssolver) | energy | AI model | NPU |
| [powerflownet](energy/energy/powerflownet) | energy | AI model | NPU |

</details>

<details>
<summary><strong>机器学习  (13 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [aeon](machine_learning/machine_learning/aeon) | machine_learning | python package | CPU |
| [pymc](machine_learning/machine_learning/pymc) | machine_learning | python package | CPU |
| [pymoo](machine_learning/machine_learning/pymoo) | machine_learning | python package | CPU |
| [pytorch-lightning](machine_learning/machine_learning/pytorch-lightning) | machine_learning | python package | GPU/CPU |
| [scikit-learn](machine_learning/machine_learning/scikit-learn) | machine_learning | python package | CPU |
| [seaborn](machine_learning/machine_learning/seaborn) | machine_learning | python package | CPU |
| [shap](machine_learning/machine_learning/shap) | machine_learning | python package | CPU/GPU |
| [simpy](machine_learning/machine_learning/simpy) | machine_learning | python package | CPU |
| [stable-baselines3](machine_learning/machine_learning/stable-baselines3) | machine_learning | python package | CPU |
| [statsmodels](machine_learning/machine_learning/statsmodels) | machine_learning | python package | CPU |
| [torch-geometric](machine_learning/machine_learning/torch-geometric) | machine_learning | python package | GPU/CPU |
| [transformers](machine_learning/machine_learning/transformers) | machine_learning | python package | GPU/CPU |
| [umap-learn](machine_learning/machine_learning/umap-learn) | machine_learning | python package | CPU |

</details>

<details>
<summary><strong>研究工具  (41 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| [bgpt-paper-search](research_tools/database_and_knowledge/bgpt-paper-search) | database_and_knowledge | know-how | CPU |
| [citation-management](research_tools/database_and_knowledge/citation-management) | database_and_knowledge | know-how | CPU |
| [dask](research_tools/data_analysis_and_processing/dask) | data_analysis_and_processing | python package | CPU |
| [datacommons-client](research_tools/general_tools/datacommons-client) | general_tools | api | CPU |
| [exploratory-data-analysis](research_tools/data_analysis_and_processing/exploratory-data-analysis) | data_analysis_and_processing | know-how | CPU |
| [generate-image](research_tools/visualization_tools/generate-image) | visualization_tools | know-how | CPU |
| [hpc-orchestration](research_tools/hpc_orchestration/hpc-orchestration) | hpc_orchestration | HPC software | CPU/GPU |
| [hpc-paraview](research_tools/visualization_tools/hpc-paraview) | visualization_tools | HPC software | CPU |
| [hypothesis-generation](research_tools/general_tools/hypothesis-generation) | general_tools | know-how | CPU |
| [infographics](research_tools/visualization_tools/infographics) | visualization_tools | know-how | CPU |
| [latex-posters](research_tools/scientific_writing/latex-posters) | scientific_writing | know-how | CPU |
| [literature-review](research_tools/database_and_knowledge/literature-review) | database_and_knowledge | know-how | CPU |
| [markdown-mermaid-writing](research_tools/scientific_writing/markdown-mermaid-writing) | scientific_writing | know-how | CPU |
| [markitdown](research_tools/scientific_writing/markitdown) | scientific_writing | python package | CPU |
| [matplotlib](research_tools/visualization_tools/matplotlib) | visualization_tools | python package | CPU |
| [model-skill-creator](research_tools/general_tools/model-skill-creator) | general_tools | know-how | CPU |
| [networkx](research_tools/visualization_tools/networkx) | visualization_tools | python package | CPU |
| [openalex-database](research_tools/database_and_knowledge/openalex-database) | database_and_knowledge | api | CPU |
| [open-notebook](research_tools/scientific_writing/open-notebook) | scientific_writing | know-how | CPU |
| [paper-2-web](research_tools/scientific_writing/paper-2-web) | scientific_writing | know-how | CPU |
| [peer-review](research_tools/database_and_knowledge/peer-review) | database_and_knowledge | know-how | CPU |
| [plotly](research_tools/visualization_tools/plotly) | visualization_tools | know-how | CPU |
| [polars](research_tools/data_analysis_and_processing/polars) | data_analysis_and_processing | python package | CPU |
| [pptx-posters](research_tools/scientific_writing/pptx-posters) | scientific_writing | know-how | CPU |
| [pyzotero](research_tools/database_and_knowledge/pyzotero) | database_and_knowledge | python package | CPU |
| [research-grants](research_tools/database_and_knowledge/research-grants) | database_and_knowledge | know-how | CPU |
| [research-lookup](research_tools/database_and_knowledge/research-lookup) | database_and_knowledge | know-how | CPU |
| [scientific-brainstorming](research_tools/general_tools/scientific-brainstorming) | general_tools | know-how | CPU |
| [scientific-critical-thinking](research_tools/general_tools/scientific-critical-thinking) | general_tools | know-how | CPU |
| [scientific-schematics](research_tools/visualization_tools/scientific-schematics) | visualization_tools | know-how | CPU |
| [scientific-slides](research_tools/scientific_writing/scientific-slides) | scientific_writing | know-how | CPU |
| [scientific-visualization](research_tools/visualization_tools/scientific-visualization) | visualization_tools | know-how | CPU |
| [scientific-writing](research_tools/scientific_writing/scientific-writing) | scientific_writing | know-how | CPU |
| [skill-creator](research_tools/general_tools/skill-creator) | general_tools | know-how | CPU |
| [statistical-analysis](research_tools/data_analysis_and_processing/statistical-analysis) | data_analysis_and_processing | know-how | CPU |
| [timesfm-forecasting](research_tools/data_analysis_and_processing/timesfm-forecasting) | data_analysis_and_processing | python package | CPU |
| [tooluniverse-literature-deep-research](research_tools/database_and_knowledge/tooluniverse-literature-deep-research) | database_and_knowledge | know-how | CPU |
| [uspto-database](research_tools/database_and_knowledge/uspto-database) | database_and_knowledge | api | CPU |
| [vaex](research_tools/data_analysis_and_processing/vaex) | data_analysis_and_processing | python package | CPU |
| [venue-templates](research_tools/scientific_writing/venue-templates) | scientific_writing | know-how | CPU |

</details>

---


## ⚙️ 快速开始与集成
MindScienceSkills 可以与 Hermes Agent、OpenClaw、Claude Code、JiuwenClaw等智能体无缝集成，也支持直接接入您专属的 AI 科研助手中。

### 🔌 以 OpenClaw 为例

#### 方式一：我是Agent

```
请先检查是否已下载MindScienceSkills仓，若未安装，请克隆https://gitcode.com/mindspore-lab/mindscience.git
， 并将MindScienceSkills对应领域目录下的所有文件夹拷贝至你的工作空间的skills目录下。
```

#### 方式二：我是Human

**Step 1：clone仓**

```
https://gitcode.com/mindspore-lab/mindscience.git
```

**Step 2：拷贝skill到对应Agent的skill文件夹下**


```
cp -r MindScienceSkills/<领域>/<子领域>/* ~/.openclaw/workspace/skills
```

或者在`~/.openclaw/openclaw.json`配置skill路径：

```json
{
  "skills": {
    "load": {
      "extraDirs": [
        "/your/custom/skills/path",
      ],
      "watch": true
    }
  }
}
```

> [!NOTE]
> JiuwenClaw、Deep Agents、MindScienceAgent等框架暂不支持 `skills` 目录下嵌套层级，可以直接运行脚本[flatten_skills.sh](flatten_skills.sh)将`skill`目录展平。


## ⚙️ 应用案例
基于本项目Skills，我们依托智能体框架MindScienceAgent搭建了若干案例：

| 应用场景  | skills                             | 案例链接 |
| --------| ---------------------------------- | -------- |
| 化学反应预测（奥林匹克竞赛题）        |         chemistry-query                   |      [olympiad-chemistry.ipynb](../MindScienceAgent/examples/frontierscience/olympiad-chemistry.ipynb)       |
|     钙钛矿掺杂材料的仿真计算     | doped-perovskite-structure-analysis |    [materials_simulation.ipynb](../MindScienceAgent/examples/materials_simulation.ipynb)      |
| 聚苯胺薄膜合成实验设计   |             liquid-dispensing-workstation 等                  |     [效果图](../MindScienceAgent/examples/experiments_design/)        |



## ⚙️ 致谢与开源参考

本项目诚挚感谢相关优秀开源项目的贡献。我们对这些开源Skills进行了集成、重分类与优化。

| 项目                                | 链接                                                   |
| ----------------------------------- | ------------------------------------------------------ |
| K-Dense-AI/claude-scientific-skills | https://github.com/K-Dense-AI/claude-scientific-skills |
| mims-harvard/ToolUniverse           | https://github.com/mims-harvard/ToolUniverse           |
| wu-yc/LabClaw                       | https://github.com/wu-yc/LabClaw                       |
| SciMate-AI/HPC-Skills               | https://github.com/SciMate-AI/HPC-Skills               |

本项目的很多核心Skills基于以下项目构建，感谢这些开源项目的贡献：
| 项目                                | 链接                                                   |
| ----------------------------------- | ------------------------------------------------------ |
|     FitDock         |        http://cao.labshare.cn/fitdock       |    
|     AbRSA         |        http://cao.labshare.cn/AbRSA       |    
|     Abalign         |        http://cao.labshare.cn/abalign       |    
|     AbCVista         |       https://github.com/JZongf/AbCVista       |    
|Ascend AI4Science             | https://gitcode.com/AI4Science          |
|  NVIDIA/biomeno                                |    https://github.com/NVIDIA/bionemo-framework                                                    |
|  NVIDIA/physicsnemo                            |    https://github.com/NVIDIA/physicsnemo                                                 |

本项目化学、生物等领域专业技能库等特性，与中国科学技术大学江俊教授团队、四川大学曹洋教授团队联合构建，昇思、昇腾模型类skills集成了诸多合作团队的原创成果和生态贡献，感谢所有共创团队在专业领域的深度指导与合作。

## 🤝 参与贡献

我们非常欢迎各位开发者、科研人员以及领域专家参与共建，共同打造更强大、更可靠的MindScienceSkills！您可以为我们贡献您的Skill，或者优化我们已有的Skill。

我们采用标准的 GitHub Pull Request 工作流。请按照以下步骤提交您的代码：

1. **Fork** 本仓库到您的个人账号。
2. 创建您的特性分支 (`git checkout -b feature/Add-Your-Amazing-Skill`)。
3. 编写代码并提交修改 (`git commit -m 'feat: 新增/优化 XXX 技能'`)，请尽量在提交信息中简要说明该Skill的应用场景或优化点。
4. 将您的分支推送到远程仓库 (`git push origin feature/Add-Your-Amazing-Skill`)。
5. 在本仓库新建一个 **Pull Request (PR)**，我们将尽快进行代码审查（Code Review）并与您交流探讨。

再次感谢您对 MindScienceSkills 的关注与贡献！