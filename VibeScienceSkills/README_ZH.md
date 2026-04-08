# Vibe Science Skills: 面向通用科学分析、仿真与实验的专业技能库

[![Skills](https://img.shields.io/badge/skills-300-blue?style=flat-square)](skills/)
[![Hardware: GPU & NPU](https://img.shields.io/badge/Hardware-GPU%20%7C%20NPU-lightgrey)](#)
[![Agent Skills](https://img.shields.io/badge/Standard-Agent_Skills-blueviolet.svg)](https://agentskills.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#参与贡献)
[![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](LICENSE)

Vibe Science Skills 是一个专为通用科学研究打造的开源智能体技能库，全面覆盖生物医药、化学材料、流体、地球科学、电磁等核心科学领域。

本仓库目前已内置了 300 +专业科研技能（Skills），涵盖了参数配置繁琐的领域专用软件、各领域前沿 AI4S 模型、各学科专用的 Python 数据分析与处理工具库等，覆盖绝大部分日常科学分析、仿真与实验场景中的基础工具需求，降低 AI 跨学科调用的技术门槛。

Vibe Science Skills 可以与 OpenClaw、Claude Code、JiuwenClaw等智能体无缝集成，也支持直接接入您专属的 AI 科研助手中。

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
| **💻 领域专业软件类**         | hpc-vasp、hpc-gromacs                 | 内置了输入构建、参数调整、计算结果分析等流程，解决专业软件参数空间庞大、配置繁琐的问题。借助这类skills，Agent可自主完成输入文件生成、计算执行及结果解析。 |
| **🐍 领域Python工具库** | deepchem、rdkit      | 覆盖各学科的专业数据处理、特征计算等工具库。             |
| **🧠 AI4S模型**           | alphafold3、proteinix | 集成各领域业界核心AI4S模型，对于用户自建模型，提供模型skill自动构建能力，让AI科研助手可以快速调用用户模型。 |
| **💡 Know-How**        |    doped-perovskite-structure-analysis   | 沉淀了顶尖实验室“隐性知识”，将复杂的长链路科研任务固化为skill，为Agent提供专家级步骤指导。 |
| **🎛️ 实验室工作站**     | dual-station-electrochemical-workstation、centrifuge-workstation                      |  连接与集成物理实验设备，支持湿实验与计算模拟的融合操作，推动科研实验的智能化与自动化。                                                            |
| **📚 API**              | openalex-database、pubchem-database          | 涵盖文献搜索、数据获取等通用科研能力。         |


### 2. 高可靠调用：提升科学工具执行成功率
在真实科研应用场景中，Agent调用技能时常遇报错，导致执行受阻。原因主要有两点：一是技能本身说明模糊（如环境依赖、版本信息等），导致Agent在执行工具时容易报错；二是像 VASP 这类计算软件过于复杂，参数空间极大，Agent仅凭简单的模板，很难针对实际场景正确调参。

为提升技能的调用成功率，我们对模型类和软件类技能进行了优化：

* 🧠 **模型类技能的自动试错：** 我们开发了 `Model-Skill-Creator` 来规范技能的构建流程。在将各类模型转换为统一技能后，我们会先让 Agent 进行实际运行测试。如果遇到报错，Agent 会根据错误信息尝试修正代码。我们将最终测试通过的可用代码保留下来，不断迭代从而提升技能可靠性。

  

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
<summary><strong>生物  (91 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| alphafold-database | database_and_knowledge | api | CPU |
| anndata | general_tools | python package | CPU |
| arboreto | multi_omics_integration | python package | CPU |
| biorxiv-database | database_and_knowledge | api | CPU |
| cellxgene-census | database_and_knowledge | api | CPU |
| deepchem | protein_and_structure | python package | GPU/CPU |
| deeptools | general_tools | know-how | CPU |
| etetoolkit | genomics_and_genetics | python package | CPU |
| flowio | general_tools | python package | CPU |
| geniml | general_tools | python package | GPU/CPU |
| gget | database_and_knowledge | python package | CPU |
| ena-database | database_and_knowledge | api | CPU |
| gtars | bioinformatics | python package | CPU |
| ensembl-database | database_and_knowledge | api | CPU |
| gene-database | database_and_knowledge | api | CPU |
| hpc-gromacs | protein_and_structure | HPC software | CPU/GPU |
| geo-database | database_and_knowledge | api | CPU |
| gnomad-database | database_and_knowledge | api | CPU |
| gtex-database | database_and_knowledge | api | CPU |
| lamindb | general_tools | python package | CPU |
| matchms | protein_and_structure | python package | CPU |
| Microbiome Research | genomics_and_genetics | know-how | CPU |
| hmdb-database | database_and_knowledge | api | CPU |
| interpro-database | database_and_knowledge | api | CPU |
| jaspar-database | database_and_knowledge | api | CPU |
| phylogenetics | genomics_and_genetics | python package | CPU |
| Protein Interaction Network Analysis | multi_omics_integration | know-how | CPU |
| pydeseq2 | general_tools | python package | CPU |
| pyopenms | protein_and_structure | python package | CPU |
| pysam | general_tools | python package | CPU |
| kegg-database | database_and_knowledge | api | CPU |
| scanpy | transcriptomics_and_sc_omics | python package | CPU |
| scikit-bio | general_tools | python package | CPU |
| scvelo | transcriptomics_and_sc_omics | python package | CPU |
| scvi-tools | transcriptomics_and_sc_omics | python package | GPU/CPU |
| metabolomics-workbench-database | database_and_knowledge | api | CPU |
| tiledbvcf | general_tools | python package | CPU |
| tooluniverse-aging-senescence | multi_omics_integration | know-how | CPU |
| tooluniverse-antibody-engineering | protein_and_structure | know-how | CPU |
| tooluniverse-comparative-genomics | genomics_and_genetics | know-how | CPU |
| tooluniverse-crispr-screen-analysis | genomics_and_genetics | know-how | CPU |
| tooluniverse-epigenomics | genomics_and_genetics | know-how | CPU |
| tooluniverse-epigenomics-chromatin | genomics_and_genetics | know-how | CPU |
| tooluniverse-expression-data-retrieval | database_and_knowledge | know-how | CPU |
| tooluniverse-functional-genomics-screens | genomics_and_genetics | know-how | CPU |
| tooluniverse-gene-enrichment | transcriptomics_and_sc_omics | know-how | CPU |
| tooluniverse-gene-regulatory-networks | multi_omics_integration | know-how | CPU |
| tooluniverse-gpcr-structural-pharmacology | protein_and_structure | know-how | CPU |
| tooluniverse-hla-immunogenomics | multi_omics_integration | know-how | CPU |
| tooluniverse-lipidomics | protein_and_structure | know-how | CPU |
| tooluniverse-metabolomics | protein_and_structure | know-how | CPU |
| tooluniverse-metabolomics-analysis | protein_and_structure | know-how | CPU |
| tooluniverse-metabolomics-pathway | protein_and_structure | know-how | CPU |
| tooluniverse-metagenomics-analysis | genomics_and_genetics | know-how | CPU |
| tooluniverse-model-organism-genetics | genomics_and_genetics | know-how | CPU |
| tooluniverse-multi-omics-integration | multi_omics_integration | know-how | CPU |
| tooluniverse-noncoding-rna | transcriptomics_and_sc_omics | know-how | CPU |
| tooluniverse-phylogenetics | genomics_and_genetics | know-how | CPU |
| tooluniverse-plant-genomics | genomics_and_genetics | know-how | CPU |
| tooluniverse-population-genetics | genomics_and_genetics | know-how | CPU |
| tooluniverse-population-genetics-1000genomes | genomics_and_genetics | know-how | CPU |
| tooluniverse-protein-modification-analysis | protein_and_structure | know-how | CPU |
| tooluniverse-protein-structure-prediction | protein_and_structure | know-how | CPU |
| tooluniverse-protein-structure-retrieval | protein_and_structure | know-how | CPU |
| tooluniverse-protein-therapeutic-design | protein_and_structure | know-how | CPU |
| tooluniverse-proteomics-analysis | multi_omics_integration | know-how | CPU |
| tooluniverse-proteomics-data-retrieval | database_and_knowledge | know-how | CPU |
| tooluniverse-regulatory-genomics | transcriptomics_and_sc_omics | know-how | CPU |
| tooluniverse-regulatory-variant-analysis | genomics_and_genetics | know-how | CPU |
| tooluniverse-rnaseq-deseq2 | transcriptomics_and_sc_omics | know-how | CPU |
| tooluniverse-sequence-analysis | genomics_and_genetics | know-how | CPU |
| tooluniverse-sequence-retrieval | database_and_knowledge | know-how | CPU |
| tooluniverse-single-cell | transcriptomics_and_sc_omics | know-how | CPU |
| tooluniverse-spatial-omics-analysis | transcriptomics_and_sc_omics | know-how | CPU |
| tooluniverse-spatial-transcriptomics | transcriptomics_and_sc_omics | know-how | CPU |
| tooluniverse-statistical-modeling | general_tools | know-how | CPU |
| tooluniverse-stem-cell-organoid | multi_omics_integration | know-how | CPU |
| tooluniverse-structural-proteomics | protein_and_structure | know-how | CPU |
| tooluniverse-structural-variant-analysis | genomics_and_genetics | know-how | CPU |
| tooluniverse-systems-biology | multi_omics_integration | know-how | CPU |
| tooluniverse-variant-analysis | genomics_and_genetics | know-how | CPU |
| monarch-database | database_and_knowledge | api | CPU |
| adaptyv | protein_and_structure | python package | CPU |
| bioservices | database_and_knowledge | python package | CPU |
| cobrapy | protein_and_structure | python package | CPU |
| diffdock | protein_and_structure | python package | CPU/GPU |
| esm | protein_and_structure | python package | CPU/GPU |
| pdb-database | database_and_knowledge | api | CPU |
| reactome-database | database_and_knowledge | api | CPU |
| string-database | database_and_knowledge | api | CPU |
| uniprot-database | database_and_knowledge | api | CPU |

</details>

<details>
<summary><strong>医学与药学  (75 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| bindingdb-database | database_and_knowledge | api | CPU |
| brenda-database | database_and_knowledge | api | CPU |
| cbioportal-database | database_and_knowledge | api | CPU |
| chembl-database | database_and_knowledge | api | CPU |
| clinical-decision-support | clinical | know-how | CPU |
| clinical-reports | clinical | know-how | CPU |
| clinicaltrials-database | database_and_knowledge | api | CPU |
| clinpgx-database | database_and_knowledge | api | CPU |
| clinvar-database | database_and_knowledge | api | CPU |
| gwas-database | database_and_knowledge | api | CPU |
| cosmic-database | database_and_knowledge | api | CPU |
| depmap | cancer_genomics | python package | CPU |
| drugbank-database | database_and_knowledge | api | CPU |
| fda-database | database_and_knowledge | api | CPU |
| histolab | medical_imaging | python package | CPU |
| imaging-data-commons | medical_imaging | know-how | CPU |
| medchem | drug_development | python package | CPU |
| neurokit2 | medical_imaging | python package | CPU |
| neuropixels-analysis | medical_imaging | know-how | CPU |
| opentargets-database | database_and_knowledge | api | CPU |
| pathml | medical_imaging | python package | CPU/GPU |
| pubmed-database | database_and_knowledge | api | CPU |
| pydicom | medical_imaging | python package | CPU |
| pyhealth | clinical | python package | GPU/CPU |
| pytdc | drug_development | python package | CPU |
| scikit-survival | cancer_genomics | python package | CPU |
| tooluniverse-acmg-variant-classification | genomic_medicine | know-how | CPU |
| tooluniverse-admet-prediction | drug_development | know-how | CPU |
| tooluniverse-adverse-event-detection | drug_development | know-how | CPU |
| tooluniverse-adverse-outcome-pathway | drug_development | know-how | CPU |
| tooluniverse-binder-discovery | drug_development | know-how | CPU |
| tooluniverse-cancer-classification | cancer_genomics | know-how | CPU |
| tooluniverse-cancer-genomics-tcga | cancer_genomics | know-how | CPU |
| tooluniverse-cancer-variant-interpretation | cancer_genomics | know-how | CPU |
| tooluniverse-cell-line-profiling | cancer_genomics | know-how | CPU |
| tooluniverse-clinical-data-integration | clinical | know-how | CPU |
| tooluniverse-clinical-guidelines | clinical | know-how | CPU |
| tooluniverse-clinical-trial-design | clinical | know-how | CPU |
| tooluniverse-clinical-trial-matching | clinical | know-how | CPU |
| tooluniverse-disease-research | clinical | know-how | CPU |
| tooluniverse-drug-drug-interaction | drug_development | know-how | CPU |
| tooluniverse-drug-mechanism-research | drug_development | know-how | CPU |
| tooluniverse-drug-regulatory | drug_development | know-how | CPU |
| tooluniverse-drug-repurposing | drug_development | know-how | CPU |
| tooluniverse-drug-research | drug_development | know-how | CPU |
| tooluniverse-drug-target-validation | drug_development | know-how | CPU |
| tooluniverse-gene-disease-association | genomic_medicine | know-how | CPU |
| tooluniverse-gwas-drug-discovery | genomic_medicine | know-how | CPU |
| tooluniverse-gwas-finemapping | genomic_medicine | know-how | CPU |
| tooluniverse-gwas-snp-interpretation | genomic_medicine | know-how | CPU |
| tooluniverse-gwas-study-explorer | genomic_medicine | know-how | CPU |
| tooluniverse-gwas-trait-to-gene | genomic_medicine | know-how | CPU |
| tooluniverse-image-analysis | medical_imaging | know-how | CPU |
| tooluniverse-immune-repertoire-analysis | drug_development | know-how | CPU |
| tooluniverse-immunology | drug_development | know-how | CPU |
| tooluniverse-immunotherapy-response-prediction | cancer_genomics | know-how | CPU |
| tooluniverse-infectious-disease | clinical | know-how | CPU |
| tooluniverse-kegg-disease-drug | drug_development | know-how | CPU |
| tooluniverse-multiomic-disease-characterization | genomic_medicine | know-how | CPU |
| tooluniverse-network-pharmacology | drug_development | know-how | CPU |
| tooluniverse-pathway-disease-genetics | genomic_medicine | know-how | CPU |
| tooluniverse-pharmacogenomics | drug_development | know-how | CPU |
| tooluniverse-pharmacovigilance | drug_development | know-how | CPU |
| tooluniverse-polygenic-risk-score | genomic_medicine | know-how | CPU |
| tooluniverse-precision-medicine-stratification | clinical | know-how | CPU |
| tooluniverse-precision-oncology | cancer_genomics | know-how | CPU |
| tooluniverse-rare-disease-diagnosis | clinical | know-how | CPU |
| tooluniverse-rare-disease-genomics | genomic_medicine | know-how | CPU |
| tooluniverse-target-research | drug_development | know-how | CPU |
| tooluniverse-toxicology | drug_development | know-how | CPU |
| tooluniverse-vaccine-design | drug_development | know-how | CPU |
| tooluniverse-variant-functional-annotation | genomic_medicine | know-how | CPU |
| tooluniverse-variant-interpretation | genomic_medicine | know-how | CPU |
| tooluniverse-variant-to-mechanism | genomic_medicine | know-how | CPU |
| torchdrug | drug_development | python package | GPU/CPU |

</details>

<details>
<summary><strong>化学与材料  (23 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| pubchem-database | database_and_knowledge | api | CPU |
| zinc-database | database_and_knowledge | api | CPU |
| datamol | cheminformatics | python package | CPU |
| hpc-cp2k | quantum_chemistry | HPC software | CPU/GPU |
| hpc-feff | quantum_chemistry | HPC software | CPU |
| hpc-gaussian | quantum_chemistry | HPC software | CPU |
| hpc-lammps | molecular_dynamics | HPC software | CPU/GPU |
| hpc-nwchem | quantum_chemistry | HPC software | CPU |
| hpc-orca | quantum_chemistry | HPC software | CPU/GPU |
| hpc-psi4 | quantum_chemistry | HPC software | CPU |
| hpc-pyscf | quantum_chemistry | HPC software | CPU |
| hpc-quantum-espresso | quantum_chemistry | HPC software | CPU/GPU |
| hpc-vasp | quantum_chemistry | HPC software | CPU/GPU |
| hpc-xtb | quantum_chemistry | HPC software | CPU |
| molecular-dynamics | molecular_dynamics | know-how | CPU |
| molfeat | cheminformatics | python package | CPU/GPU |
| pymatgen | general_tools | python package | CPU |
| rdkit | cheminformatics | python package | CPU |
| tooluniverse-chemical-compound-retrieval | cheminformatics | know-how | CPU |
| tooluniverse-chemical-safety | general_tools | know-how | CPU |
| tooluniverse-chemical-sourcing | general_tools | know-how | CPU |
| tooluniverse-small-molecule-discovery | cheminformatics | know-how | CPU |
| tooluniverse-electron-microscopy | general_tools | know-how | CPU |

</details>

<details>
<summary><strong>地球科学  (12 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| cfgrib | meteorology | python package | CPU |
| eccodes | meteorology | python package | CPU |
| geomaster | geospatial | python package | CPU |
| geopandas | geospatial | python package | CPU |
| hpc-openfwi | geophysics | HPC software | CPU |
| hpc-wrf | meteorology | HPC software | CPU |
| metpy | meteorology | python package | CPU |
| py-art | meteorology | python package | CPU |
| satpy | meteorology | python package | CPU |
| siphon | meteorology | python package | CPU |
| wrf-python | meteorology | python package | CPU |
| xesmf | meteorology | python package | CPU |

</details>



<details>
<summary><strong>电磁学  (7 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| gprmax | electromagnetics | python package | CPU |
| hpc-cst | electromagnetics | HPC software | CPU/GPU |
| meep | electromagnetics | python package | CPU |
| ngsolve | electromagnetics | python package | CPU |
| pyaedt | electromagnetics | python package | CPU |
| pyfemm | electromagnetics | python package | CPU |
| scikit-rf | electromagnetics | python package | CPU |

</details>

<details>
<summary><strong>流体力学  (12 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| cantera | fluid_dynamics | python package | CPU |
| coolprop | fluid_dynamics | python package | CPU |
| fenics | fluid_dynamics | python package | CPU |
| fipy | fluid_dynamics | python package | CPU |
| fluidsim | fluid_dynamics | python package | CPU |
| hpc-openfoam | fluid_dynamics | HPC software | CPU |
| hpc-su2 | fluid_dynamics | HPC software | CPU/GPU |
| phiflow | fluid_dynamics | python package | CPU |
| pyfoam | fluid_dynamics | python package | CPU |
| pysph | fluid_dynamics | python package | CPU |
| pyvista | fluid_dynamics | python package | CPU |
| hpc-fenics | fluid_dynamics | HPC software | CPU |

</details>

<details>
<summary><strong>数学  (4 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| hpc-fftw | numerical-computing | HPC software | CPU |
| hpc-openblas | numerical-computing | HPC software | CPU |
| matlab | numerical-computing | python package | CPU |
| sympy | numerical-computing | python package | CPU |

</details>



<details>
<summary><strong>经济学  (4 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| fred-economic-data | economic-analysis | api | CPU |
| usfiscaldata | economic-analysis | api | CPU |
| edgartools | economic-analysis | know-how | CPU |
| market-research-reports | economic-analysis | know-how | CPU |

</details>

<details>
<summary><strong>量子计算  (4 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| cirq | quantum | python package | CPU |
| pennylane | quantum | python package | GPU/CPU |
| qiskit | quantum | python package | CPU |
| qutip | quantum | python package | CPU |

</details>



<details>
<summary><strong>天文学  (1 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| astropy | astronomical-analysis | python package | CPU |

</details>

<details>
<summary><strong>机器学习与人工智能  (13 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| aeon | ML | python package | CPU |
| pymc | ML | python package | CPU |
| pymoo | ML | python package | CPU |
| pytorch-lightning | AI | python package | GPU/CPU |
| scikit-learn | ML | python package | CPU |
| seaborn | ML | python package | CPU |
| shap | ML | python package | CPU/GPU |
| simpy | AI | python package | CPU |
| stable-baselines3 | AI | python package | CPU |
| statsmodels | ML | python package | CPU |
| torch-geometric | AI | python package | GPU/CPU |
| transformers | AI | python package | GPU/CPU |
| umap-learn | ML | python package | CPU |

</details>

<details>
<summary><strong>研究工具  (38 skills)</strong></summary>

| skill | 子领域 | skill类型 | 支持的硬件 |
|:------|:------|:---------|:-----------|
| bgpt-paper-search | database_and_knowledge | know-how | CPU |
| citation-management | database_and_knowledge | know-how | CPU |
| dask | data_analysis_and_processing | python package | CPU |
| datacommons-client | general_tools | api | CPU |
| exploratory-data-analysis | data_analysis_and_processing | know-how | CPU |
| generate-image | visualization_tools | know-how | CPU |
| hpc-orchestration | hpc_orchestration | HPC software | CPU/GPU |
| hpc-paraview | visualization_tools | HPC software | CPU |
| hypothesis-generation | general_tools | know-how | CPU |
| infographics | visualization_tools | know-how | CPU |
| latex-posters | scientific_writing | know-how | CPU |
| literature-review | database_and_knowledge | know-how | CPU |
| markdown-mermaid-writing | scientific_writing | know-how | CPU |
| markitdown | scientific_writing | python package | CPU |
| matplotlib | visualization_tools | python package | CPU |
| networkx | visualization_tools | python package | CPU |
| openalex-database | database_and_knowledge | api | CPU |
| open-notebook | scientific_writing | know-how | CPU |
| paper-2-web | scientific_writing | know-how | CPU |
| peer-review | database_and_knowledge | know-how | CPU |
| plotly | visualization_tools | know-how | CPU |
| polars | data_analysis_and_processing | python package | CPU |
| pptx-posters | scientific_writing | know-how | CPU |
| pyzotero | database_and_knowledge | python package | CPU |
| research-grants | database_and_knowledge | know-how | CPU |
| research-lookup | database_and_knowledge | know-how | CPU |
| scientific-brainstorming | general_tools | know-how | CPU |
| scientific-critical-thinking | general_tools | know-how | CPU |
| scientific-schematics | visualization_tools | know-how | CPU |
| scientific-slides | scientific_writing | know-how | CPU |
| scientific-visualization | visualization_tools | know-how | CPU |
| scientific-writing | scientific_writing | know-how | CPU |
| statistical-analysis | data_analysis_and_processing | know-how | CPU |
| timesfm-forecasting | data_analysis_and_processing | python package | CPU |
| tooluniverse-literature-deep-research | database_and_knowledge | know-how | CPU |
| uspto-database | database_and_knowledge | api | CPU |
| vaex | data_analysis_and_processing | python package | CPU |
| venue-templates | scientific_writing | know-how | CPU |

</details>

---


## ⚙️ 快速开始与集成
Vibe Science Skills 可以与 OpenClaw、Claude Code、JiuwenClaw等智能体无缝集成，也支持直接接入您专属的 AI 科研助手中。

### 🔌 以 OpenClaw 为例

#### 方式一：我是Agent

```
请先检查是否已下载Vibe Science Skills仓，若未安装，请克隆https://gitcode.com/mindspore-lab/mindscience.git
， 并将mindscience/skills目录下的所有文件夹拷贝至你的工作空间的skills目录下。
```

#### 方式二：我是Human

**Step 1：clone仓**

```
https://gitcode.com/mindspore-lab/mindscience.git
```

**Step 2：拷贝skill到对应Agent的skill文件夹下**


```
cp -r mindscience/skills/* ~/.openclaw/workspace/skills
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

**Step 3：在OpenClaw中使用skills**
[添加视频]

> [!NOTE]
> JiuwenClaw、deepagents等框架暂不支持 `skills` 目录下嵌套层级，可以直接运行脚本`flatten_skills.sh`将`skill`目录展平。


## ⚙️ 应用案例
基于本项目Skills，我们依托智能体框架VibeScienceAgent[添加超链接]搭建了若干案例：

| 应用场景  | skills                             | 案例链接 |
| --------| ---------------------------------- | -------- |
|     氨基酸序列设计     | proteinmpnn                           |      [待添加]       |
|     晶体材料的仿真计算与表征     | pymatgen、hpc-vasp |    [待添加]      |
| 电催化-高熵氧化物合成   |                                |     [待添加]        |



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



## 🤝 参与贡献

我们非常欢迎各位开发者、科研人员以及领域专家参与共建，共同打造更强大、更可靠的Vibe Science Skills！您可以为我们贡献您的Skill，或者优化我们已有的Skill。

我们采用标准的 GitHub Pull Request 工作流。请按照以下步骤提交您的代码：

1. **Fork** 本仓库到您的个人账号。
2. 创建您的特性分支 (`git checkout -b feature/Add-Your-Amazing-Skill`)。
3. 编写代码并提交修改 (`git commit -m 'feat: 新增/优化 XXX 技能'`)，请尽量在提交信息中简要说明该Skill的应用场景或优化点。
4. 将您的分支推送到远程仓库 (`git push origin feature/Add-Your-Amazing-Skill`)。
5. 在本仓库新建一个 **Pull Request (PR)**，我们将尽快进行代码审查（Code Review）并与您交流探讨。

再次感谢您对 Vibe Science Skills 的关注与贡献！