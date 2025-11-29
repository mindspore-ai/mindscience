![MindSPONGE标志](docs/MindSPONGE.png "MindSPONGE logo")

[![PyPI](https://badge.fury.io/py/mindspore.svg)](https://badge.fury.io/py/mindspore)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://gitee.com/mindspore/mindscience/pulls)

## 介绍

从精准药物设计、蛋白质结构预测到个性化医疗，AI正在重塑生命科学的研究范式，极大地加速了科学发现的过程。然而，这一融合也对底层计算工具提出了更高要求：传统的模拟软件难以满足AI模型训练与大规模生物数据处理的效率需求。

MindSpore SPONGE(Simulation Package tOwards Next GEneration molecular modelling)是基于昇思MindSpore的AI计算生物领域套件，包含蛋白质结构、序列及功能预测、基因组结构及语言模型等通用模型应用，旨在于为广大的科研人员、老师及学生提供高效、易用的AI计算生物软件。

## 目录

- [MindSpore MindSPONGE](#mindspore-sponge)
    - [重要消息](#重要消息)
    - [模型应用](#模型应用)
        - [结构预测](#结构预测)
            - [端到端](#端到端)
            - [结构序列联合设计](#结构序列联合设计)
            - [RNA](#rna)
        - [蛋白质语言模型](#蛋白质语言模型)
        - [功能与属性预测](#功能与属性预测)
        - [小分子与相互作用](#小分子与相互作用)
        - [基因组/转录组语言模型](#基因组转录组语言模型)
        - [虚拟细胞](#虚拟细胞)
    - [分子动力学程序库](#分子动力学程序库)
    - [社区](#社区)
        - [MindSpore Science SIG](#mindspore-science-sig)
        - [核心贡献者](#核心贡献者)
        - [合作伙伴](#合作伙伴)
    - [贡献指南](#贡献指南)
    - [许可证](#许可证)

---

## 重要消息

- 🙌`[置顶]` `[进行中]` [**2025 MindScience开源实习任务**](https://gitee.com/mindspore/community/issues/ICIROO)火热进行中！持续发布新任务，欢迎大家认领~！！[[活动详情]](https://mp.weixin.qq.com/s/R-t8-u4ak2fN4gxe13m3Gw)
- 🔥`2025.11.6` 基于MindSpore的RFdiffusion推理发布，支持抗体设计，性能持续提升中！[[代码]](./applications/rf_diffusion)
- 🔥`2025.09.15` 昌平实验室、北京大学高毅勤、刘思睿团队发表基于MindSpore训练的[**GRASP**](https://www.nature.com/articles/s41592-025-02820-1)模型收录《Nature Methods》。该模型整合多源实验信息实现复合物建模，抗体预测超越 AlphaFold3。[[相关新闻]](https://mp.weixin.qq.com/s/OyqGvoIbtZaOTgEM5UFXJw)
- 🔥`2025.7.26` 基于MindSpore的DeepMind版AlphaFold3推理发布，性能持续提升中！[[代码]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/AlphaFold3)
- 🔥`2025.8.18—2025.8.22` MindSpore SPONGE[**暑期学校第五季**](https://mp.weixin.qq.com/s/GAPyziaXcZyGPSB09BeMgg)圆满收官！[[活动回放]](https://mp.weixin.qq.com/s/loUBlJhIWYAn646w3n3wVA)
- 🔥`2025.05.20` 基于MindSpore训练的8亿参数单细胞基础大模型[**CellFM**](https://www.nature.com/articles/s41467-025-59926-5)重磅发布于《Nature Communications》。由中山大学杨跃东研究团队牵头，联合重庆大学曾远松团队，首次实现 1 亿级人类细胞数据高效建模，零样本细胞类型注释等任务性能全面领先。[[相关新闻]](https://mp.weixin.qq.com/s/KW-j92yUfeiS4EFfjZSfLw)
- `2023.12.07` 抗体设计天工大模型荣获“2023 AIIA人工智能十大先锋应用案例”, [[相关新闻]](https://mp.weixin.qq.com/s/UQStKzm0fdXbA4RQgLE8fw)
- `2023.11.10` MSA生成增强模型MEGA-EvoGen论文"Unsupervisedly Prompting AlphaFold2 for Accurate Few-Shot Protein Structure Prediction"发表于计算化学期刊JCTC。[[论文]](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00528?cookieSet=1) [[代码]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/MEGAProtein.md)
- `2023.6.26` MindSPONGE论文"Artificial Intelligence Enhanced Molecular Simulations"发表于计算化学期刊JCTC，同时当选Most Read Articles。[[论文]](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00214)
- `2023.5.31` NMR核磁共振动态蛋白质结构解析方法正式开源，详见论文Assisting and Accelerating NMR Assignment with Restrained Structure Prediction [[论文]](https://www.biorxiv.org/content/10.1101/2023.04.14.536890v1) [[代码]](https://gitee.com/mindspore/mindscience/tree/r0.5/MindSPONGE/applications/research/FAAST/)
- `2023.1.31` MindSPONGE 1.0.0-alpha版本发布，文档介绍可参见MindSpore官网中的[**科学计算套件MindSPONGE模块**](https://mindspore.cn/mindsponge/docs/zh-CN/r1.0.0-alpha/index.html)
- `2022.8.23` 论文"Few-Shot Learning of Accurate Folding Landscape for Protein Structure Prediction" arxiv预印。[[论文]](https://arxiv.org/abs/2208.09652)
- `2022.07.18` 论文"SPONGE: A GPU-Accelerated Molecular Dynamics Package with Enhanced Sampling and AI-Driven Algorithms"发表于期刊Chinese Journal of Chemistry。[[论文]](https://onlinelibrary.wiley.com/doi/epdf/10.1002/cjoc.202100456) [[代码]](https://gitee.com/mindspore/mindscience/tree/dev-md/MindSPONGE/applications/molecular_dynamics)
- `2022.07.09` MEGA-Assessment在CAMEO-QE月榜取得第一名
- `2022.04.21` MEGA-Fold CAMEO竞赛月榜第一。[[相关新闻]](https://www.huawei.com/cn/news/2022/4/mindspore-cameo-protein-ascend)

---

## 模型应用

### 结构预测

#### 端到端

- 🔥AlphaFold3 [[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/AlphaFold3)
- 🔥Protenix `In Progress`
- 🔥RFdiffusion [[Available]](./applications/rf_diffusion)
- Alphafold-Multimer [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/afmultimer.md)
- MEGAProtein [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/MEGAProtein.md)
- FAAST & RASP [[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/FAAST)
- UFold [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/UFold.md)

#### 结构序列联合设计

- ProteinMPNN [[Available]](./applications/proteinmpnn/)
- ESM-IF1 [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ESM-IF1.md)
- ColabDesign [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ColabDesign.md)

#### RNA

- UFold [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/UFold.md)

### 蛋白质语言模型

- Esm2 [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ESM-2.md)
- ProtT5 [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ProtT5.md)

### 功能与属性预测

- DeepFRI [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/DeepFri.md)
- Pafnucy [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/pafnucy.md)

### 小分子与相互作用

- GROVER [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/GROVER.MD)
- GraphDTA [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/GraphDTA.MD)
- MGBERT [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/MGBERT.MD)
- JIT-VAE[[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/JT-VAE)

### 基因组/转录组语言模型

- 🔥Evo2 `In Progress`
- DNABERT [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/DNABERT.MD)
- Geneformer[[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/Geneformer)

### 虚拟细胞

- 🔥MedFormer [[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/medformer)

---

### 分子动力学

MindSPONGE (MindSpore Simulation Package tOwards Next Generation molecular modelling)是
一款基于MindSpore开发的模块化、高通量、端到端可微的下一代智能分子模拟程序库。
[[链接]](https://gitee.com/helloyesterday/mindsponge/blob/develop/README.md)

---

## 社区

### MindSpore Science SIG

[社区官网](https://www.mindspore.cn/sig/MindSpore%20Science)

MindScience是基于MindSpore融合架构打造的科学计算行业套件，包含了业界领先的数据集、基础模型、预置高精度模型和前后处理工具，加速了科学行业应用开发。

### 核心贡献者

- MindSpore AI4S Lab团队: [longyangyang](https://gitee.com/yanglong_unimelb), [Yuheng Wang](https://gitee.com/yuheng_wang)，[chendanyang](https://gitee.com/birfied)，[Jinxl-pp](https://gitee.com/jinxl-pp), [wangbo](https://gitee.com/wangbo572)
- [高毅勤课题组](https://www.chem.pku.edu.cn/gaoyq/):  [杨奕](https://gitee.com/helloyesterday)，[张骏](https://gitee.com/jz_90)，[刘思睿](https://gitee.com/sirui63)，[夏义杰](https://gitee.com/xiayijie)，[陈迪青](https://gitee.com/dechin)，[刘许晗](https://gitee.com/XuhanLiu)

### 合作伙伴

<div class="item1">
    <img src="docs/cooperative_partner/北京大学.png" width=20%>
    &emsp;
    <img src="docs/cooperative_partner/昌平实验室.png" width=20%>
    &emsp;
    <img src="docs/cooperative_partner/深圳湾.jpg" width=20%>
    &emsp;
    <img src="docs/cooperative_partner/西电.png" width=20%>
</div>

---

## 贡献指南

- 如何贡献您的代码，请点击此处查看：[贡献指南](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)

## 许可证

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)