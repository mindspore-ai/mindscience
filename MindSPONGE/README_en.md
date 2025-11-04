![MindSPONGE标志](docs/MindSPONGE.png "MindSPONGE logo")

[![PyPI](https://badge.fury.io/py/mindspore.svg)](https://badge.fury.io/py/mindspore)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://gitee.com/mindspore/mindscience/pulls)

## Introduction

From precise drug design and protein structure prediction to personalized medicine, AI is reshaping the research paradigm in life sciences, greatly accelerating the process of scientific discovery. However, this integration also imposes higher demands on underlying computational tools: traditional simulation software struggles to meet the efficiency requirements for AI model training and large-scale biological data processing.

MindSpore SPONGE (Simulation Package tOwards Next GEneration molecular modelling) is an AI computing biological suite based on Ascend MindSpore. It includes general model applications such as protein structure, sequence and function prediction, genomic structure, and language models. It aims to provide researchers, educators, and students with efficient and easy-to-use AI computational biology software.

## Contents

- [MindSpore MindSPONGE](#mindspore-sponge)
    - [News](#news)
    - [Models](#models)
        - [Structure Prediction](#structure-prediction)
            - [End2End](#end2end)
            - [Structure-Sequence Co-design](#structure-sequence-co-design)
            - [RNA](#rna)
        - [Protein Language Model](#protein-language-model)
        - [Functionality and Property Prediction](#functionality-and-property-prediction)
        - [Small Molecules Interaction](#small-molecules-interaction)
        - [Genome/Transcriptome Language Model](#genometranscriptomerna)
        - [Virtual Cell](#virtual-cell)
    - [Molecular Dynamics Library](#molecular-dynamics-library)
    - [Community](#community)
        - [MindSpore Science SIG](#mindspore-science-sig)
        - [Core Contributors](#core-contributor)
        - [Partners](#partners)
    - [How to Contribute](#how-to-contribute)
    - [License](#license)

---

## News

- 🙌`[Pinned]` `[In Progress]` [**2025 MindScience Open-source Internship**](https://gitee.com/mindspore/community/issues/ICIROO) is now open! Come apply your tasks! [[Link]](https://mp.weixin.qq.com/s/R-t8-u4ak2fN4gxe13m3Gw)
- 🔥`2025.09.15` The Changping Laboratory, along with Gao Yiqin and Liu Sirui's team from Peking University, published the [**GRASP**](https://www.nature.com/articles/s41592-025-02820-1) model, trained based on MindSpore, in *Nature Methods*. This model integrates multi-source experimental information to enable complex modeling, and its antibody predictions surpass AlphaFold3. [[Link]](https://mp.weixin.qq.com/s/OyqGvoIbtZaOTgEM5UFXJw)
- 🔥`2025.7.26` Now support DeepMind's AlphaFold3 run on Mindspore! [[Try]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/AlphaFold3)
- 🔥`2025.8.18—2025.8.22` MindSpore SPONGE Summer School Season 5 [[Intro]](https://mp.weixin.qq.com/s/GAPyziaXcZyGPSB09BeMgg)  [[Review]](https://mp.weixin.qq.com/s/loUBlJhIWYAn646w3n3wVA)
- 🔥`2025.05.20` The 800-million-parameter single-cell foundational model [**CellFM**](https://www.nature.com/articles/s41467-025-59926-5), trained based on MindSpore, has been prominently released in *Nature Communications*.[[Link]](https://mp.weixin.qq.com/s/KW-j92yUfeiS4EFfjZSfLw)
- `2023.12.07` The Tiangong large model for antibody design won the '2023 AIIA Top 10 Pioneer AI Application Cases' award. [[Link]](https://mp.weixin.qq.com/s/UQStKzm0fdXbA4RQgLE8fw)
- `2023.11.10` The MSA generation enhancement model MEGA-EvoGen paper, 'Unsupervisedly Prompting AlphaFold2 for Accurate Few-Shot Protein Structure Prediction,' was published in *the Journal of Chemical Theory and Computation (JCTC)*. [[Paper]](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00528?cookieSet=1) [[Code]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/MEGAProtein.md)
- `2023.6.26` The MindSPONGE paper "Artificial Intelligence Enhanced Molecular Simulations" was published in the computational chemistry journal JCTC and was also selected as one of the Most Read Articles. [[Paper]](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00214)
- `2023.5.31` The NMR nuclear magnetic resonance dynamic protein structure analysis method has been officially open-sourced. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.04.14.536890v1) [[Code]](https://gitee.com/mindspore/mindscience/tree/r0.5/MindSPONGE/applications/research/FAAST/)
- `2023.1.31` MindSPONGE 1.0.0-alpha released. [[Link]](https://mindspore.cn/mindsponge/docs/zh-CN/r1.0.0-alpha/index.html)
- `2022.07.18` The paper 'SPONGE: A GPU-Accelerated Molecular Dynamics Package with Enhanced Sampling and AI-Driven Algorithms' was published in the journal Chinese Journal of Chemistry.[[Paper]](https://onlinelibrary.wiley.com/doi/epdf/10.1002/cjoc.202100456) [[Code]](https://gitee.com/mindspore/mindscience/tree/dev-md/MindSPONGE/applications/molecular_dynamics)
- `2022.07.09` MEGA-Assessment won the CAMEO-QE Monthly Champion.
- `2022.04.21` MEGA-Fold won the CAMEO Competition Monthly Champion [[Link]](https://www.huawei.com/cn/news/2022/4/mindspore-cameo-protein-ascend)

---

## Models

### Structure Prediction

#### End2End

- 🔥AlphaFold3 [[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/AlphaFold3)
- 🔥Protenix `In Progress`
- 🔥RFdiffusion `In Progress`
- Alphafold-Multimer [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/afmultimer.md)
- MEGAProtein [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/MEGAProtein.md)
- FAAST & RASP [[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/FAAST)

#### Structure-Sequence Co-design

- ProteinMPNN [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ProteinMPNN.MD)
- ESM-IF1 [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ESM-IF1.md)
- ColabDesign [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ColabDesign.md)

#### RNA

- UFold [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/UFold.md)

### Protein Language Model

- Esm2 [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ESM-2.md)
- ProtT5 [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/ProtT5.md)

### Functionality and Property Prediction

- DeepFRI [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/DeepFri.md)
- Pafnucy [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/pafnucy.md)

### Small Molecules Interaction

- GROVER [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/GROVER.MD)
- GraphDTA [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/GraphDTA.MD)
- MGBERT [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/MGBERT.MD)
- JIT-VAE[[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/JT-VAE)

### Genome/Transcriptome Language Model

- 🔥Evo2 `In Progress`
- DNABERT [[Available]](https://gitee.com/mindspore/mindscience/blob/legacy-master/MindSPONGE/applications/model_cards/DNABERT.MD)
- Geneformer[[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/Geneformer)

### Virtual Cell

- 🔥MedFormer [[Available]](https://gitee.com/mindspore/mindscience/tree/legacy-master/MindSPONGE/applications/research/medformer)

---

### Molecular Dynamics Library

MindSPONGE (MindSpore Simulation Package tOwards Next Generation molecular modelling) is a next-generation intelligent molecular simulation library developed based on MindSpore, featuring modularity, high-throughput capabilities, and end-to-end differentiability.
[[Link]](https://gitee.com/helloyesterday/mindsponge/blob/develop/README.md)

---

## Community

### MindSpore Science SIG

[Official Link](https://www.mindspore.cn/sig/MindSpore%20Science)

MindScience is a scientific computing industry suite built on the MindSpore unified architecture, featuring industry-leading datasets, foundational models, pre-configured high-precision models, and pre- and post-processing tools, accelerating the development of applications in the scientific sector.

### Core Contributors

- MindSpore AI4S Lab: [longyangyang](https://gitee.com/yanglong_unimelb), [Yuheng Wang](https://gitee.com/yuheng_wang)，[chendanyang](https://gitee.com/birfied)，[Jinxl-pp](https://gitee.com/jinxl-pp), [wangbo](https://gitee.com/wangbo572)
- [CCME Gao Group](https://www.chem.pku.edu.cn/gaoyq/):  [Yi Yang](https://gitee.com/helloyesterday)，[Jun Zhang](https://gitee.com/jz_90)，[Sirui Liu](https://gitee.com/sirui63)，[Yijie Xia](https://gitee.com/xiayijie)，[Diqing Chen](https://gitee.com/dechin)，[Xuhan Liu](https://gitee.com/XuhanLiu)

### Partners

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

## How to Contribute

Please refer to [link](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md).

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)