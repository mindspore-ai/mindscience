[ENGLISH](README_EN.md) | 简体中文

# **MindSPONGE-APPLICATIONS**

- [简介](#简介)
- [目录](#目录)

## **简介**

Application底层依托计算生物工具包MindSPONGE以及昇思MindSpore构建。旨在为大家提供丰富的计算生物案例，同时也欢迎大家为MindSPONGE提供更多更优秀的案例。

MindSPONGE还集成了20个自研以及业界主流模型，主要涵盖分子表征，结构预测，性质预测，分子设计和基础模型等多个方向。

- 分子表征方向提供了MolCT，SchNet和PhysNet共3个模型，均为基于图神经网络的深度分子模型，能够提取小分子的特征向量用于后续任务。

- 分子结构预测方向有MEGA-Fold，MEGA-EvoGen，MEGA-Assessment，AlphFold Multimer，UFold共5个模型，支持预测单链蛋白质，复合物等分子3D空间结构以及RNA的二级结构。

- 分子性质预测方向集成了KGNN，DeepDR，pafnucy，JTVAE，DeepFRI，GraphDTA共6个模型，具备蛋白质-小分子化合物亲和性预测，药物-药物反应预测， 药物-疾病关联预测等功能。

- 分子设计方向提供了ProteinMPNN，ESM-IF1，DeepHops，ColabDesign共4个模型，提供从头设计大分子蛋白质以及设计与目标小分子具有相同特性的小分子的能力。

- 分子基础方向有GROVER，MG-BERT共2个模型，均为小分子化合物预训练模型，用户可使用该预训练模型，通过微调的方式完成生物计算，药物设计等领域的下游任务。

我们同时还提供生物计算领域常用的通用工具，比如蛋白质结构预测需要的多重序列比对与模板检索，还有基于Amber力场的OpenMM Relaxation，更多详细信息请参考common_utils目录。

## **目录**

- 业界主流模型
    - 分子动力学
        - 蛋白质松弛 (To be released)
        - 传统分子动力学 (To be released)
    - 分子表征
        - MolCT (To be released)
        - SchNet (To be released)
        - PyhsNet (To be released)
    - 结构预测
        - [MEGA-Protein](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/MEGAProtein.md)
        - [FAAST&RASP](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/FAAST)
        - [Multimer-AlphaFold](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/afmultimer.md)
        - [UFold](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/UFold.md)
    - 性质预测
        - KGNN (To be released)
        - DeepDR (To be released)
        - [pafnucy](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/pafnucy.md)
        - [JTVAE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/JT-VAE)
        - [DeepFRI](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/DeepFri.md)
        - [GraphDTA](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/GraphDTA.MD)
    - 分子设计
        - [ProteinMPNN](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/ProteinMPNN.MD)
        - [ESM-IF1](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/ESM-IF1.md)
        - [DeepHops](https://gitee.com/mindspore/mindscience/pulls/848)
        - [ColabDesign](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/ColabDesign.md)
    - 基础模型
        - [GROVER](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/GROVER.MD)
        - [MG-BERT](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/MGBERT.MD)
- 通用工具
    - 多重序列比对(MSA)&模板检索
    - 结构弛豫
