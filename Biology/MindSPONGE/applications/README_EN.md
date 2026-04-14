ENGLISH | [简体中文](README.md)

# **MindSPONGE-APPLICATIONS**

- [**MindSPONGE-APPLICATIONS**](#mindsponge-applications)
    - [**Introduction**](#introduction)
    - [**Contents**](#contents)

## **Introduction**

Application is based on MindSPONGE and MindSpore. We aim to provide powerful applications. Any suggestion is welcome.

MindSPONGE also integrates 20 self-developed or third party SOTA models covering some areas, such as molecular representation, structure prediction, property prediction, molecular design and foundation models and so on.

- For Molecular representation, three models of MolCT, SchNet and PhysNet are provided, all of which are deep molecular models based on graph neural network. They can be used to extract feature vectors of small moleculars for subsequent tasks.

- For molecular structure prediction, there are five models: MEGA-Fold, MEGA-EvoGen, MEGA-Assessment, AlphFold Multimer and UFold, which support the prediction of 3D spatial structure of single-chain proteins, complexes and other molecules and 2D structure of RNA.

- For molecular property prediction, it integrates 6 models including KGNN, DeepDR, pafnucy, JTVAE, DeepFRI and GraphDTA, which are capable of predicting protein-small molecule compound affinity, drug-drug reaction, drug-disease association and other functions.

- For molecular design, four models including ProteinMPNN, ESM-IF1, DeepHops and ColabDesign are provided, providing the ability to design large molecular proteins from scratch and small molecules with the same characteristics as target small molecules.

- For molecular basis, there are GROVER and MG-BERT models, both of which are small molecule compound pre-training models. Users can use this pre-training model for downstream tasks in the fields of biological computation and drug design by fine-tuning.

We also provide common tools commonly used in the field of biological computing, such as Multiple Sequence Alignment and Template search for protein structure prediction, and OpenMM Relaxation based on Amber force field, see common_utils directory for more details.

## **Contents**

- SOTA models
    - Molecular Dynamics
        - Protein Relaxation (To be released)
        - Traditional MD (To be released)
    - Molecular Representation
        - MolCT (To be released)
        - SchNet (To be released)
        - PyhsNet (To be released)
    - Structure Prediction
        - [MEGA-Protein](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/MEGAProtein.md)
        - [FAAST&RASP](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/FAAST)
        - [Multimer-AlphaFold](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/afmultimer.md)
        - [UFold](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/UFold.md)
    - Properties Prediction
        - KGNN (To be released)
        - DeepDR (To be released)
        - [pafnucy](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/pafnucy.md)
        - [JTVAE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/JT-VAE)
        - [DeepFRI](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/DeepFri.md)
        - [GraphDTA](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/GraphDTA.MD)
    - Molecular Design
        - [ProteinMPNN](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/ProteinMPNN.MD)
        - [ESM-IF1](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/ESM-IF1.md)
        - [DeepHops](https://gitee.com/mindspore/mindscience/pulls/848)
        - [ColabDesign](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/ColabDesign.md)
    - Foundation Model
        - [GROVER](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/GROVER.MD)
        - [MG-BERT](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/MGBERT.MD)
- Common Utils
    - Multiple Sequence Alignment and Template search
    - OpenMM Relaxation based on Amber force field
