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

## **Contents**

- [Molecular Dynamics](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/molecular_dynamics/)
    - [Protein Relaxation](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/molecular_dynamics/protein_relaxation)
    - [Traditional MD](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/molecular_dynamics/tradition)
- Molecular Representation
    - [MolCT](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/cybertron)
    - [SchNet](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/cybertron)
    - [PyhsNet](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/cybertron)
- Structure Prediction
    - [MEGA-Protein](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/)
        - [MEGA-Fold](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/model/fold.py)
        - [MEGA-EvoGen](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/model/evogen.py)
        - [MEGA-Assessment](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/model/assessment.py)
    - [Multimer-AlphaFold](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/Multimer)
    - [UFold](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/UFold)
- Properties Prediction
    - [KGNN](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/KGNN)
    - [DeepDR](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/DeepDR)
    - [pafnucy](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/pafnucy)
    - [JTVAE](https://gitee.com/mindspore/mindscience/pulls/685)
    - [DeepFRI](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/DeepFRI)
    - [GraphDTA](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/GraphDTA)
- Molecular Design
    - [ProteinMPNN](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/ProteinMPNN)
    - [ESM-IF1](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/esm)
    - [DeepHops](https://gitee.com/mindspore/mindscience/pulls/848)
    - [ColabDesign](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/Colabdesign)
- Foundation Model
    - [GROVER](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/grover)
    - [MG-BERT](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/MG_BERT)