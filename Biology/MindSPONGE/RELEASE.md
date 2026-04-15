# MindSpore SPONGE Release Notes

[查看中文](./RELEASE_CN.md)

## MindSPONGE 1.0.0rc2 Release Notes

### RASP & FAAST

- [STABLE] RASP & FAAST is a protein structure analysis tool developed by Gao Yiqin's team at Changping Laboratory. The RASP (Restraints Assisted Structure Predictor) model accepts abstract or experimental constraints, enabling it to generate structures based on abstract or experimental, sparse or dense constraints, and can be used for a variety of applications, including improved structure prediction for multi-domain proteins and proteins with fewer msa. The iterative Folding Assisted peak ASsignmenT (FAAST) method realizes the automatic analysis of NMR data by combining RASP and traditional NMR data analysis methods.

### Bug Fixes

- [I8G9N5] Fixes the issue where the molecular simulation sample tutorial_b01.py::get_item running failure in SPONGE.
- [I78EJO] Fixed mindsponge.cell.TriangleAttention issue (shape is inconsistent).
- [I7QZVK] Fixes the issue that the sequence length supported by MEGA-Protein is inconsistent with that in the documentation.

## Contributors

Thanks to the contributions of the following people:

yangyi, zhangjun, liusirui, xiayijie, chendiqing, huangyupeng,  yufan, wangzidong, niningxi, chenmengyun, chuhaotian,  fengxun, huyingtong, liqingguo, liushuo, luxingyu, pantianyuan, wangmin, xuchen, zhangweijie

You are welcome to contribute to the project in any form!

## MindSpore SPONGE 1.0.0-rc1 Release Notes

MindSpore SPONGE(Simulation Package tOwards Next GEneration molecular modelling) is a toolkit for Computational Biology based on AI framework MindSpore，which supports MD, folding and so on. It aims to provide efficient AI computational biology software for a wide range of scientific researchers, staff, teachers and students.

### Major Features and Improvements

- [STABLE] Protein Structure Prediction Tool MEGA-Fold.
- [STABLE] MSA Generation Tool MEGA-EvoGen：Multiple Sequence Alignment Dataset for protein structure and function research.
- [STABLE] Protein Structure Assessment Tool MEGA-Assessment：This tool evaluates the prediction accuracy of each residue in the protein structure and the inter-residue distance error. It further optimizes the protein structure based on the evaluation results.
- [STABLE] MindSPONGE.PipeLine module: This module contains over than 10 models and a unified invocation interface. User can invocate the model needed to perform training task and prediction task.

### Contributors

Thanks goes to these wonderful people:

yufan, gaoyiqin, wangzidong, lujiale, chuht, wangmin0104, mamba_ni, yujialiang, melody, Yesterday, xiayijie, jun.zhang, siruil, Dechin Chen, 十六夜, wangchenghao, liushuo, lijunbin.

Contributions of any kind are welcome!

## MindSPONGE 1.0.0-alpha Release Notes

MindSPONGE(Simulation Package tOwards Next GEneration molecular modelling) is a toolkit for Computational Biology based on AI framework MindSpore，which supports MD, folding and so on. It aims to provide efficient AI computational biology software for a wide range of scientific researchers, staff, teachers and students.

### Major Features and Improvements

- [STABLE] Protein Structure Prediction Tool MEGA-Fold.
- [STABLE] MSA Generation Tool MEGA-EvoGen：Multiple Sequence Alignment Dataset for protein structure and function research.
- [STABLE] Protein Structure Assessment Tool MEGA-Assessment：This tool evaluates the prediction accuracy of each residue in the protein structure and the inter-residue distance error. It further optimizes the protein structure based on the evaluation results.

### Contributors

Thanks goes to these wonderful people:

yufan, gaoyiqin, wangzidong, lujiale, chuht, wangmin0104, mamba_ni, yujialiang, melody, Yesterday, xiayijie, jun.zhang, siruil, Dechin Chen, 十六夜, wangchenghao, liushuo, lijunbin.

Contributions of any kind are welcome!
