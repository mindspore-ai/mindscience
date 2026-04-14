# MindSpore SPONGE Release Notes

[View English](./RELEASE.md)

## MindSPONGE 1.0.0rc2 Release Notes

### RASP & FAAST

- [STABLE] RASP & FAAST是昌平实验室高毅勤团队开发的蛋白质结构解析工具。RASP（Restraints Assisted Structure Predictor）模型能接受抽象或实验约束，使它能根据抽象或实验、稀疏或密集的约束生成结构，可用于多种应用，包括改进多结构域蛋白和msa较少的蛋白的结构预测。FAAST（iterative Folding Assisted peak ASsignmenT）方法通过结合RASP与传统核磁共振数据解析方法，实现了核磁共振数据全自动解析。

### Bug Fixes

- [I8G9N5] 修复SPONGE中的分子模拟样例tutorial_b01.py get_item报错问题。
- [I78EJO] 修复mindsponge.cell.TriangleAttention问题（shape不一致）。
- [I7QZVK] 修复MEGA-Protein支持序列长度与文档不一致的问题。

## 贡献者

感谢以下人员做出的贡献:

yangyi, zhangjun, liusirui, xiayijie, chendiqing, huangyupeng,  yufan, wangzidong, niningxi, chenmengyun, chuhaotian, fengxun, huyingtong, liqingguo, liushuo, luxingyu, pantianyuan, wangmin, xuchen, zhangweijie

欢迎以任何形式对项目提供贡献！

## MindSpore SPONGE 1.0.0-rc1 Release Notes

MindSpore SPONGE(Simulation Package tOwards Next GEneration molecular modelling)是基于昇思MindSpore的计算生物领域套件，支持分子动力学、蛋白质折叠等常用功能，旨在于为广大的科研人员、老师及学生提供高效易用的AI计算生物软件。

### 主要特性和增强

- [STABLE] 蛋白质结构预测工具MEGA-Fold。
- [STABLE] MSA生成工具MEGA-EvoGen：突破在孤儿序列、高异变序列和人造蛋白等MSA匮乏场景下无法做出准确预测的限制。
- [STABLE] 蛋白质结构评分工具MEGA-Assessment：该工具可以评价蛋白质结构每个残基的准确性以及残基-残基之间的距离误差，同时可以基于评价结果对蛋白结构作出进一步的优化。
- [STABLE] MindSPONGE.PipeLine模块：该模块中包含了10+模型及统一调用接口，可直接调用所需模型进行训练和推理任务。

### 贡献者

感谢以下人员做出的贡献:

yufan, gaoyiqin, wangzidong, lujiale, chuht, wangmin0104, mamba_ni, yujialiang, melody, Yesterday, xiayijie, jun.zhang, siruil, Dechin Chen, 十六夜, wangchenghao, liushuo, lijunbin.

欢迎以任何形式对项目提供贡献！

## MindSPONGE 1.0.0-alpha Release Notes

MindSPONGE(Simulation Package tOwards Next GEneration molecular modelling)是基于昇思MindSpore的计算生物领域套件，支持分子动力学、蛋白质折叠等常用功能，旨在于为广大的科研人员、老师及学生提供高效易用的AI计算生物软件。

### 主要特性和增强

- [STABLE] 蛋白质结构预测工具MEGA-Fold。
- [STABLE] MSA生成工具MEGA-EvoGen：突破在孤儿序列、高异变序列和人造蛋白等MSA匮乏场景下无法做出准确预测的限制。
- [STABLE] 蛋白质结构评分工具MEGA-Assessment：该工具可以评价蛋白质结构每个残基的准确性以及残基-残基之间的距离误差，同时可以基于评价结果对蛋白结构作出进一步的优化。

### 贡献者

感谢以下人员做出的贡献:

yufan, gaoyiqin, wangzidong, lujiale, chuht, wangmin0104, mamba_ni, yujialiang, melody, Yesterday, xiayijie, jun.zhang, siruil, Dechin Chen, 十六夜, wangchenghao, liushuo, lijunbin.

欢迎以任何形式对项目提供贡献！
