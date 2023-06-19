# MindScience

[View English](README.md)

- [MindScience概述](#概述)
    - [MindFlow](#mindflow)
    - [MindElec](#mindelec)
    - [MindChemistry](#mindchemistry)
    - [MindSPONGE](#mindsponge)
- [架构图](#架构图)
- [合作伙伴](#合作伙伴)

## 概述

MindScience是基于MindSpore融合架构打造的科学计算行业套件，包含了业界领先的数据集、基础模型、预置高精度模型和前后处理工具，加速了科学行业应用开发。

### [MindFlow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow)

计算流体求解套件MindFlow支持物理驱动、数据驱动和数据机理融合驱动的AI流体仿真；对接国产CFD求解器PHengLei，实现AI和传统流体求解的耦合；内置可微分CFD求解器，实现流场求解的端到端微分。

MindFlow提供了翼型流场、湍流等常用数据集，支持Al流体仿真模型的训练和模型评估。

基于上述功能和数据集，MindFlow已经成功孵化一系列有影响力的成果，包括和中国商飞合作，发布“东方御风”、“东方翼风”大模型，机翼气动仿真由小时级降低到秒级，助力国产大飞机起飞；和西北工业大学合作，发布“泰岭翱翔”大模型，实现AI湍流模型的高精度耦合仿真。

### [MindElec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)

计算电磁仿真领域套件MindElec，支持数据建构及转换、仿真计算、结果可视化以及端到端的AI电磁仿真。在手机电磁仿真已取得技术实破，仿真精度媲美传统科学计算软件，同时性能提升10倍。

MindElec支持CSG模式的几何构建，如矩形、圆形等结构的交集、并集和差集，以及cst和stp数据的高效张量转换。

基于上述功能和数据集，MindElec已经成功孵化一系列有影响力的成果。和华为诺亚合作，实现了端到端可微FDTD，并且在贴片天线、贴片滤波器以及二维电磁逆散射等场景进行了验证。和东南大学合作，发布“金陵·电磁脑”基础模型，阵列天线仿真效率提升10X+倍以上，且随着阵列规模增大，效率提升更加显著。

### [MindChemistry](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry)

计算化学领域套件MindChemistry支持多体系，多尺度任务的AI+化学仿真，致力于高效使能AI与化学的融合研究。

MindChemistry内置等变计算库，显著提高科学场景建模数据的表征效率和模型的训练效率。MindChemistry还提供rMD17等业界高频使用的数据集，支持分子生成、预测模型的训练与评估，提供等变计算、高阶优化器等接口与功能。

基于上述功能和数据集，MindChemistry已经成功孵化一系列有影响力的成果。对接分子生成与分子预测SOTA模型，实现AI在化学领域的高效材料设计与分子性能预测。

### [MindSPONGE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE)

计算生物领域套件MindSPONGE支持高性能、模块化，端到端可微，类AI架构编写的分子模拟功能以及MSA生成，蛋白质折叠训练推理和蛋白质结构打分，NMR数据解析等常用功能。

MindSPONGE提供了高覆盖度和多样性的百万级蛋白质结构预测数据集——PSP，支持蛋白质结构训练和推理。

基于上述功能和数据集，MindSPONGE已经成功孵化一系列有影响力的成果。包括与高毅勤老师团队合作，发布分子模拟软件，支持可微分编程和高通量模拟；发布全流程蛋白质结构预测工具
MEGA-Protein，支持高性能高精度预测蛋白质结构；以及核磁共振波谱法数据自动解析FAAST，实现了NMR数据解析时间从数月到数小时的缩短。

## 架构图

<div align=center>
<img src="docs/MindScience_Architecture.jpg" alt="MindScience Architecture" width="600"/>
</div>

## 合作伙伴

<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
</head>

<body>
    <table width=100% align="center">
        <tr id='tr1'>
            <td>
                <img src="MindFlow/docs/partners/CACC.jpeg" />
            </td>
            <td>
                <img src="MindFlow/docs/partners/NorthwesternPolytechnical.jpeg" />
            </td>
            <td>
                <img src="MindFlow/docs/partners/Peking_University.jpeg" />
            </td>
        </tr>
        <tr id='tr2'>
            <td>
                <img src="MindSPONGE/docs/cooperative_partner/深圳湾.jpg" />
            </td>
            <td>
                <img src="MindSPONGE/docs/cooperative_partner/西电.png" />
            </td>
            <td>
                <img src="MindFlow/docs/partners/TaiHuLaboratory.jpeg" />
            </td>
        </tr>
        <tr id='tr3'>
            <td>
                <img src="MindElec/docs/shanghai_jiaotong_university.jpg" />
            </td>
            <td>
                <img src="MindElec/docs/dongnan_university.jpg" />
            </td>
            <td>
                <img src="MindFlow/docs/partners/RenminUniversity.jpeg" />
            </td>
        </tr>
    </table>
</body>
</html>