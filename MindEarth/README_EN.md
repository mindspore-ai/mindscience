 ENGLISH | [简体中文](README_CN.md)

[![master](https://img.shields.io/badge/version-master-blue.svg?style=flat?logo=Gitee)](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/README.md)
[![docs](https://img.shields.io/badge/docs-master-yellow.svg?style=flat)](https://mindspore.cn/mindflow/docs/en/master/index.html)
[![internship](https://img.shields.io/badge/internship-tasks-important.svg?style=flat)](https://gitee.com/mindspore/community/issues/I55B5A?from=project-issue)
[![SIG](https://img.shields.io/badge/community-SIG-yellowgreen.svg?style=flat)](https://mindspore.cn/community/SIG/detail/en?name=mindflow%20SIG)
[![Downloads](https://static.pepy.tech/badge/mindflow-gpu)](https://pepy.tech/project/mindflow-gpu)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://gitee.com/mindspore/mindscience/pulls)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)

# **MindEarth

- [Introduction](#Introduction)
- [Latest News](#Latest-News)
- [Applications](#Applications)
    - [Precipitation Nowcasting](#Precipitation Nowcasting)
    - [Medium-range Forecast](#Medium-range Forecast)
    - [Data Preprocessing](#Data Preprocessing)
- [Installation](#Installation)
    - [Version Dependency](#Version-Dependency)
    - [Install Dependency](#Install-Dependency)
    - [Hardware](#Hardware)
    - [pip install](#pip-install)
    - [source code install](#source-code-install)
- [Community](#Community)
    - [SIG](#Join-MindFlow-SIG)
    - [Core Contributor](#Core-Contributor)
    - [Community Partners](#Community-Partners)
- [Contribution Guide](#Contribution-Guide)
- [License](#License)

## **Introduction**

Weather phenomena are closely related to human production and life, social economy, military activities and other aspects. Accurate weather forecasts can mitigate the impact of disaster weather events, avoid economic losses, and generate continuous fiscal revenue, such as energy, agriculture, transportation and entertainment industries. At present, the weather forecast mainly adopts numerical weather prediction models, which processes the observation data collected by meteorological satellites, observation stations and radars, solves the atmospheric dynamic equations describing weather evolution, and then provides weather and climate prediction information. The prediction process of numerical prediction model involves a lot of computation, which consumes a long time and a large amount of computation resources. Compared with the numerical prediction model, the data-driven deep learning model can effectively reduce the computational cost by several orders of magnitude.

MindEarth is a meteorological and oceanographic suite developed based on [MindSpore](https://www.mindspore.cn/). It supports AI meteorological prediction of short-term, medium-term, and long-term weather and catastrophic weather such as precipitation and typhoon. The aim is to provide efficient and easy-to-use AI meteorological prediction software for industrial scientific research engineers, college teachers and students.

<div align=center><img src="docs/mindearth_archi_en.png" alt="MindFlow Architecture" width="700"/></div>

## **Latest News**

- 🔥`2023.02.06` MindSpore Helps Ocean Terrain Overscore: Huang Xiaomeng Team of Tsinghua University Releases Global 3 Arc Second (90 m) Sea and Land DEM Data Products.[Page](https://blog.csdn.net/Kenji_Shinji/article/details/128906754).

## Applications

### Precipitation Nowcasting

|        Case            |        Dataset               |    Network       |  GPU    |  NPU  |
|:----------------------:|:--------------------------:|:---------------:|:-------:|:------:|
|DGMs       |             Radar dataset             | GAN、ConvGRU |   ✔️     |   ✔️   |

### Medium-range Forecast

|        Case            |              Dataset                  |    Network       |  GPU    |  NPU  |
|:----------------------:|:-------------------------------------:|:---------------:|:-------:|:------:|
|FourCastNet        |       ERA5 Reanalysis Dataset       |      AFNO      |   ✔️     |   ✔️   |
|ViT-KNO       | ERA5 Reanalysis Dataset     |       ViT       |   ✔️     |   ✔️   |
|GraphCast        |      ERA5 Reanalysis Dataset      |       GNN       |   ✔️     |   ✔️   |

### Data Preprocessing

|          Case              |        Dataset               |    Network       |  GPU    |  NPU  |
|:--------------------------:|:--------------------------:|:---------------:|:-------:|:------:|
|   DEM Super-resolution   | NASADEM、GEBCO_2021 |    SRGAN    |   ✔️     |   ✔️   |

## **Installation**

### Version Dependency

Because MindFlow is dependent on MindSpore, please click [MindSpore Download Page](https://www.mindspore.cn/versions) according to the corresponding relationship indicated in the following table. Download and install the corresponding whl package.

| MindFlow |                                  Branch                                |  MindSpore  |Python |
|:--------:|:----------------------------------------------------------------------:|:-----------:|:-------:|
|  master  | [master](https://gitee.com/mindspore/mindscience/tree/master/MindFlow) |        \       | \>=3.7 |
| 0.1.0rc1 | [r0.2.0](https://gitee.com/mindspore/mindscience/tree/r0.2.0/MindFlow) |   \>=2.0.0rc1  | \>=3.7 |

### Install Dependency

```bash
pip install -r requirements.txt
```

### Hardware

| Hardware      | OS              | Status |
|:--------------| :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️ |
|               | Ubuntu-aarch64  | ✔️ |
|               | EulerOS-aarch64 | ✔️ |
|               | CentOS-x86      | ✔️ |
|               | CentOS-aarch64  | ✔️ |
| GPU CUDA 11.1 | Ubuntu-x86      | ✔️ |

### **pip install**

```bash

# GPU version
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindScience/gpu/x86_64/cuda-11.1/mindflow_gpu-0.1.0rc1-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Ascend version
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindScience/ascend/aarch64/mindflow_ascend-0.1.0rc1-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### **source code install**

```bash
git clone https://gitee.com/mindspore/mindscience.git
cd {PATH}/mindscience/MindFlow
```

- Ascend backend

```bash
bash build.sh -e ascend -j8
```

- GPU backend

```bash
export CUDA_PATH={your_cuda_path}
bash build.sh -e gpu -j8
```

- Install whl package

```bash
cd {PATH}/mindscience/MindFLow/output
pip install mindflow_*.whl
```

## **Community**

### Join MindFlow SIG

<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html" charset="utf-8">
</head>
<body>

<table id="t2" style="text-align:center" align="center">
    <tr id="tr2">
        <td align="center">
            <img src="docs/co-chairs/张伟伟.jpeg" width="200" height="243"/>
            <p align="center">
                Northwestern Polytechnical University ZhangWeiwei
            </p>
        </td>
        <td align="center">
            <img src="docs/co-chairs/董彬.jpeg" width="200" height="243"/>
            <p align="center">
                Peking University DongBin
            </p>
        </td>
        <td align="center">
            <img src="docs/co-chairs/孙浩.jpeg" width="200" height="243"/>
            <p align="center">
                RenMin University of China SunHao
            </p>
        </td>
    </tr>
</table>
</body>
</html>

[Join](https://mp.weixin.qq.com/s/e00lvKx30TsqjRhYa8nlhQ) MindSpore [MindFlow SIG](https://mindspore.cn/community/SIG/detail/?name=mindflow%20SIG) to help AI fluid simulation development.
MindSpore AI for Science, [Learning and Learning to solve PDEs](https://www.bilibili.com/video/BV1ur4y1H7vB?p=4) topic report by Dong Bin, Peking University.
We will continue to release [open source internship tasks](https://gitee.com/mindspore/community/issues/I55B5A?from=project-issue), build MindFlow ecology with you, and promote the development of computational fluid dynamics with experts, professors and students in the field. Welcome to actively claim the task.

### Core Contributor

Thanks goes to these wonderful people 🧑‍🤝‍🧑:

yufan, wangzidong, liuhongsheng, zhouhongye, zhangyi, dengzhiwen, liulei, libokai, yangge, longzichao, qiuyisheng, haojiwei, leiyixiang

### Community Partners

<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
</head>
<body>
<table id="t1" style="text-align:center" align="center">
    <tr id="tr1">
        <td>
            <img src="docs/partners/CACC.jpeg"/>
            <p align="center">
                Commercial Aircraft Corporation of China Ltd
            </p>
        </td>
        <td>
            <img src="docs/partners/NorthwesternPolytechnical.jpeg"/>
            <p align="center">
                Northwestern Polytechnical University
            </p>
        </td>
        <td>
            <img src="docs/partners/Peking_University.jpeg"/>
            <p align="center">
                Peking University
            </p>
        </td>
        <td>
            <img src="docs/partners/RenminUniversity.jpeg"/>
            <p align="center">
                Renmin University of China
            </p>
        </td>
    </tr>
</table>
</body>
</html>

## **Contribution Guide**

- Please click here to see how to contribute your code:[Contribution Guide](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)

## **License**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)