<!-- markdownlint-disable first-line-h1 -->

简体中文 | [English](README.md)

# Mindspore Earth

- [Mindspore Earth](#mindspore-earth)
    - [Mindspore Earth 介绍](#mindspore-earth-介绍)
    - [最新消息](#最新消息)
    - [应用案例](#应用案例)
        - [海洋](#海洋)
        - [DEM](#dem)
        - [厄尔尼诺](#厄尔尼诺)
        - [短临降水](#短临降水)
        - [中期气象预报](#中期气象预报)
        - [地震预警](#地震预警)
    - [核心贡献者](#核心贡献者)
    - [贡献指南](#贡献指南)
    - [许可证](#许可证)

## Mindspore Earth 介绍

天气现象与人类的生产生活、社会经济、军事活动等方方面面都密切相关，准确的天气预报能够在灾害天气事件中减轻影响、避免经济损失，还能创造持续不断地财政收入，例如能源、农业、交通和娱乐行业。目前，天气预报主要采用数值天气预报模式，通过处理由气象卫星、观测台站、雷达等收集到的观测资料，求解描写天气演变的大气动力学方程组，进而提供天气气候的预测信息。数值预报模式的预测过程涉及大量计算，耗费较长时间与较大的计算资源。相较于数值预报模式，数据驱动的深度学习模型能够有效地将计算成本降低数个量级。

Mindspore Earth 是基于[昇思 MindSpore](https://www.mindspore.cn/) 开发的地球科学领域套件，支持短临、中期、长期等多时空尺度以及降水、台风等灾害性天气的AI气象预测，旨在于为广大的工业界科研工程人员、高校老师及学生提供高效易用的AI气象预测软件。

## 最新消息

## 应用案例

### 海洋

|                   案例                    |           简介           |  数据集  |  模型架构   |  NPU  |
| :---------------------------------------: | :----------------------: | :------: | :---------: | :---: |
| [LeadFormer](applications/sea/LeadFormer) | 北极海冰高分辨率智能预报 | 暂不开源 | Transformer |   ✔️   |

### DEM

|                 案例                  |                   简介                   | 数据集  | 模型架构 |  NPU  |
| :-----------------------------------: | :--------------------------------------: | :-----: | :------: | :---: |
| [DEM-SRNet][dem-super-resolution-URL] | 全球3弧秒（90m）海陆高分辨率数字高程模型 | nasadem |   EDSR   |   ✔️   |

### 厄尔尼诺

|             案例              |                 简介                  |   数据集    | 模型架构 |  NPU  |
| :---------------------------: | :-----------------------------------: | :---------: | :------: | :---: |
| [CTEFNet][ensoforecast-URL] | 基于 CNN 与迁移学习的厄尔尼诺预测模型 | CMIP5、SODA |   CNN    |   ✔️   |

### 短临降水

|             案例             |                  简介                  |     数据集     |      模型架构       |  NPU  |
| :--------------------------: | :------------------------------------: | :------------: | :-----------------: | :---: |
|       [DGMs][dgmr-URL]       | 基于深度生成模型的雷达数据气象短临预报 |    雷达数据    |    GAN、ConvGRU     |   ✔️   |
| [NowcastNet][Nowcastnet-URL] |  融入物理机制的生成式短临降水预报模型  | USA-MRMS数据集 | GAN、two-path U-Net |   ✔️   |
|    [PreDiff][PreDiff-URL]    |     基于潜在扩散模型的降水短时预报     | SEVIR_LR数据集 |  LDM、 Earthformer  |   ✔️   |

### 中期气象预报

|              案例              |                      简介                      |      数据集      |         模型架构         |  NPU  |
| :----------------------------: | :--------------------------------------------: | :--------------: | :----------------------: | :---: |
| [FourCastNet][fourcastnet-URL] |           数据驱动的全球气象预测模型           | ERA5再分析数据集 |           AFNO           |   ✔️   |
|   [ViT-KNO][koopman_vit-URL]   | 学习Koopman Operator算子预测非线性系统的动力学 | ERA5再分析数据集 |           ViT            |   ✔️   |
|   [GraphCast][graphcast-URL]   |        基于图神经网络的全球中期天气预报        | ERA5再分析数据集 |           GNN            |   ✔️   |
|        [FuXi][fuxi-URL]        |         基于级联架构的全球中期天气预报         | ERA5再分析数据集 | CNN、Swin Transformer V2 |   ✔️   |
|        [SKNO][skno-URL]        |             融合了KNO模型和SHT算子             | ERA5再分析数据集 |           SKNO           |   ✔️   |

### 地震预警

|         案例         |            简介            |      数据集      |     模型架构     |  NPU  |
| :------------------: | :------------------------: | :--------------: | :--------------: | :---: |
| [G-TEAM][G-TEAM-URL] | 数据驱动的全国地震预警系统 | diting 2.0数据集 | CNN、Transformer |   ✔️   |

## 核心贡献者

感谢以下开发者对 Mindspore Earth 的贡献：

yufan, wangzidong, liuhongsheng, zhouhongye, liulei, libokai, chengqiang, dongyonghan, zhouchuansai, liuruoyan, funfunplus

## 贡献指南

欢迎参考[贡献指南](../CONTRIBUTION.md)为 Mindspore Earth 贡献您的代码！

## 许可证

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

[ensoforecast-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/climate-prediction/ensoforecast
[dem-super-resolution-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/dem-super-resolution
[dgmr-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/nowcasting/dgmr
[Nowcastnet-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/nowcasting/Nowcastnet
[PreDiff-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/nowcasting/PreDiff
[koopman_vit-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/medium-range/koopman_vit
[graphcast-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/medium-range/graphcast
[fuxi-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/medium-range/fuxi
[fourcastnet-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/medium-range/fourcastnet
[skno-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/medium-range/skno
[G-TEAM-URL]: https://gitee.com/mindspore/mindscience/tree/legacy-master/MindEarth/applications/earthquake/G-TEAM
