# Mindspore Earth

English | [简体中文](README_CN.md)

- [Mindspore Earth](#mindspore-earth)
    - [Mindspore Earth Introduction](#mindspore-earth-introduction)
    - [Latest News](#latest-news)
    - [Application Cases](#application-cases)
        - [Ocean](#ocean)
        - [DEM](#dem)
        - [El Niño](#el-niño)
        - [Nowcasting Precipitation](#nowcasting-precipitation)
        - [Medium-range Weather Forecast](#medium-range-weather-forecast)
        - [Earthquake Early Warning](#earthquake-early-warning)
    - [Core Contributors](#core-contributors)
    - [Contribution Guide](#contribution-guide)
    - [License](#license)

## Mindspore Earth Introduction

Weather phenomena are closely related to human production and life, socioeconomic conditions, military activities, and more. Accurate weather forecasts can mitigate the impact of severe weather events, avoid economic losses, and create ongoing financial revenue in areas such as energy, agriculture, transportation, and entertainment. Currently, weather forecasts mainly use numerical weather prediction models to solve atmospheric dynamic equations that describe weather evolution by processing observational data collected from meteorological satellites, observation stations, radars, etc., thereby providing weather and climate prediction information. The prediction process of numerical models involves massive calculations that require considerable time and computational resources. Compared to numerical models, data-driven deep learning models can effectively reduce computational costs by several orders of magnitude.

Mindspore Earth is an Earth science toolkit developed based on [MindSpore](https://www.mindspore.cn/). It supports AI meteorological predictions for multiple spatiotemporal scales such as nowcasting, medium-term, and long-term forecasts, as well as disaster weather predictions such as precipitation and typhoons. It aims to provide efficient and easy-to-use AI meteorological prediction software for industrial researchers and engineers, university teachers, and students.

## Latest News

## Application Cases

### Ocean

|                   Case                    |                 Description                 | Dataset | Model Architecture | NPU |
| :---------------------------------------: | :---------------------------------------: | :-----: | :----------------: | :---: |
| [LeadFormer](applications/sea/LeadFormer) | High-resolution intelligent Arctic sea ice forecasting | Not yet open source | Transformer | ✔️ |

### DEM

|                   Case                   |                       Description                       | Dataset | Model Architecture | NPU |
| :--------------------------------------: | :-----------------------------------------------------: | :-----: | :----------------: | :---: |
| [DEM-SRNet][dem-super-resolution-URL] | Global 3-arc-second (90m) high-resolution land-sea digital elevation model | nasadem | EDSR | ✔️ |

### El Niño

|                Case                 |                    Description                    |    Dataset     | Model Architecture | NPU |
| :---------------------------------: | :---------------------------------------------: | :------------: | :----------------: | :---: |
| [CTEFNet][ensoforecast-URL] | CNN and Transfer Learning-based El Niño prediction model | CMIP5, SODA | CNN | ✔️ |

### Nowcasting Precipitation

|              Case              |                    Description                     |      Dataset       |        Model Architecture        | NPU |
| :----------------------------: | :-----------------------------------------------: | :----------------: | :------------------------------: | :---: |
|       [DGMs][dgmr-URL]       | Radar data meteorological nowcasting based on deep generative models | Radar data | GAN, ConvGRU | ✔️ |
| [NowcastNet][Nowcastnet-URL] | Generative nowcasting precipitation model incorporating physical mechanisms | USA-MRMS dataset | GAN, two-path U-Net | ✔️ |
|    [PreDiff][PreDiff-URL]    | Short-term precipitation forecasting based on latent diffusion models | SEVIR_LR dataset | LDM, Earthformer | ✔️ |

### Medium-range Weather Forecast

|               Case               |                     Description                      |       Dataset        |         Model Architecture         | NPU |
| :------------------------------: | :--------------------------------------------------: | :------------------: | :--------------------------------: | :---: |
| [FourCastNet][fourcastnet-URL] | Data-driven global weather prediction model | ERA5 reanalysis dataset | AFNO | ✔️ |
|   [ViT-KNO][koopman_vit-URL]   | Learning Koopman Operator for predicting nonlinear system dynamics | ERA5 reanalysis dataset | ViT | ✔️ |
|   [GraphCast][graphcast-URL]   | Global medium-range weather forecast based on graph neural networks | ERA5 reanalysis dataset | GNN | ✔️ |
|        [FuXi][fuxi-URL]        | Global medium-range weather forecast based on cascaded architecture | ERA5 reanalysis dataset | CNN, Swin Transformer V2 | ✔️ |
|        [SKNO][skno-URL]        | Integration of KNO model and SHT operator | ERA5 reanalysis dataset | SKNO | ✔️ |

### Earthquake Early Warning

|          Case          |                Description                |       Dataset        |     Model Architecture     | NPU |
| :--------------------: | :---------------------------------------: | :------------------: | :------------------------: | :---: |
| [G-TEAM][G-TEAM-URL] | Nationwide earthquake early warning system based on data-driven approach | Diting 2.0 dataset | CNN, Transformer | ✔️ |

## Core Contributors

Thanks to the following developers for their contributions to Mindspore Earth:

yufan, wangzidong, liuhongsheng, zhouhongye, liulei, libokai, chengqiang, dongyonghan, zhouchuansai, liuruoyan, funfunplus

## Contribution Guide

Welcome to contribute your code to Mindspore Earth by referring to the [Contribution Guide](../CONTRIBUTION.md)!

## License

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