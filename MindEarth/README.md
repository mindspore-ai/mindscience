# MindSpore Earth

English | [简体中文](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindEarth/README_CN.md)

- [MindSpore Earth](#mindspore-earth)
  - [MindSpore Earth Introduction](#mindspore-earth-introduction)
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

## MindSpore Earth Introduction

Weather phenomena are closely related to human production and life, socioeconomic conditions, military activities, and more. Accurate weather forecasts can mitigate the impact of severe weather events, avoid economic losses, and create ongoing financial revenue in areas such as energy, agriculture, transportation, and entertainment. Currently, weather forecasts mainly use numerical weather prediction models to solve atmospheric dynamic equations that describe weather evolution by processing observational data collected from meteorological satellites, observation stations, radars, etc., thereby providing weather and climate prediction information. The prediction process of numerical models involves massive calculations that require considerable time and computational resources. Compared to numerical models, data-driven deep learning models can effectively reduce computational costs by several orders of magnitude.

MindSpore Earth is an Earth science toolkit developed based on [MindSpore](https://www.mindspore.cn/). It supports AI meteorological predictions for multiple spatiotemporal scales such as nowcasting, medium-term, and long-term forecasts, as well as disaster weather predictions such as precipitation and typhoons. It aims to provide efficient and easy-to-use AI meteorological prediction software for industrial researchers and engineers, university teachers, and students.

## Latest News

## Application Cases

### Ocean

|                             Case                             |                      Description                       |       Dataset       | Model Architecture | NPU  |
| :----------------------------------------------------------: | :----------------------------------------------------: | :-----------------: | :----------------: | :--: |
| [LeadFormer](https://atomgit.com/mindspore-lab/mindscience/tree/master/MindEarth/applications/sea/LeadFormer) | High-resolution intelligent Arctic sea ice forecasting | Not yet open source |    Transformer     |  ✔️   |

### DEM

|                             Case                             |                         Description                          | Dataset | Model Architecture | NPU  |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-----: | :----------------: | :--: |
| [DEM-SRNet](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/dem-super-resolution) | Global 3-arc-second (90m) high-resolution land-sea digital elevation model | nasadem |        EDSR        |  ✔️   |

### El Niño

|                             Case                             |                       Description                        |   Dataset   | Model Architecture | NPU  |
| :----------------------------------------------------------: | :------------------------------------------------------: | :---------: | :----------------: | :--: |
| [CTEFNet](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/climate-prediction/ensoforecast) | CNN and Transfer Learning-based El Niño prediction model | CMIP5, SODA |        CNN         |  ✔️   |

### Nowcasting Precipitation

|                             Case                             |                         Description                          |     Dataset      | Model Architecture  | NPU  |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------: | :-----------------: | :--: |
| [DGMs](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/nowcasting/dgmr) | Radar data meteorological nowcasting based on deep generative models |    Radar data    |    GAN, ConvGRU     |  ✔️   |
| [NowcastNet](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/nowcasting/Nowcastnet) | Generative nowcasting precipitation model incorporating physical mechanisms | USA-MRMS dataset | GAN, two-path U-Net |  ✔️   |
| [PreDiff](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/nowcasting/PreDiff) | Short-term precipitation forecasting based on latent diffusion models | SEVIR_LR dataset |  LDM, Earthformer   |  ✔️   |

### Medium-range Weather Forecast

|                             Case                             |                         Description                          |         Dataset         |    Model Architecture    | NPU  |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------: | :----------------------: | :--: |
| [FourCastNet](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/medium-range/fourcastnet) |         Data-driven global weather prediction model          | ERA5 reanalysis dataset |           AFNO           |  ✔️   |
| [ViT-KNO](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/medium-range/koopman_vit) | Learning Koopman Operator for predicting nonlinear system dynamics | ERA5 reanalysis dataset |           ViT            |  ✔️   |
| [GraphCast](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/medium-range/graphcast) | Global medium-range weather forecast based on graph neural networks | ERA5 reanalysis dataset |           GNN            |  ✔️   |
| [FuXi](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/medium-range/fuxi) | Global medium-range weather forecast based on cascaded architecture | ERA5 reanalysis dataset | CNN, Swin Transformer V2 |  ✔️   |
| [SKNO](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/medium-range/skno) |          Integration of KNO model and SHT operator           | ERA5 reanalysis dataset |           SKNO           |  ✔️   |

### Earthquake Early Warning

|                             Case                             |                         Description                          |      Dataset       | Model Architecture | NPU  |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------: | :----------------: | :--: |
| [G-TEAM](https://atomgit.com/mindspore-lab/mindscience/tree/legacy-master/MindEarth/applications/earthquake/G-TEAM) | Nationwide earthquake early warning system based on data-driven approach | Diting 2.0 dataset |  CNN, Transformer  |  ✔️   |

## Core Contributors

Thanks to the following developers for their contributions to MindSpore Earth:

yufan, wangzidong, liuhongsheng, zhouhongye, liulei, libokai, chengqiang, dongyonghan, zhouchuansai, liuruoyan, funfunplus

## Contribution Guide

Welcome to contribute your code to MindSpore Earth by referring to the [Contribution Guide](https://atomgit.com/mindspore-lab/mindscience/blob/master/CONTRIBUTION.md)!

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)