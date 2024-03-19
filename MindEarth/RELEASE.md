# MindSpore Earth Release Notes

ENGLISH | [简体中文](RELEASE_CN.md)

MindSpore Earth is a earth science suite developed based on MindSpore. It supports tasks such as multi-temporal-spatial-scale meteorological forecasting and data preprocessing, committed to efficiently enabling AI+meteorological/ocean fusion research.

## MindSpore Earth 0.2.0 Release Notes

### Major Feature and Improvements

#### Short-Term Precipitation Forecast

- [STABLE] [NowcastNet](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/nowcasting/Nowcastnet): A new short-term and imminent precipitation model based on physical constraints is added to support radar extrapolation precipitation forecast in the next 3 hours.

#### Medium-Range Global Predictions

- [STABLE] [GraphCast](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/graphcast): Added support for 0.25° high-resolution data, GraphCast full-scale model training and inference.
- [STABLE] [FuXi](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/fuxi): The weather model Fuxi is added to support the training and inference of medium-term forecast models.
- [STABLE] [Medium Precipitation](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/graphcast): The medium-term precipitation module case is added to support medium-term precipitation forecast. The pre-trained medium-term large model needs to be used as the backbone.
- [RESEARCH] [CTEFNet](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/climate-prediction/ensoforecast): The El Nino prediction model based on CNN and transfer learning is added to extend the valid ENSO prediction period to 19 months.

### Contributors

Thanks to the following developers for their contributions:

hsliu_ustc, hong-ye-zhou, liulei277, kevinli123, Zhou Chuansai, alancheng511, Cui Yinghao, xingzhongfan, cmy_melody

Contributions to the project in any form are welcome!

## MindSpore Earth 0.1.0 Release Notes

### Major Feature and Improvements

#### Nowcasting precipitation predictions

- [STABLE] [Dgmr](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/nowcasting/dgmr): Dgmr (Deep Generative Model of Radar Network) is a deep generative model for the probabilistic nowcasting of precipitation from radar developed by researchers from DeepMind. The main body of the model is a generator, which is trained adversarially with temporal and spatial discriminator losses and additional regularization terms. The model learns contextual representations from the first four frames of the radar sequence, which are used as input to the sampler. The sampler is a recurrent network composed of convolutional gated recurrent units (GRU), which takes context representations and latent vectors sampled from a Gaussian distribution as input to predict 18 future radar fields.

#### Medium-range global predictions

- [STABLE] [FourCastNet](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/fourcastnet): FourCastNet adopts the adaptive Fourier neural operator AFNO, which is an improvement on the Vision Transformer model. It constructs a continuous global convolution of the mixing operation steps and effectively implements it through FFT in the Fourier domain, reducing the spatial mixing complexity to O(NlogN). This model is the first AI prediction model that can compare its prediction accuracy with the high-resolution integrated prediction system (IFS) model of the European Centre for Medium Range Weather Forecasting (ECMWF).
- [STABLE] [ViT-KNO](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/koopman_vit): Koopman Neural Operator is a lightweight, grid independent model designed based on Koopman's global linearization theory and combined with neural operator ideas. This model was developed in collaboration between Huawei ACS lab and Tsinghua University. By embedding complex dynamics into the linear structure to constrain the reconstruction process, this model can capture complex nonlinear behavior while maintaining its lightweight and computational effectiveness. Compared with FNO, KNO has more efficient training performance and better prediction accuracy.
- [STABLE] [GraphCast](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/medium-range/graphcast): GraphCast was proposed by Google DeepMind, which uses GNN to autoregressively generate prediction results in the "encoding-processing-decoding" architecture. The encoder maps the latitude-longitude input grid of meteorological elements at historical times to a multi-scale icosahedral grid representation. The processor performs multiple rounds of message passing on a multi grid representation. The decoder maps the multi grid representation back to the latitude-longitude grid as the prediction for the next time step. In addition, MindEarth has implemented multi-step iterative training to reduce model error accumulation in response to the attenuation of multiple prediction accuracy.

#### Super-resolution reconstruction of global DEM

- [STABLE] [DEM-SRNet](https://gitee.com/mindspore/mindscience/tree/master/MindEarth/applications/dem-super-resolution): DEM-SRNet is a super-resolution model of a digital elevation model. The model is based on 30m resolution NASADEM satellite images, 450m resolution GEBCO_2021 public data of the United Nations Intergovernmental Oceanographic Commission and high-resolution ocean terrain data of some areas, using a combination of deep residual pre-trained neural network and transfer learning technology to generate a global 90m high-resolution DEM. This data set can provide more accurate basic geographical information and plays a vital role in research fields such as global climate change, ocean tidal movements, and material exchange in the geosphere.

### Contributors

Thanks goes to these wonderful people:

hsliu_ustc, hong-ye-zhou, liulei277, kevinli123, alancheng511

Contributions of any kind are welcome!
