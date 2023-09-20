# MindEarth 0.1.0 Release Notes

Initial release of MindEarth.

## Major Features

### Super-resolution reconstruction of global DEM

* DemNet

DemNet, short for digital elevation model network, can provide accurate basic geographical data and therefore plays a
vital role in research fields such as global climate change, ocean tidal movement, and material exchange on the Earth's
sphere. Global ocean DEM is a cutting-edge branch of marine geology and ocean mapping, providing direct reference for
understanding seafloor tectonic movements and seafloor establishment processes.

### Medium-range global predictions

* FourCastNet

FourCastNet, short for Fourier Forecasting Neural Network, is a global data-driven weather forecasting model that
provides accurate short to medium-range global predictions at 0.25∘ resolution. FourCastNet accurately forecasts
high-resolution, fast-timescale variables such as the surface wind speed, precipitation, and atmospheric water vapor. It
has important implications for planning wind energy resources, predicting extreme weather events such as tropical
cyclones, extra-tropical cyclones, and atmospheric rivers.

* ViT-KNO

Koopman neural operator (KNO) is a representative demonstration and outperforms other state-of-the art alternatives in
terms of accuracy and efficiency.
Beyond the original version of KNO, there are
multiple new variants of KNO based on different neural network architectures to improve the general applicability of our
module, such as ViT-KNO. These variants are validated by mesh-independent and long-term prediction experiments
implemented on representative
PDEs.

* GraphCast

GraphCast is a new ML-based weather simulator surpasses the world's most accurate deterministic operational medium-range
weather forecast system and all ML baselines. The GraphCast autoregressive model was trained using meteorological data
from the ERA5 reanalysis archive of the European Centre for Medium Range Weather Forecasts (ECMWF). The model is based
on a neural network and a novel high-resolution multi-scale grid representation. It has a resolution of about 25 × 25 km
at the equator and can create a six-day forecast every 10 hours for five surfaces and six atmospheric variables, each at
37 vertical pressure levels.

### Nowcasting precipitation predictions

* Dgmr

DgmrNet (Deep Generative Model of Radar Network) is a deep generative model for the probabilistic nowcasting of
precipitation from radar developed by researchers from DeepMind. It produces realistic and spatiotemporally consistent
predictions over regions up to 1,536 km x 1,280 km and with lead times from 5-90 min ahead. Using a systematic
evaluation by more than 50 expert meteorologists, this method show that DgmrNet ranked first for its accuracy and
usefulness in 89% of cases against two competitive methods. It can provide probalilistic predictions that improve
forecast value and support operational utility, and at resolutions and lead times where alternative methods struggle.

## Contributors

Thanks goes to these wonderful people:

yufan, wangzidong, liuhongsheng, zhouhongye, liulei, libokai, chengqiang, dongyonghan, zhouchuansai

Contributions of any kind are welcome!
