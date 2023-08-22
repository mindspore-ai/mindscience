# MindEarth 0.1.0 Release Notes

Initial release of MindEarth.

## Major Features

### Medium-range global predictions

* FourCastNet

FourCastNet, short for Fourier Forecasting Neural Network, is a global data-driven weather forecasting model that
provides accurate short to medium-range global predictions at 0.25∘ resolution. FourCastNet accurately forecasts
high-resolution, fast-timescale variables such as the surface wind speed, precipitation, and atmospheric water vapor. It
has important implications for planning wind energy resources, predicting extreme weather events such as tropical
cyclones, extra-tropical cyclones, and atmospheric rivers.

* ViT-KNO

Koopman neural operator (KNO) is a representative demonstration and outperforms other state-of-theart alternatives in
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

## Contributors

Thanks goes to these wonderful people:

wangzidong, liuhongsheng, zhouhongye, liboka, liulei

Contributions of any kind are welcome!
