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

FourCastNet adopts the adaptive Fourier neural operator AFNO, which is an improvement on the Vision Transformer model. It constructs a continuous global convolution of the mixing operation steps and effectively implements it through FFT in the Fourier domain, reducing the spatial mixing complexity to O(NlogN). This model is the first AI prediction model that can compare its prediction accuracy with the high-resolution integrated prediction system (IFS) model of the European Centre for Medium Range Weather Forecasting (ECMWF).

* ViT-KNO

Koopman Neural Operator is a lightweight, grid independent model designed based on Koopman's global linearization theory and combined with neural operator ideas. This model was developed in collaboration between Huawei ACS lab and Tsinghua University. By embedding complex dynamics into the linear structure to constrain the reconstruction process, this model can capture complex nonlinear behavior while maintaining its lightweight and computational effectiveness. Compared with FNO, KNO has more efficient training performance and better prediction accuracy.

* GraphCast

GraphCast was proposed by Google DeepMind, which uses GNN to autoregressively generate prediction results in the "encoding-processing-decoding" architecture. The encoder maps the latitude-longitude input grid of meteorological elements at historical times to a multi-scale icosahedral grid representation. The processor performs multiple rounds of message passing on a multi grid representation. The decoder maps the multi grid representation back to the latitude-longitude grid as the prediction for the next time step. In addition, MindEarth has implemented multi-step iterative training to reduce model error accumulation in response to the attenuation of multiple prediction accuracy.

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
