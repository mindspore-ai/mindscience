# MindEarth 0.1.0 Release Notes

Initial release of MindEarth.

## Major Features

### Medium-range global predictions

* FourCastNet

FourCastNet adopts the adaptive Fourier neural operator AFNO, which is an improvement on the Vision Transformer model. It constructs a continuous global convolution of the mixing operation steps and effectively implements it through FFT in the Fourier domain, reducing the spatial mixing complexity to O(NlogN). This model is the first AI prediction model that can compare its prediction accuracy with the high-resolution integrated prediction system (IFS) model of the European Centre for Medium Range Weather Forecasting (ECMWF).

* ViT-KNO

Koopman Neural Operator is a lightweight, grid independent model designed based on Koopman's global linearization theory and combined with neural operator ideas. This model was developed in collaboration between Huawei ACS lab and Tsinghua University. By embedding complex dynamics into the linear structure to constrain the reconstruction process, this model can capture complex nonlinear behavior while maintaining its lightweight and computational effectiveness. Compared with FNO, KNO has more efficient training performance and better prediction accuracy.

* GraphCast

GraphCast was proposed by Google DeepMind, which uses GNN to autoregressively generate prediction results in the "encoding-processing-decoding" architecture. The encoder maps the latitude-longitude input grid of meteorological elements at historical times to a multi-scale icosahedral grid representation. The processor performs multiple rounds of message passing on a multi grid representation. The decoder maps the multi grid representation back to the latitude-longitude grid as the prediction for the next time step. In addition, MindEarth has implemented multi-step iterative training to reduce model error accumulation in response to the attenuation of multiple prediction accuracy.

## Contributors

Thanks goes to these wonderful people:

yufan, wangzidong, liuhongsheng, zhouhongye, liulei, libokai, chengqiang, dongyonghan, zhouchuansai

Contributions of any kind are welcome!
