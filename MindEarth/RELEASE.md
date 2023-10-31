ENGLISH | [简体中文](RELEASE_CN.md)

# MindEarth 0.1.0 Release Notes

## Major Features

### Medium-range global predictions

* FourCastNet

    FourCastNet adopts the adaptive Fourier neural operator AFNO, which is an improvement on the Vision Transformer model. It constructs a continuous global convolution of the mixing operation steps and effectively implements it through FFT in the Fourier domain, reducing the spatial mixing complexity to O(NlogN). This model is the first AI prediction model that can compare its prediction accuracy with the high-resolution integrated prediction system (IFS) model of the European Centre for Medium Range Weather Forecasting (ECMWF).

* ViT-KNO

    Koopman Neural Operator is a lightweight, grid independent model designed based on Koopman's global linearization theory and combined with neural operator ideas. This model was developed in collaboration between Huawei ACS lab and Tsinghua University. By embedding complex dynamics into the linear structure to constrain the reconstruction process, this model can capture complex nonlinear behavior while maintaining its lightweight and computational effectiveness. Compared with FNO, KNO has more efficient training performance and better prediction accuracy.

* GraphCast

    GraphCast was proposed by Google DeepMind, which uses GNN to autoregressively generate prediction results in the "encoding-processing-decoding" architecture. The encoder maps the latitude-longitude input grid of meteorological elements at historical times to a multi-scale icosahedral grid representation. The processor performs multiple rounds of message passing on a multi grid representation. The decoder maps the multi grid representation back to the latitude-longitude grid as the prediction for the next time step. In addition, MindEarth has implemented multi-step iterative training to reduce model error accumulation in response to the attenuation of multiple prediction accuracy.

### Nowcasting precipitation predictions

* Dgmr

    Dgmr (Deep Generative Model of Radar Network) is a deep generative model for the probabilistic nowcasting of precipitation from radar developed by researchers from DeepMind. The main body of the model is a generator, which is trained adversarially with temporal and spatial discriminator losses and additional regularization terms. The model learns contextual representations from the first four frames of the radar sequence, which are used as input to the sampler. The sampler is a recurrent network composed of convolutional gated recurrent units (GRU), which takes context representations and latent vectors sampled from a Gaussian distribution as input to predict 18 future radar fields.

### Super-resolution reconstruction of global DEM

* DEM-SRNet

    DEM-SRNet is a super-resolution model of a digital elevation model. The model is based on 30m resolution NASADEM satellite images, 450m resolution GEBCO_2021 public data of the United Nations Intergovernmental Oceanographic Commission and high-resolution ocean terrain data of some areas, using a combination of deep residual pre-trained neural network and transfer learning technology to generate a global 90m high-resolution DEM. This data set can provide more accurate basic geographical information and plays a vital role in research fields such as global climate change, ocean tidal movements, and material exchange in the geosphere.

## Contributors

Thanks goes to these wonderful people:

hsliu_ustc, hong-ye-zhou, liulei277, kevinli123, alancheng511

Contributions of any kind are welcome!
