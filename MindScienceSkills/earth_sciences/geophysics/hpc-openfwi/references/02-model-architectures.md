# OpenFWI Model Architectures

## InversionNet

### Architecture
Encoder-Decoder network for direct velocity inversion.

```
Input: Seismic data (n_shots, n_time, n_receivers)
       |
Encoder: Conv layers with decreasing spatial dimensions
       |
Bottleneck: Compressed representation
       |
Decoder: Transposed conv layers with increasing dimensions
       |
Output: Velocity model (n_z, n_x)
```

### Key Features
- 17 convolutional layers
- Batch normalization
- LeakyReLU activation
- Skip connections (optional)

### Implementation
```python
class InversionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # ... more layers
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        # ... more layers
        
        self.final = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Encoder
        x = F.leaky_relu(self.enc1(x))
        # ...
        
        # Decoder
        x = F.leaky_relu(self.dec1(x))
        # ...
        
        return self.final(x)
```

## VelocityGAN

### Architecture
Generative Adversarial Network for velocity inversion.

```
Generator: InversionNet-like architecture
Discriminator: PatchGAN discriminator
```

### Loss Function
```
L_G = L_adv + lambda * L_L1
L_D = BCE(real) + BCE(fake)
```

### Training Strategy
1. Train discriminator
2. Train generator with adversarial + L1 loss
3. Balance G/D training ratio

## FWI-Net

### Architecture
Multi-scale network with wavelet transform.

### Features
- Multi-resolution processing
- Wavelet domain learning
- Better edge preservation

## UPFWI (Uncertainty-aware)

### Architecture
Bayesian neural network for uncertainty quantification.

### Features
- Monte Carlo dropout
- Ensemble predictions
- Uncertainty maps
