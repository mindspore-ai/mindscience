# OpenFWI Inference

## Single Sample Inference

```python
import torch
import numpy as np
from model import InversionNet

# Load model
model = InversionNet()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Load seismic data
seismic = np.load('seismic.npy')  # Shape: (n_shots, n_time, n_receivers)
seismic_tensor = torch.from_numpy(seismic).float().unsqueeze(0)

# Run inference
with torch.no_grad():
    velocity_pred = model(seismic_tensor)
```

## Batch Inference

```python
from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

predictions = []
with torch.no_grad():
    for seismic, _ in test_loader:
        velocity_pred = model(seismic)
        predictions.append(velocity_pred.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
```

## Uncertainty Quantification

### Monte Carlo Dropout
```python
def predict_with_uncertainty(model, input, n_samples=100):
    model.train()  # Enable dropout
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(input)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    
    return mean, std
```

### Ensemble Prediction
```python
models = [load_model(f'model_{i}.pth') for i in range(5)]

predictions = []
for model in models:
    with torch.no_grad():
        pred = model(input)
        predictions.append(pred.cpu().numpy())

mean = np.mean(predictions, axis=0)
std = np.std(predictions, axis=0)
```

## Post-processing

### Denormalization
```python
# If normalized during training
velocity_real = velocity_pred * (v_max - v_min) + v_min
```

### Smoothing
```python
from scipy.ndimage import gaussian_filter

velocity_smooth = gaussian_filter(velocity_pred, sigma=1.0)
```

## Visualization

```python
import matplotlib.pyplot as plt

def plot_velocity(velocity, title='Velocity Model'):
    plt.figure(figsize=(10, 6))
    plt.imshow(velocity, cmap='jet', aspect='auto')
    plt.colorbar(label='Velocity (m/s)')
    plt.title(title)
    plt.xlabel('X (grid points)')
    plt.ylabel('Z (grid points)')
    plt.show()
```

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| MSE | mean((pred - true)^2) | Mean squared error |
| MAE | mean(|pred - true|) | Mean absolute error |
| SSIM | Structural similarity | Perceptual quality |
| PSNR | 10 * log10(MAX^2 / MSE) | Peak signal-to-noise |
