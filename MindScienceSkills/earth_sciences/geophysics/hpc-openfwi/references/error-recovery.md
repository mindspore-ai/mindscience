# OpenFWI Error Recovery

## Common Errors

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size
```python
batch_size = 8  # Reduce from 16
```

2. Enable mixed precision
```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
```

3. Use gradient checkpointing
```python
from torch.utils.checkpoint import checkpoint

# In model forward
x = checkpoint(self.encoder_block, x)
```

### Slow Training

**Causes and Solutions**:

1. **Single GPU bottleneck**
```bash
# Use multi-GPU
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

2. **Data loading bottleneck**
```python
# Increase workers
DataLoader(dataset, num_workers=8, pin_memory=True)
```

3. **I/O bottleneck**
```python
# Cache data in memory
data = h5py.File('data.h5', 'r', driver='core')
```

### GAN Training Issues

**Problem**: Mode collapse

**Solutions**:
1. Adjust loss weights
```python
lambda_l1 = 100.0  # Increase L1 weight
```

2. Use spectral normalization
```python
from torch.nn.utils import spectral_norm

self.conv = spectral_norm(nn.Conv2d(...))
```

3. Train discriminator more
```python
# Train D twice per G iteration
for _ in range(2):
    train_discriminator()
train_generator()
```

### Poor Inversion Quality

**Causes and Solutions**:

| Symptom | Cause | Solution |
|---------|-------|----------|
| Blurry results | L2 loss | Use L1 or perceptual loss |
| Missing structures | Insufficient training | Increase epochs, use transfer learning |
| Artifacts | Overfitting | Add dropout, data augmentation |
| Wrong scale | Normalization issue | Check data preprocessing |

### Data Loading Errors

**Error**: `KeyError: 'seismic'`

**Solution**: Check HDF5 file structure
```python
import h5py
with h5py.File('data.h5', 'r') as f:
    print(list(f.keys()))
```

**Error**: `ValueError: could not broadcast input array`

**Solution**: Check data shapes
```python
print(f'Seismic shape: {seismic.shape}')
print(f'Velocity shape: {velocity.shape}')
```

## Debugging Tips

1. **Check GPU memory**
```python
print(torch.cuda.memory_allocated() / 1e9, 'GB')
print(torch.cuda.memory_reserved() / 1e9, 'GB')
```

2. **Profile training**
```python
with torch.profiler.profile() as prof:
    model(input)
print(prof.key_averages().table())
```

3. **Visualize predictions**
```python
# During training
if epoch % 10 == 0:
    plt.imshow(velocity_pred[0].cpu().detach())
    plt.savefig(f'pred_epoch_{epoch}.png')
```
