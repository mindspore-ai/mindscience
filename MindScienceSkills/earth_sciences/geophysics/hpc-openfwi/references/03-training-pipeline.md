# OpenFWI Training Pipeline

## Data Preparation

### Download Data
```bash
# From OpenFWI repository
wget https://opengfwi.blob.core.windows.net/opengfwi/Vel-1.0.h5
```

### Data Splitting
```python
# Train/Val/Test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
```

## Training Configuration

### Basic Setup
```python
import torch
from torch.utils.data import DataLoader

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = InversionNet().to(device)

# Loss
criterion = nn.L1Loss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

### Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| L1 | |x - y| | Smooth results |
| L2 | (x - y)^2 | Penalize outliers |
| Perceptual | ||phi(x) - phi(y)|| | Detailed structures |
| Adversarial | BCE(G(z), 1) | Realistic textures |

### Optimizers

| Optimizer | Recommended LR | Use Case |
|-----------|---------------|----------|
| Adam | 1e-4 | Default |
| AdamW | 1e-4 | Weight decay |
| SGD | 1e-2 | Fine-tuning |

## Multi-GPU Training

### DataParallel
```python
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### DistributedDataParallel
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_ddp.py
```

## Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Monitoring

### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_image('Velocity/pred', pred_velocity, epoch)
```

### Checkpointing
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```
