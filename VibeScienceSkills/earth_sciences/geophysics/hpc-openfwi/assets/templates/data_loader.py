#!/usr/bin/env python
"""
Custom Data Loader for OpenFWI
"""

import os
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import torch


class FWIDataset(Dataset):
    """
    Full Waveform Inversion Dataset
    
    Args:
        data_path: Path to HDF5 file or directory containing data
        split: 'train', 'val', or 'test'
        transform: Optional transforms to apply
    """
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # Load data indices
        if os.path.isdir(data_path):
            self.files = sorted([os.path.join(data_path, f) 
                                for f in os.listdir(data_path) 
                                if f.endswith('.h5') or f.endswith('.npy')])
        else:
            self.files = [data_path]
        
        # Determine split indices
        total = len(self.files)
        if split == 'train':
            self.indices = range(0, int(0.8 * total))
        elif split == 'val':
            self.indices = range(int(0.8 * total), int(0.9 * total))
        else:  # test
            self.indices = range(int(0.9 * total), total)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_path = self.files[self.indices[idx]]
        
        if file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                seismic = f['seismic'][:]
                velocity = f['velocity'][:]
        else:
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                seismic = data.item()['seismic']
                velocity = data.item()['velocity']
            else:
                seismic, velocity = data
        
        # Normalize
        seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        velocity = (velocity - velocity.min()) / (velocity.max() - velocity.min() + 1e-8)
        
        # Convert to tensor
        seismic = torch.from_numpy(seismic).float()
        velocity = torch.from_numpy(velocity).float()
        
        if self.transform:
            seismic = self.transform(seismic)
            velocity = self.transform(velocity)
        
        return seismic, velocity


def get_dataloader(data_path, batch_size=16, num_workers=4, split='train'):
    """Create DataLoader for FWI dataset"""
    dataset = FWIDataset(data_path, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                      num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    # Test data loader
    loader = get_dataloader('/path/to/data.h5', batch_size=4)
    
    for seismic, velocity in loader:
        print(f'Seismic shape: {seismic.shape}')
        print(f'Velocity shape: {velocity.shape}')
        break
