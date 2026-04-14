#!/usr/bin/env python
"""
Inference Script for OpenFWI Models
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from model.InversionNet import InversionNet
from model.VelocityGAN import Generator


def load_model(model_path, model_type='inversionnet', device='cuda'):
    """Load trained model"""
    if model_type == 'inversionnet':
        model = InversionNet()
    elif model_type == 'velocitygan':
        model = Generator()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def inference(model, seismic_data, device):
    """Run inference on seismic data"""
    with torch.no_grad():
        seismic_tensor = torch.from_numpy(seismic_data).float().to(device)
        if seismic_tensor.dim() == 3:
            seismic_tensor = seismic_tensor.unsqueeze(0)
        
        velocity_pred = model(seismic_tensor)
        return velocity_pred.cpu().numpy()


def plot_velocity(velocity, title='Velocity Model', save_path=None):
    """Plot velocity model"""
    plt.figure(figsize=(10, 6))
    plt.imshow(velocity, cmap='jet', aspect='auto')
    plt.colorbar(label='Velocity (m/s)')
    plt.title(title)
    plt.xlabel('X (grid points)')
    plt.ylabel('Z (grid points)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.model_type, device)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    if args.data_path.endswith('.h5'):
        with h5py.File(args.data_path, 'r') as f:
            seismic = f['seismic'][:]
            if 'velocity' in f:
                velocity_true = f['velocity'][:]
            else:
                velocity_true = None
    else:
        seismic = np.load(args.data_path)
        velocity_true = None
    
    # Run inference
    print("Running inference...")
    velocity_pred = inference(model, seismic, device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'velocity_pred.npy'), velocity_pred)
    
    # Plot
    if velocity_pred.ndim == 3:
        velocity_pred = velocity_pred[0]
    
    plot_velocity(velocity_pred, title='Predicted Velocity Model',
                  save_path=os.path.join(args.output_dir, 'velocity_pred.png'))
    
    if velocity_true is not None:
        if velocity_true.ndim == 3:
            velocity_true = velocity_true[0]
        plot_velocity(velocity_true, title='True Velocity Model',
                      save_path=os.path.join(args.output_dir, 'velocity_true.png'))
        
        # Compute error
        error = np.abs(velocity_pred - velocity_true)
        plot_velocity(error, title='Absolute Error',
                      save_path=os.path.join(args.output_dir, 'error.png'))
        
        mse = np.mean((velocity_pred - velocity_true) ** 2)
        mae = np.mean(np.abs(velocity_pred - velocity_true))
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    print(f"Results saved to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='OpenFWI Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--model_type', type=str, default='inversionnet', 
                        choices=['inversionnet', 'velocitygan'])
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
