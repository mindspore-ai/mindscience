#!/usr/bin/env python
"""
InversionNet Training Script for OpenFWI
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from model.InversionNet import InversionNet
from data.dataset import FWIDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train InversionNet')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, amp=False):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (seismic, velocity) in enumerate(dataloader):
        seismic = seismic.to(device)
        velocity = velocity.to(device)
        
        optimizer.zero_grad()
        
        if amp:
            with autocast():
                output = model(seismic)
                loss = criterion(output, velocity)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(seismic)
            loss = criterion(output, velocity)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for seismic, velocity in dataloader:
            seismic = seismic.to(device)
            velocity = velocity.to(device)
            
            output = model(seismic)
            loss = criterion(output, velocity)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = InversionNet().to(device)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision
    scaler = GradScaler() if args.amp else None
    
    # Data loaders
    train_dataset = FWIDataset(args.data_path, split='train')
    val_dataset = FWIDataset(args.data_path, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, 
                                      scaler, device, args.amp)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  Saved best model (val_loss: {val_loss:.6f})')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()
