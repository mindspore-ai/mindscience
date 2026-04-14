#!/usr/bin/env python
"""
BETE-NET Complete Training Script - MindSpore Version
With comprehensive progress tracking and periodic result display
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'notebooks_ms'))
import setup_paths
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.mint import optim
from mindspore.experimental import optim as optimex
sys.path.append('./notebooks_ms')
import utils_data_ms as data_utils
import utils_model_ms as model_utils

import pandas as pd
import numpy as np
from tqdm import tqdm
import ase.io
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sharker batch processing
from sharker.data import Batch

# Set random seeds
np.random.seed(42)
ms.set_seed(42)

def save_training_state(epoch, train_losses, val_losses, config, start_time, save_path="training_state.json"):
    """Save current training state"""
    state = {
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'start_time': start_time.isoformat(),
        'current_time': datetime.now().isoformat(),
        'elapsed_hours': (datetime.now() - start_time).total_seconds() / 3600
    }
    with open(save_path, 'w') as f:
        json.dump(state, f, indent=2)

def load_training_state(save_path="training_state.json"):
    """Load training state if exists"""
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            state = json.load(f)
        state['start_time'] = datetime.fromisoformat(state['start_time'])
        return state
    return None

def plot_training_progress(train_losses, val_losses, config_name, save_path=None):
    """Plot and save training progress"""
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7, linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{config_name} Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Recent progress (last 20 epochs)
    plt.subplot(2, 2, 2)
    recent_epochs = max(1, len(train_losses) - 20)
    recent_range = range(recent_epochs, len(train_losses) + 1)
    plt.plot(recent_range, train_losses[recent_epochs-1:], 'b-', label='Training Loss', linewidth=2)
    plt.plot(recent_range, val_losses[recent_epochs-1:], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Recent Progress (Last 20 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss distribution
    plt.subplot(2, 2, 3)
    plt.hist(train_losses, bins=20, alpha=0.7, label='Training Loss', color='blue')
    plt.hist(val_losses, bins=20, alpha=0.7, label='Validation Loss', color='red')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution')
    plt.legend()
    
    # Training statistics
    plt.subplot(2, 2, 4)
    stats_text = f"""Training Statistics:
    
Current Epoch: {len(train_losses)}
Best Train Loss: {min(train_losses):.6f}
Best Val Loss: {min(val_losses):.6f}
Current Train Loss: {train_losses[-1]:.6f}
Current Val Loss: {val_losses[-1]:.6f}

Improvement Rate:
Train: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%
Val: {((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.2f}%
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', 
             transform=plt.gca().transAxes, fontfamily='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Progress plot saved to: {save_path}")
    
    plt.show()

def display_progress_summary(epoch, total_epochs, train_loss, val_loss, train_losses, val_losses, 
                           start_time, config_name, model_params):
    """Display comprehensive progress summary"""
    current_time = datetime.now()
    elapsed = current_time - start_time
    
    # Calculate ETA
    if epoch > 0:
        avg_time_per_epoch = elapsed.total_seconds() / epoch
        remaining_epochs = total_epochs - epoch
        eta = current_time + timedelta(seconds=avg_time_per_epoch * remaining_epochs)
    else:
        eta = None
    
    # Best losses
    best_train = min(train_losses)
    best_val = min(val_losses)
    best_train_epoch = train_losses.index(best_train) + 1
    best_val_epoch = val_losses.index(best_val) + 1
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING PROGRESS SUMMARY - {config_name}")
    print(f"{'='*80}")
    print(f"📊 Progress: Epoch {epoch}/{total_epochs} ({epoch/total_epochs*100:.1f}%)")
    print(f"⏱️  Elapsed Time: {str(elapsed).split('.')[0]}")
    if eta:
        print(f"🎯 ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')} (≈{str(timedelta(seconds=avg_time_per_epoch * remaining_epochs)).split('.')[0]} remaining)")
    
    print(f"\n📈 Current Performance:")
    print(f"   Training Loss:   {train_loss:.6f}")
    print(f"   Validation Loss: {val_loss:.6f}")
    print(f"   Loss Ratio:      {val_loss/train_loss:.3f}")
    
    print(f"\n🏆 Best Performance:")
    print(f"   Best Train Loss: {best_train:.6f} (Epoch {best_train_epoch})")
    print(f"   Best Val Loss:   {best_val:.6f} (Epoch {best_val_epoch})")
    
    if len(train_losses) >= 5:
        recent_train_trend = np.mean(train_losses[-5:]) - np.mean(train_losses[-10:-5]) if len(train_losses) >= 10 else 0
        recent_val_trend = np.mean(val_losses[-5:]) - np.mean(val_losses[-10:-5]) if len(val_losses) >= 10 else 0
        
        print(f"\n📊 Recent Trends (Last 5 epochs):")
        trend_train = "📈 Increasing" if recent_train_trend > 0 else "📉 Decreasing" if recent_train_trend < 0 else "➡️ Stable"
        trend_val = "📈 Increasing" if recent_val_trend > 0 else "📉 Decreasing" if recent_val_trend < 0 else "➡️ Stable"
        print(f"   Training:   {trend_train} ({recent_train_trend:+.6f})")
        print(f"   Validation: {trend_val} ({recent_val_trend:+.6f})")
    
    print(f"\n🏗️  Model Configuration:")
    # print(f"   Parameters: {sum(p.size for p in model_params):,}")
    print(f"   Input Dim:  {model_params.get('input_dim', 'N/A')}")
    print(f"   Output Dim: {model_params.get('output_dim', 'N/A')}")
    
    print(f"{'='*80}\n")

def full_training(config_name="CPD", max_epochs=100, display_interval=5, plot_interval=10, batch_size=32, idx=0):
    """Complete training with progress tracking"""
    
    print(f"🚀 BETE-NET Full Training - {config_name} Configuration")
    print(f"{'='*60}")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️  Configuration: {config_name}")
    print(f"🔄 Max Epochs: {max_epochs}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"📊 Display Interval: Every {display_interval} epochs")
    print(f"📈 Plot Interval: Every {plot_interval} epochs")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    # Configuration settings
    configs = {
        'CSO': {'in_dim':118, 'embed_ph_dos': False, 'embed_e_dos': False, 'fine': False, 'layers': 2, 'mul': 32, 'lr': 0.005},
        'CPD': {'in_dim':118 + 51, 'embed_ph_dos': True, 'embed_e_dos': False, 'fine': False, 'layers': 2, 'mul': 32, 'lr': 0.005},
        'FPD': {'in_dim':118 + 51, 'embed_ph_dos': True, 'embed_e_dos': False, 'fine': True, 'layers': 2, 'mul': 32, 'lr': 0.005}
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name]
    
    # Load data
    print("📊 Loading database...")
    df = pd.read_json('database.json')
    df.dropna(inplace=True)
    print(f"✅ Loaded {len(df)} samples from database")
    
    # Load structures
    print("🔬 Loading crystal structures...")
    structures = []
    for index, row in tqdm(df.iterrows(), desc="Loading structures", ncols=100):
        try:
            structures.append(ase.io.read(f'structures/{index}.cif'))
        except Exception as e:
            structures.append(None)
    
    df['structure'] = structures
    df = df[df['structure'].notna()]
    print(f"✅ Successfully loaded {len(df)} crystal structures")
    
    # Process data
    print(f"⚙️  Processing data for {config_name} configuration...")
    r_max = 4
    df['target'] = df.apply(data_utils.get_target, axis=1)
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    
    tqdm.pandas(desc="Building graph data", ncols=100)
    df['data'] = df.progress_apply(
        data_utils.build_data, 
        embed_ph_dos=config['embed_ph_dos'],
        embed_e_dos=config['embed_e_dos'],
        fine=config['fine'], 
        r_max=r_max, 
        axis=1
    )
    
    # Get dimensions
    sample_data = df.iloc[0]['data']
    out_dim = len(df.iloc[0]['target'])
    in_dim = sample_data.x[0]
    em_dim = 64
    
    print(f"📊 Data Information:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Input features: {in_dim}")
    print(f"   - Output targets: {out_dim}")
    print(f"   - Configuration: {config_name}")
    
    # 数据划分
    print(f"📋 使用原版论文的数据划分...,共{len(df)}条数据")
    train_df, test_df, val_df = data_utils.get_original_data_split(df, idx)
    
    # 从训练集中进一步划分验证集
    if val_df is None:
        val_split_idx = int(len(train_df) * 0.8)
        val_df = train_df.iloc[val_split_idx:].copy()
        train_df = train_df.iloc[:val_split_idx].copy()
    
    print(f"📊 Data Split:")
    print(f"   - Training: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   - Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   - Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    # Create model
    model_params = {
        'in_dim': config['in_dim'],                           
        'em_dim': 64,
        'irreps_in': f'{em_dim}x0e',
        'irreps_out': f'{out_dim}x0e',
        'irreps_node_attr': f'{em_dim}x0e',
        'layers': config['layers'],
        'mul': config['mul'],
        'lmax': 1,
        'max_radius': r_max,
        'number_of_basis': 10,
        'radial_layers': 1,
        'radial_neurons': 128,
        'num_neighbors': data_utils.get_neighbors(train_df, train_df.index).mean(),
        'num_nodes': 8.0,
        'reduce_output': True,
        'dropout': False,
        'input_dim': in_dim,
        'output_dim': out_dim
    }
    
    print("run full training")
    if config_name == 'CSO':
        model = model_utils.PeriodicNetwork(**{k: v for k, v in model_params.items() if k not in ['input_dim', 'output_dim']})
        model.pool = True
    elif config_name == 'CPD':
        model = model_utils.PeriodicNetwork(**{k: v for k, v in model_params.items() if k not in ['input_dim', 'output_dim']})
    elif config_name == 'FPD':
        model = model_utils.PeriodicNetwork(**{k: v for k, v in model_params.items() if k not in ['input_dim', 'output_dim']})
    param_count = sum(p.size for p in model.get_parameters())
    print(f"🏗️  Model created with {param_count:,} parameters")
    
    # Training setup
    # def loss_func(pred, target):
    #     lambda_pred = 0
    #     freq_w = ms.ops.arange(0.25, 101, 2)
    #     for i in range(1, len(freq_w)):
    #         dw = freq_w[i] - freq_w[i-1]
    #         w = freq_w[i]
    #         alpha_F_w = pred[i]
    #         lambda_pred = lambda_pred + ((alpha_F_w/w)*dw)
    #     lambda_target = 0
    #     for i in range(1, len(freq_w)):
    #         dw = freq_w[i] - freq_w[i-1]
    #         w = freq_w[i]
    #         alpha_F_w = target[i]
    #         lambda_target = lambda_target + ((alpha_F_w/w)*dw)
    #     return (lambda_pred - lambda_target).abs().mean()
    #     return nn.MSELoss(pred, target) + 5 * nn.MAELoss(pred, target)
    # # loss_fn = loss_func #nn.MSELoss()
    # loss_fn = loss_func
    loss_fn = nn.MSELoss()
    optimizer = optimex.AdamW(model.trainable_params(), lr=config['lr'])
    scheduler = optimex.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180],gamma=0.3)
    # scheduler = optimex.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    def forward_fn(data, targets):
        pred = model(data)
        loss = loss_fn(pred, targets)
        return loss, pred
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    
    # Training variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print(f"\n🎯 Starting Training...")
    print(f"   Learning Rate: {config['lr']}")
    print(f"   Early Stopping Patience: {patience}")
    print(f"   Model Save Path: best_{config_name.lower()}_model_ms.ckpt")
    
    def create_batches(df, batch_size, shuffle=True):
        """Create batches from dataframe using Sharker's Batch.from_data_list"""
        # indices = df.index.tolist()
        indices = np.arange(0, len(df), 1)
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            if len(batch_indices) == 1:
                # Single sample - no batching needed
                data = df.loc[batch_indices[0], 'data']
                target = ms.Tensor([df.loc[batch_indices[0], 'target']], dtype=ms.float32)
                batches.append((data, target))
            else:
                # Multiple samples - use Sharker batching
                data_list = [df.iloc[idx]['data'] for idx in batch_indices]
                targets = [df.iloc[idx]['target'] for idx in batch_indices]

                batch_data = Batch.from_data_list(data_list)
                batch_targets = ms.Tensor(targets, dtype=ms.float32)
                batches.append((batch_data, batch_targets))
        
        return batches
    
    # Training loop
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        
        # Create batches for this epoch
        train_batches = create_batches(train_df, batch_size, shuffle=True)
        val_batches = create_batches(val_df, batch_size, shuffle=False)
        
        # Training phase
        model.set_train()
        train_loss = 0.0
        train_count = 0
        
        train_pbar = tqdm(train_batches, desc=f"Epoch {epoch+1:3d}/{max_epochs} [Train]", 
                         ncols=100, leave=False)
        
        for batch_data, batch_targets in train_pbar:
            if len(batch_targets.shape) == 1:
                # Single sample case
                batch_targets = batch_targets.expand_dims(0)
            
            scheduler.step()
            (loss, pred), grads = grad_fn(batch_data, batch_targets)
            optimizer(grads)

            current_lr = scheduler.get_last_lr()
            print(current_lr)

            current_loss = float(loss.asnumpy())
            train_loss += current_loss
            train_count += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.6f}',
                'Avg': f'{train_loss/train_count:.6f}',
                'Batch': f'{batch_targets.shape[0]}'
            })
        
        avg_train_loss = train_loss / train_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.set_train(False)
        val_loss = 0.0
        val_count = 0
        
        val_pbar = tqdm(val_batches, desc=f"Epoch {epoch+1:3d}/{max_epochs} [Val]", 
                       ncols=100, leave=False)
        
        for batch_data, batch_targets in val_pbar:
            if len(batch_targets.shape) == 1:
                # Single sample case
                batch_targets = batch_targets.expand_dims(0)
            
            pred = model(batch_data)
            loss = loss_fn(pred, batch_targets)
            
            current_loss = float(loss.asnumpy())
            val_loss += current_loss
            val_count += 1
            
            val_pbar.set_postfix({
                'Loss': f'{current_loss:.6f}',
                'Avg': f'{val_loss/val_count:.6f}',
                'Batch': f'{batch_targets.shape[0]}'
            })
        
        avg_val_loss = val_loss / val_count
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Basic epoch summary
        improvement = "🟢" if avg_val_loss < best_val_loss else "🔴"
        print(f"Epoch {epoch+1:3d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f} {improvement} ({epoch_time:.1f}s)")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ms.save_checkpoint(model, f"./{config_name.lower()}/best_{config_name.lower()}_model_ms_{idx}.ckpt")
            patience_counter = 0
            print(f"   ✅ New best model saved! (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Detailed progress display
        if (epoch + 1) % display_interval == 0:
            display_progress_summary(epoch + 1, max_epochs, avg_train_loss, avg_val_loss,
                                   train_losses, val_losses, start_time, config_name, model_params)
        
        # Plot progress
        if (epoch + 1) % plot_interval == 0:
            plot_path = f"{config_name.lower()}_training_progress_epoch_{epoch+1}.png"
            plot_training_progress(train_losses, val_losses, config_name, plot_path)
        
        # Save training state
        save_training_state(epoch + 1, train_losses, val_losses, config, start_time,
                           f"{config_name.lower()}_training_state.json")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Final evaluation
    # print(f"\n🧪 Final Evaluation on Test Set...")
    # model.set_train(False)
    
    # test_predictions = []
    # test_targets = []
    
    # for idx in tqdm(test_df.index, desc="Testing", ncols=100):
    #     data = test_df.loc[idx, 'data']
    #     target = test_df.loc[idx, 'target']
        
    #     pred = model(data)
    #     pred_np = pred.asnumpy().flatten()
        
    #     test_predictions.append(pred_np)
    #     test_targets.append(target)
    
    # # Calculate final metrics
    # all_preds = np.concatenate(test_predictions)
    # all_targets = np.concatenate(test_targets)
    
    # mae = mean_absolute_error(all_targets, all_preds)
    # rmse = mean_squared_error(all_targets, all_preds)
    # r2 = r2_score(all_targets, all_preds)
    
    # # Final summary
    # total_time = datetime.now() - start_time
    # print(f"\n{'='*60}")
    # print(f"🎉 TRAINING COMPLETED - {config_name}")
    # print(f"{'='*60}")
    # print(f"⏱️  Total Training Time: {str(total_time).split('.')[0]}")
    # print(f"📊 Total Epochs: {len(train_losses)}")
    # print(f"🏆 Best Validation Loss: {best_val_loss:.6f}")
    # print(f"\n📊 Final Test Results:")
    # print(f"   - MAE:  {mae:.6f}")
    # print(f"   - RMSE: {rmse:.6f}")
    # print(f"   - R²:   {r2:.6f}")
    # print(f"\n📁 Saved Files:")
    # print(f"   - Model: best_{config_name.lower()}_model_ms.ckpt")
    # print(f"   - State: {config_name.lower()}_training_state.json")
    # print(f"   - Plots: {config_name.lower()}_training_progress_*.png")
    # print(f"{'='*60}")
    
    # Final plot
    final_plot_path = f"{config_name.lower()}_final_training_results.png"
    plot_training_progress(train_losses, val_losses, config_name, final_plot_path)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        # 'test_predictions': test_predictions,
        # 'test_targets': test_targets,
        # 'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        # 'total_time': total_time,
        'config': config_name
    }

if __name__ == "__main__":
    # Configuration - Change this to run different models
    CONFIG = "CPD"  # Options: "CSO", "CPD", "FPD"
    MAX_EPOCHS = 100
    BATCH_SIZE = 256    # Batch size for training
    DISPLAY_INTERVAL = 5   # Show detailed progress every N epochs
    PLOT_INTERVAL = 10     # Generate plots every N epochs
    
    print(f"🚀 Starting BETE-NET Training")
    print(f"Configuration: {CONFIG}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Progress Display: Every {DISPLAY_INTERVAL} epochs")
    print(f"Plot Generation: Every {PLOT_INTERVAL} epochs")
    print(f"\nPress Ctrl+C to stop training gracefully...")
    
    try:
        results = full_training(
            config_name=CONFIG,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            display_interval=DISPLAY_INTERVAL,
            plot_interval=PLOT_INTERVAL
        )
        print(f"\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        print(f"Training state has been saved and can be resumed later")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc() 