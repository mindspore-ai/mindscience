#!/usr/bin/env python3

# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a derivative work based on the original PowerFlowNet implementation
# (https://github.com/stavrosorf/poweflownet) which was licensed under the MIT License.
# Significant modifications have been made to adapt the code for the MindSpore framework,
# including MindSpore equivalents and
# optimization for Ascend hardware acceleration.
# ============================================================================
"""
Training script for MindSpore PowerFlowNet
Supports both v2 dataset (4D features) and legacy dataset (12D features)
- v2 dataset: Compatible with mlp, mpn, gcn, mask_embed_multi_mpn, mpn_simplenet
- legacy dataset: Additionally supports skip_mpn, mask_embed_mpn, multi_mpn, etc.
Includes logging, weight saving, and visualization
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn

from configs.config import init_device
from src import (
    MLPNet, MPN, GCNNet, SkipMPN, MaskEmbedMPN, MultiMPN,
    MaskEmbedMultiMPN, MaskEmbedMultiMPNNoMP, MultiConvNet, MPNSimplenet
)
from src.custom_loss_functions import MaskedL2Loss, PowerImbalance, MixedMSEPowerImbalance
from src.power_flow_data import PowerFlowDataV2, PowerFlowDataLoaderV2

# Unset RANK_TABLE_FILE to avoid Ascend distributed training mode
# which forces JIT level O2 and causes optimizer compilation issues
if 'RANK_TABLE_FILE' in os.environ:
    del os.environ['RANK_TABLE_FILE']

sys.path.insert(0, str(Path(__file__).parent))

# Check if running on Ascend - use PYNATIVE mode for compatibility
def is_ascend():
    """Check if running on Ascend device"""
    try:
        device_target = ms.get_context("device_target")
        return device_target == "Ascend"
    except Exception:
        return False


class MaskedMSELoss(nn.Cell):
    """Masked MSE loss - for backward compatibility and simple loss"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def construct(self, pred, target, mask):
        loss = self.mse(pred, target)
        return (loss * mask).mean()





class Trainer:
    """Training manager with logging and checkpointing"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 args, log_dir='logs', model_dir='models',
                 norm_stats=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.norm_stats = norm_stats  # (xymean, xystd, edgemean, edgestd)

        # Setup directories
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(log_dir) / args.case / f"{args.model}_{self.run_id}"
        self.model_dir = Path(model_dir) / args.case / f"{args.model}_{self.run_id}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self.optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)

        # Create loss function based on args
        self.setup_loss_function()

        # History for logging
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': None,
            'epochs': [],
        }

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = args.patience
        self.patience_counter = 0

        # Save config
        self.save_config()

    def setup_loss_function(self):
        """Setup loss function based on training configuration"""
        if hasattr(self.args, 'train_loss_fn'):
            loss_fn_type = self.args.train_loss_fn.lower()
        else:
            loss_fn_type = 'masked_mse'  # default

        if loss_fn_type in ('masked_l2', 'masked_mse'):
            self.loss_fn = MaskedL2Loss()
            self.eval_loss_fn = MaskedL2Loss()
        elif loss_fn_type == 'power_imbalance':
            # Physics-informed power imbalance loss
            if self.norm_stats is not None:
                xymean, xystd, edgemean, edgestd = self.norm_stats
                self.loss_fn = PowerImbalance(xymean, xystd, edgemean, edgestd)
            else:
                print("⚠ Warning: norm_stats not provided, falling back to MaskedL2Loss")
                self.loss_fn = MaskedL2Loss()
            self.eval_loss_fn = MaskedL2Loss()
        elif loss_fn_type == 'mixed_mse_power_imbalance':
            # Mixed MSE + Power Imbalance loss
            if self.norm_stats is not None:
                xymean, xystd, edgemean, edgestd = self.norm_stats
                self.loss_fn = MixedMSEPowerImbalance(xymean, xystd, edgemean, edgestd, alpha=0.9)
            else:
                print("⚠ Warning: norm_stats not provided, falling back to MaskedL2Loss")
                self.loss_fn = MaskedL2Loss()
            self.eval_loss_fn = MaskedL2Loss()
        else:
            # Default simple MSE loss
            self.loss_fn = MaskedMSELoss()
            self.eval_loss_fn = MaskedMSELoss()

    def save_config(self):
        """Save training configuration"""
        config = {
            'case': self.args.case,
            'model': self.args.model,
            'epochs': self.args.epochs,
            'batch_size': self.args.batch_size,
            'lr': self.args.lr,
            'hidden_dim': self.args.hidden_dim,
            'device': self.args.device,
            'run_id': self.run_id,
        }
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved to {config_path}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.set_train(True)
        total_loss = 0.0
        num_batches = 0

        # Define gradient function based on loss type
        loss_fn_type = getattr(self.args, 'train_loss_fn', 'masked_l2').lower()

        if loss_fn_type == 'power_imbalance':
            def forward_fn(batch):
                pred = self.model(batch)
                masked_pred = pred * batch.pred_mask + batch.x * (1 - batch.pred_mask)
                loss = self.loss_fn(masked_pred, batch.edge_index, batch.edge_attr)
                return loss
        elif loss_fn_type == 'mixed_mse_power_imbalance':
            def forward_fn(batch):
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.edge_index, batch.edge_attr, batch.y)
                return loss
        else:
            def forward_fn(batch):
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y, batch.pred_mask)
                return loss

        grad_fn = ms.value_and_grad(forward_fn, None, self.model.trainable_params())

        for batch in self.train_loader:
            # Compute loss and gradients
            loss, grads = grad_fn(batch)

            # Update parameters
            self.optimizer(grads)

            total_loss += float(loss.asnumpy())
            num_batches += 1
        return total_loss / max(1, num_batches)

    def validate(self):
        """Validate model"""
        self.model.set_train(False)
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # batch is already a Batch object from DataLoader
            pred = self.model(batch)
            loss = self.eval_loss_fn(pred, batch.y, batch.pred_mask)
            total_loss += float(loss.asnumpy())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def test(self):
        """Test model"""
        self.model.set_train(False)
        total_loss = 0.0
        num_batches = 0

        for batch in self.test_loader:
            # batch is already a Batch object from DataLoader
            pred = self.model(batch)
            loss = self.eval_loss_fn(pred, batch.y, batch.pred_mask)
            total_loss += float(loss.asnumpy())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def save_model(self, epoch):
        """Save model checkpoint"""
        model_path = self.model_dir / f'model_epoch_{epoch}.ckpt'
        ms.save_checkpoint(self.model, str(model_path))
        print(f"✓ Model saved to {model_path}")

    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Training {self.args.model.upper()} on {self.args.case}")
        print(f"{'='*60}")

        for epoch in range(1, self.args.epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'].append(epoch)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | {elapsed:.2f}s")

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(epoch)
            else:
                self.patience_counter += 1
                # patience=0 means disabled
                if self.patience > 0 and self.patience_counter >= self.patience:
                    print(f"\n✓ Early stopping at epoch {epoch}")
                    break

        # Test
        print(f"\n{'='*60}")
        test_loss = self.test()
        self.history['test_loss'] = test_loss
        print(f"Test Loss: {test_loss:.6f}")
        print(f"{'='*60}\n")

        # Save history
        self.save_history()
        self.plot_training()

    def save_history(self):
        """Save training history"""
        history_path = self.log_dir / 'history.json'
        history = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_loss': [float(x) for x in self.history['val_loss']],
            'test_loss': float(self.history['test_loss']) if self.history['test_loss'] else None,
            'epochs': self.history['epochs'],
        }
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        print(f"✓ History saved to {history_path}")

    def plot_training(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epochs'], self.history['train_loss'], 'b-', label='Train Loss')
        plt.plot(self.history['epochs'], self.history['val_loss'], 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{self.args.model.upper()} Training Curve')
        plt.grid(True, alpha=0.3)

        plot_path = self.log_dir / 'training_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved to {plot_path}")


def _get_model_config(args) -> dict:
    """Extract model configuration from arguments."""
    return {
        'nfeature_dim': 4,
        'efeature_dim': 2,
        'output_dim': 4,
        'hidden_dim': args.hidden_dim,
        'n_gnn_layers': args.n_layers,
        'k': args.k,
        'dropout_rate': args.dropout,
    }


def _build_gnn_model(model_name: str, config: dict):
    """Build GNN models (MPN variants and GCN)."""
    gnn_map = {
        'mpn': MPN,
        'skip_mpn': SkipMPN,
        'mask_embed_mpn': MaskEmbedMPN,
        'multi_mpn': MultiMPN,
        'mask_embed_multi_mpn': MaskEmbedMultiMPN,
        'mask_embed_multi_mpn_nomp': MaskEmbedMultiMPNNoMP,
        'mpn_simplenet': MPNSimplenet,
    }

    if model_name in gnn_map:
        return gnn_map[model_name](**config)
    if model_name == 'multi_conv_net':
        config['efeature_dim'] = 5  # MultiConvNet requires 5 edge features
        return MultiConvNet(**config)

    raise ValueError(f"Unknown GNN model: {model_name}")


def _build_simple_model(model_name: str, args) -> nn.Cell:
    """Build MLP or GCN models."""
    if model_name == 'mlp':
        return MLPNet(
            nfeature_dim=4, output_dim=4,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
            dropout_rate=args.dropout
        )
    if model_name == 'gcn':
        return GCNNet(
            nfeature_dim=4, output_dim=4,
            hidden_dim=args.hidden_dim, n_gnn_layers=args.n_layers,
            dropout_rate=args.dropout
        )

    raise ValueError(f"Unknown simple model: {model_name}")


def create_model(args) -> nn.Cell:
    """
    Create model based on args.

    Models are divided into:
    1. Simple models (MLP, GCN): 4D input
    2. GNN models (MPN variants): 4D or 12D input
    """
    # Try simple models first
    simple = _build_simple_model(args.model, args)
    if simple is not None:
        return simple

    # Build GNN models
    config = _get_model_config(args)
    gnn = _build_gnn_model(args.model, config)
    if gnn is not None:
        return gnn

    raise ValueError(f"Unknown model: {args.model}")


def _setup_training_parser() -> argparse.ArgumentParser:
    """Setup argument parser for training."""
    parser = argparse.ArgumentParser(description='Train PowerFlowNet with multiple dataset formats')

    parser.add_argument('--case', type=str, default='14',
                        help='Case name. Use v2 suffix (e.g., 14v2, 118v2) for V2 format')
    parser.add_argument('--data-root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--model', type=str, default='mpn',
                        choices=['mlp', 'mpn', 'gcn', 'skip_mpn', 'mask_embed_mpn',
                                'multi_mpn', 'mask_embed_multi_mpn', 'mask_embed_multi_mpn_nomp',
                                'multi_conv_net', 'mpn_simplenet'],
                        help='Model type')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--k', type=int, default=3, help='k for TAGConv')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience')
    parser.add_argument('--train-loss-fn', type=str, default='masked_l2',
                        choices=['masked_l2', 'power_imbalance', 'mixed_mse_power_imbalance'],
                        help='Training loss function')
    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU', 'Ascend'],
                        help='Device to use')

    return parser


def _load_training_datasets(args):
    """Load train/val/test datasets based on case format."""
    split = [0.7, 0.15, 0.15]
    is_v2 = args.case.endswith('v2')

    if is_v2:
        print(f"Using V2 dataset format for case '{args.case}'")
        train_ds = PowerFlowDataV2(root=args.data_root, case=args.case, split=split,
                                   task='train', normalize=True)
        xymean, xystd, edgemean, edgestd = train_ds.get_normalization_stats()
        val_ds = PowerFlowDataV2(root=args.data_root, case=args.case, split=split,
                                 task='val', normalize=True,
                                 xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)
        test_ds = PowerFlowDataV2(root=args.data_root, case=args.case, split=split,
                                  task='test', normalize=True,
                                  xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)
        train_loader = PowerFlowDataLoaderV2(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = PowerFlowDataLoaderV2(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = PowerFlowDataLoaderV2(test_ds, batch_size=args.batch_size, shuffle=False)
    else:
        print(f"Using legacy dataset format for case '{args.case}'")
        train_ds = PowerFlowData(root=args.data_root, case=args.case, split=split,
                                 task='train', normalize=True)
        xymean, xystd, edgemean, edgestd = train_ds.get_data_means_stds()
        val_ds = PowerFlowData(root=args.data_root, case=args.case, split=split,
                               task='val', normalize=True,
                               xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)
        test_ds = PowerFlowData(root=args.data_root, case=args.case, split=split,
                                task='test', normalize=True,
                                xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)
        train_loader = PowerFlowDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = PowerFlowDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = PowerFlowDataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("✓ Normalization stats computed from training set")
    print(f"✓ Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, (xymean, xystd, edgemean, edgestd)


def _warn_12d_models(model_name: str) -> None:
    """Warn about 12D model compatibility with 4D datasets."""
    models_12d = ['skip_mpn', 'mask_embed_mpn', 'multi_mpn',
                  'mask_embed_multi_mpn_nomp', 'multi_conv_net']
    if model_name in models_12d:
        print(f"\n⚠️  WARNING: Model '{model_name}' expects 12D input format:")
        print("   [one-hot(4) + features(4) + mask(4)]")
        print("   Current dataset provides 4D features only.\n")


def main():
    """Main training function."""
    parser = _setup_training_parser()
    args = parser.parse_args()

    # Initialize device
    init_device(args.device)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    if args.device == 'Ascend':
        try:
            ms.set_context(jit_config={"jit_level": "O0"})
            print("✓ JIT level set to O0 for Ascend compatibility")
        except Exception as e:
            print(f"⚠ Could not set JIT config: {e}")

    print("\n📊 Loading datasets...")
    train_loader, val_loader, test_loader, norm_stats = _load_training_datasets(args)

    _warn_12d_models(args.model)

    # Create model
    print(f"\n🧠 Creating {args.model.upper()} model...")
    model = create_model(args)

    num_params = sum(p.size for p in model.trainable_params())
    print(f"✓ Model has {num_params:,} trainable parameters")

    # Train
    trainer = Trainer(model, train_loader, val_loader, test_loader, args, norm_stats=norm_stats)
    trainer.train()

    print("✅ Training completed!")


if __name__ == '__main__':
    main()
