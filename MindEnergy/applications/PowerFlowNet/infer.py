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
Evaluation Script for MindSpore PowerFlowNet

Tests trained models and computes detailed loss metrics.

Usage:
    python test.py --model mlp --run_id <run_id> --loss masked_l2
    python test.py --model mpn --run_id <run_id> --loss masked_l2v2
    python test.py --model skip_mpn --run_id <run_id> --loss masked_l1
"""

import traceback
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import mindspore as ms
from mindspore import nn, context
from src import (
    MPN, SkipMPN, MaskEmbedMPN, MultiMPN, MaskEmbedMultiMPN,
    MaskEmbedMultiMPNNoMP, MultiConvNet, MPNSimplenet, MLPNet, GCNNet
)
from src.power_flow_data import PowerFlowData, PowerFlowDataV2, PowerFlowDataLoader, PowerFlowDataLoaderV2
from src.evaluation import num_params
from src.custom_loss_functions import (
    MaskedL2Loss, MaskedL2V2, MaskedL1,
    PowerImbalance, MixedMSEPowerImbalance
)
from tqdm import tqdm

# Set default context to PYNATIVE_MODE for better compatibility with complex operations
context.set_context(mode=context.PYNATIVE_MODE)
ms.set_device(device_target='CPU')

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

def evaluate_simple(model, loader, loss_fn):
    """
    Simple evaluation matching train_v2.py logic exactly.
    Uses batch-average loss (same as training).
    """
    model.set_train(False)
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc='Evaluating')
    for batch in pbar:
        pred = model(batch)
        loss = loss_fn(pred, batch.y, batch.pred_mask)
        total_loss += float(loss.asnumpy())
        num_batches += 1
        pbar.set_postfix({'loss': f'{total_loss / num_batches:.6f}'})

    return total_loss / max(1, num_batches)


def _get_gnn_model_params(config: Dict) -> Dict:
    """Extract GNN-specific model parameters from config."""
    return {
        'nfeature_dim': config['nfeature_dim'],
        'efeature_dim': config.get('efeature_dim', 2),
        'output_dim': config['output_dim'],
        'hidden_dim': config['hidden_dim'],
        'n_gnn_layers': config['n_layers'],
        'k': config.get('k', 3),
        'dropout_rate': config.get('dropout_rate', 0.0),
    }


def _get_mlp_params(config: Dict) -> Dict:
    """Extract MLP model parameters from config."""
    return {
        'nfeature_dim': config['nfeature_dim'],
        'output_dim': config['output_dim'],
        'hidden_dim': config['hidden_dim'],
        'n_layers': config['n_layers'],
        'dropout_rate': config.get('dropout_rate', 0.0),
    }


def _get_gcn_params(config: Dict) -> Dict:
    """Extract GCN model parameters from config."""
    return {
        'nfeature_dim': config['nfeature_dim'],
        'output_dim': config['output_dim'],
        'hidden_dim': config['hidden_dim'],
        'n_gnn_layers': config['n_layers'],
        'dropout_rate': config.get('dropout_rate', 0.0),
    }


def create_model(model_name: str, config: Dict) -> nn.Cell:
    """
    Create model instance based on name.
    
    Args:
        model_name: Model type identifier
        config: Configuration dictionary with model parameters
    
    Returns:
        Instantiated MindSpore model
    """
    model_name_lower = model_name.lower()

    # MLP model
    if model_name_lower == 'mlp':
        return MLPNet(**_get_mlp_params(config))

    # GCN model uses n_gnn_layers
    if model_name_lower == 'gcn':
        return GCNNet(**_get_gcn_params(config))

    # GNN models using n_gnn_layers parameter
    gnn_params = _get_gnn_model_params(config)

    gnn_models = {
        'mpn': MPN,
        'skip_mpn': SkipMPN,
        'mask_embed_mpn': MaskEmbedMPN,
        'multi_mpn': MultiMPN,
        'mask_embed_multi_mpn': MaskEmbedMultiMPN,
        'mask_embed_multi_mpn_nomp': MaskEmbedMultiMPNNoMP,
        'multi_conv_net': MultiConvNet,
        'mpn_simplenet': MPNSimplenet,
    }

    if model_name_lower in gnn_models:
        return gnn_models[model_name_lower](**gnn_params)

    raise ValueError(f"Unknown model: {model_name}")


def load_loss_function(loss_name: str, config: Optional[Dict] = None):
    """
    Load loss function by name.
    
    Args:
        loss_name: Loss function identifier
        config: Optional configuration for loss function
    
    Returns:
        Instantiated loss function
    """
    loss_name = loss_name.lower()

    if loss_name == 'masked_l2':
        return MaskedL2Loss()

    if loss_name == 'masked_l2v2':
        return MaskedL2V2()

    if loss_name == 'masked_l1':
        return MaskedL1()

    if loss_name == 'power_imbalance':
        if config is None:
            raise ValueError("PowerImbalance requires config with normalization parameters")
        return PowerImbalance(
            xymean=config.get('xymean'),
            xystd=config.get('xystd'),
            edgemean=config.get('edgemean'),
            edgestd=config.get('edgestd')
        )

    if loss_name == 'mixed_mse_power_imbalance':
        if config is None:
            raise ValueError("MixedMSEPowerImbalance requires config with normalization parameters")
        return MixedMSEPowerImbalance(
            xymean=config.get('xymean'),
            xystd=config.get('xystd'),
            edgemean=config.get('edgemean'),
            edgestd=config.get('edgestd'),
            alpha=config.get('alpha', 0.9)
        )

    raise ValueError(f"Unknown loss function: {loss_name}")


def _setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return argument parser for test script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PowerFlowNet models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --model mpn --run_id exp_001 --loss masked_l2
  python test.py --model mask_embed_mpn --run_id exp_002 --loss masked_l2v2
  python test.py --model multi_mpn --run_id exp_003 --loss masked_l1
        """
    )

    parser.add_argument('--model', type=str, default='mpn',
                        choices=['mlp', 'gcn', 'mpn', 'skip_mpn', 'mask_embed_mpn',
                                'multi_mpn', 'mask_embed_multi_mpn', 'mask_embed_multi_mpn_nomp',
                                'multi_conv_net', 'mpn_simplenet'],
                        help='Model architecture to evaluate')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Run ID of the saved model checkpoint')
    parser.add_argument('--loss', type=str, default='masked_l2',
                        choices=['masked_l2', 'masked_l2v2', 'masked_l1',
                            'power_imbalance', 'mixed_mse_power_imbalance'],
                        help='Loss function for evaluation (default: masked_l2)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data files')
    parser.add_argument('--case', type=str, default='14',
                        help='Case name. Use v2 suffix (e.g., 14v2, 118v2) for V2 format, '
                            'otherwise legacy format (e.g., 14, 118)')
    parser.add_argument('--device', type=str, default='CPU',
                        choices=['CPU', 'GPU', 'Ascend'],
                        help='Computation device')
    parser.add_argument('--batch_size', type=int, default=64,
                            help='Batch size for evaluation')

    return parser


def _print_header(args) -> None:
    """Print evaluation header."""
    print("\n{'='*60}")
    print("PowerFlowNet MindSpore Evaluation")
    print(f"{'='*60}")
    print(f"Model:     {args.model}")
    print(f"Run ID:    {args.run_id}")
    print(f"Case:      {args.case}")
    print(f"Loss:      {args.loss}")
    print(f"Device:    {args.device}")
    print(f"{'='*60}\n")


def _load_datasets(args):
    """Load train, val, test datasets with automatic format detection.
    
    V2 format: case names ending with 'v2' (e.g., 14v2, 118v2)
    Legacy format: case names without 'v2' (e.g., 14, 118)
    """
    split = [0.7, 0.15, 0.15]
    is_v2_format = args.case.endswith('v2')

    if is_v2_format:
        print(f"Loading V2 dataset format for case '{args.case}'")
        train_dataset = PowerFlowDataV2(
            root=args.data_dir, case=args.case, split=split,
            task='train', normalize=True)
        xymean, xystd, edgemean, edgestd = train_dataset.get_normalization_stats()

        val_dataset = PowerFlowDataV2(
            root=args.data_dir, case=args.case, split=split,
            task='val', normalize=True,
            xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)

        test_dataset = PowerFlowDataV2(
            root=args.data_dir, case=args.case, split=split,
            task='test', normalize=True,
            xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)

        train_loader = PowerFlowDataLoaderV2(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = PowerFlowDataLoaderV2(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = PowerFlowDataLoaderV2(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        print(f"Loading legacy dataset format for case '{args.case}'")
        train_dataset = PowerFlowData(
            root=args.data_dir, case=args.case, split=split,
            task='train', normalize=True)
        xymean, xystd, edgemean, edgestd = train_dataset.get_data_means_stds()

        val_dataset = PowerFlowData(
            root=args.data_dir, case=args.case, split=split,
            task='val', normalize=True,
            xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)

        test_dataset = PowerFlowData(
            root=args.data_dir, case=args.case, split=split,
            task='test', normalize=True,
            xymean=xymean, xystd=xystd, edgemean=edgemean, edgestd=edgestd)

        train_loader = PowerFlowDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = PowerFlowDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = PowerFlowDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, xymean, xystd, edgemean, edgestd


def _load_model_checkpoint(args, model):
    """Load model checkpoint from disk."""
    model_dir = Path('models') / args.case / f"{args.model}_{args.run_id}"

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        case_dir = Path('models') / args.case
        if case_dir.exists():
            print(f"Available directories in models/{args.case}/:")
            for d in sorted(case_dir.iterdir()):
                print(f"  {d.name}")
        return False

    checkpoints = sorted(model_dir.glob('model_epoch_*.ckpt'))
    if not checkpoints:
        print(f"Error: No checkpoints found in {model_dir}")
        return False

    checkpoint_path = checkpoints[-1]
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        param_dict = ms.load_checkpoint(str(checkpoint_path))
        ms.load_param_into_net(model, param_dict)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False


def _print_evaluation_results(train_loss, val_loss, test_loss) -> None:
    """Print evaluation results summary."""
    print(f"\n{'='*60}")
    print("Evaluation Summary:")
    print(f"{'='*60}")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss:   {val_loss:.6f}")
    print(f"  Test Loss:  {test_loss:.6f}")
    print(f"{'='*60}\n")


def main():
    """Main evaluation function."""
    parser = _setup_argument_parser()
    args = parser.parse_args()

    context.set_context(mode=context.PYNATIVE_MODE)
    ms.set_device(device_target=args.device)

    _print_header(args)

    # Load dataset
    print("Loading dataset...")
    try:
        train_loader, val_loader, test_loader, xymean, xystd, edgemean, edgestd = _load_datasets(args)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        print(f"Make sure data exists at {args.data_dir}/mindspore/raw/case{args.case}_*.npy")
        return

    # Create and load model
    config = {
        'nfeature_dim': 4, 'efeature_dim': 2, 'output_dim': 4,
        'hidden_dim': 64, 'n_layers': 3, 'k': 3, 'dropout_rate': 0.0,
    }

    print("Creating model...")
    model = create_model(args.model, config)

    if not _load_model_checkpoint(args, model):
        return

    print(f"Model parameters: {num_params(model):,}\n")

    # Load loss function
    loss_config = {
        'xymean': xymean, 'xystd': xystd,
        'edgemean': edgemean, 'edgestd': edgestd,
    }
    loss_fn = load_loss_function(args.loss, loss_config)

    # Evaluate
    print("Evaluating on Training Set:")
    print("-" * 60)
    train_loss = evaluate_simple(model, train_loader, loss_fn)
    print(f"  Loss: {train_loss:.6f}")

    print("\nEvaluating on Validation Set:")
    print("-" * 60)
    val_loss = evaluate_simple(model, val_loader, loss_fn)
    print(f"  Loss: {val_loss:.6f}")

    print("\nEvaluating on Test Set:")
    print("-" * 60)
    test_loss = evaluate_simple(model, test_loader, loss_fn)
    print(f"  Loss: {test_loss:.6f}")

    _print_evaluation_results(train_loss, val_loss, test_loss)


if __name__ == '__main__':
    main()
