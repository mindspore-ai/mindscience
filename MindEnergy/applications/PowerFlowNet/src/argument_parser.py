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
# Significant modifications have been made to adapt the code for the MindSpore framework.
# ============================================================================
"""
Argument parser for MindSpore PowerFlowNet.

This module implements a unified argument parser that supports both JSON configuration
files and command-line arguments. It has been adapted from the original PyTorch version
to work seamlessly with MindSpore framework while maintaining API compatibility.

Key features:
- JSON configuration file support for reproducibility
- Command-line argument overrides for flexibility
- Framework-agnostic parameter definitions
- Device and training hyperparameter management
- Model architecture configuration

The parser is designed to work with both local training and distributed Ascend
device execution, providing a single configuration interface for all use cases.
"""

import argparse
import os
import json


def argument_parser():
    """Parse command-line and config file arguments

    Supports:
    1. JSON config file (--cfg_json)
    2. Command-line arguments (override JSON values)

    Returns:
        argparse.Namespace: Parsed arguments
    """
    # Config file parser (doesn't print help by itself)
    config_parser = argparse.ArgumentParser(
        prog='PowerFlowNet',
        description='parse json configs',
        add_help=False
    )
    config_parser.add_argument(
        '--cfg_json', '--config', '--configs',
        default='configs/standard.json',
        type=str,
        help='Path to JSON configuration file'
    )

    # Main parser
    parser = argparse.ArgumentParser(
        prog='PowerFlowNet',
        description='Train MindSpore neural network for power flow approximation',
        parents=[config_parser]
    )

    # Network Parameters
    parser.add_argument('--nfeature_dim', type=int, default=6,
                       help='Number of node features')
    parser.add_argument('--efeature_dim', type=int, default=2,
                       help='Number of edge features')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Number of hidden features')
    parser.add_argument('--output_dim', type=int, default=6,
                       help='Number of output features')
    parser.add_argument('--n_gnn_layers', type=int, default=4,
                       help='Number of GNN layers')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of conv filter taps')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--model', type=str, default='MPN',
                       choices=['MLPNet', 'MPN', 'GCNNet', 'SkipMPN', 'MaskEmbdMPN',
                               'MultiMPN', 'MaskEmbdMultiMPN', 'MPNSimplenet', 'MultiConvNet'],
                       help='Model architecture')
    parser.add_argument('--regularize', type=bool, default=True,
                       help='Include regularization in loss function')
    parser.add_argument('--regularization_coeff', type=float, default=1.0,
                       help='Regularization coefficient')

    # Training parameters
    parser.add_argument('--data_dir', '--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--disable_normalize', default=False, action='store_true',
                       help='Disable normalizing data')
    parser.add_argument('--train_loss_fn',
                       type=str, default='masked_l2',
                       choices=['masked_l2', 'power_imbalance', 'mixed_mse_power_imbalance', 'mse'],
                       help='Training loss function')
    parser.add_argument('--num_epochs', '--num-epochs', type=int, default=100,
                       help='Number of epochs to train for')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--case', type=str, default='14',
                       help='Grid case (14, 118, 14v2, 118v2)')
    parser.add_argument('--save', default=True, action='store_true',
                       help='Save model checkpoint')

    # Device parameters
    parser.add_argument('--device', type=str, default='CPU',
                       choices=['CPU', 'GPU', 'NPU'],
                       help='Device type for training')
    parser.add_argument('--device_id', type=int, default=0,
                       help='Device ID')

    # Misc parameters
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility')

    # Step 0: Parse config file if specified
    args, left_argv = config_parser.parse_known_args()

    if args.cfg_json is not None and os.path.exists(args.cfg_json):
        with open(args.cfg_json, encoding='utf-8') as f:
            json_dict = json.load(f)

        # Convert JSON dict to command-line format
        json_argv = []
        for key, value in json_dict.items():
            json_argv.append('--' + key)
            json_argv.append(str(value))

        # Parse JSON arguments into args namespace
        parser.parse_known_args(json_argv, args)

    # Step 1: Parse command-line arguments and override JSON values
    parser.parse_args(left_argv, args)

    return args
