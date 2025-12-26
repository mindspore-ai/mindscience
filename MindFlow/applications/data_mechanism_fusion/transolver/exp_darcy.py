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
# ============================================================================
"""
Training and Evaluation script for Transolver (Darcy Flow 2D)
"""
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn, ops, context

from src.models.Transolver_Structured_Mesh_2D import Model as Transolver
from src.datasets.dataset import create_dataset, DarcyDataset
from src.utils.normalizer import GaussianNormalizer


def get_parser():
    """get parser"""
    parser = argparse.ArgumentParser(description='Training Transolver')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='Transolver_2D')
    parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
    parser.add_argument('--n-layers', type=int, default=3, help='layers')
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--gpu", type=str, default='0', help="GPU index")
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--unified_pos', type=int, default=0)
    parser.add_argument('--ref', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=32)
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--save_name', type=str, default='darcy_Transolver')
    parser.add_argument('--data-path', type=str, default='./piececonst_r421_N1024_smooth1.mat')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--subsampling', type=int, default=13)
    parser.add_argument('--device_target', type=str, default='Ascend')
    return parser


def train(model, train_loader, x_normalizer, y_normalizer, step_per_epoch, args):
    """train process"""
    print('Training...')
    loss_fn = nn.MSELoss()
    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=args.lr,
                                   weight_decay=args.weight_decay)

    def forward_fn(pos, x, label):
        x_enc = x_normalizer.encode(x)
        label_enc = y_normalizer.encode(label)
        logits = model(pos, x_enc)
        loss = loss_fn(logits, label_enc)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @ms.jit
    def train_step(pos, x, label):
        loss, grads = grad_fn(pos, x, label)
        if args.max_grad_norm is not None:
            grads = ops.clip_by_global_norm(grads, args.max_grad_norm)
        optimizer(grads)
        return loss

    model.set_train()
    for ep in range(args.epochs):
        t0 = time.time()
        train_loss = 0
        for pos, x, y in train_loader:
            loss = train_step(pos, x, y)
            train_loss += loss.asnumpy()

        train_loss = train_loss / step_per_epoch
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep + 1}/{args.epochs} Loss: {train_loss:.5f} Time: {time.time() - t0:.2f}s")

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    ms.save_checkpoint(model, os.path.join('./checkpoints', args.save_name + '.ckpt'))


def test(model, test_data_tuple, x_normalizer, y_normalizer, args):
    """test process"""
    print('Start Evaluation...')
    ckpt_path = os.path.join("./checkpoints", args.save_name + ".ckpt")
    ms.load_checkpoint(ckpt_path, model)
    model.set_train(False)

    pos, x, y = test_data_tuple
    x_enc = x_normalizer.encode(x)
    out = model(pos, x_enc)
    out = y_normalizer.decode(out)

    res = args.resolution
    y_np = y.asnumpy().reshape(res, res)
    out_np = out.asnumpy().reshape(res, res)

    rmse = np.sqrt(np.mean((y_np - out_np)**2))
    print(f"Validation RMSE: {rmse:.4e}")

    if not os.path.exists('./images'):
        os.makedirs('./images')

    # Optimization: 2-Column High-Res Visualization (GT vs Pred)
    # Using interpolation='bicubic' for smoother gradients
    plt.figure(figsize=(10, 5), dpi=300)

    # Subplot 1: Ground Truth
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(y_np, cmap='jet', origin='lower', interpolation='bicubic')
    plt.colorbar()
    plt.axis('off')

    # Subplot 2: Prediction
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(out_np, cmap='jet', origin='lower', interpolation='bicubic')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('./images', "result_darcy_hd.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print("Visualization saved to ./images/result_darcy_hd.png")


def main():
    """main function"""
    parser = get_parser()
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        device_id=int(args.gpu))

    raw_dataset = DarcyDataset(args.data_path, ntrain=args.ntrain,
                               subsampling=args.subsampling, resolution=args.resolution)

    raw_x = ms.Tensor(raw_dataset.coeff.astype(np.float32), ms.float32)
    raw_y = ms.Tensor(raw_dataset.solution.astype(np.float32), ms.float32)

    x_normalizer = GaussianNormalizer(raw_x)
    y_normalizer = GaussianNormalizer(raw_y)

    train_loader = create_dataset(args.data_path,
                                  batch_size=args.batch_size,
                                  ntrain=args.ntrain,
                                  subsampling=args.subsampling,
                                  resolution=args.resolution,
                                  shuffle=True)

    test_idx = 0
    test_pos = ms.Tensor(raw_dataset.pos[test_idx:test_idx+1].astype(np.float32), ms.float32)
    test_x = ms.Tensor(raw_dataset.coeff[test_idx:test_idx+1].astype(np.float32), ms.float32)
    test_y = ms.Tensor(raw_dataset.solution[test_idx:test_idx+1].astype(np.float32), ms.float32)

    model = Transolver(space_dim=2,
                       n_layers=args.n_layers,
                       n_hidden=args.n_hidden,
                       n_head=args.n_heads,
                       slice_num=args.slice_num,
                       fun_dim=1,
                       out_dim=1,
                       H=args.resolution,
                       W=args.resolution,
                       unified_pos=bool(args.unified_pos),
                       ref=args.ref,
                       mlp_ratio=args.mlp_ratio,
                       dropout=args.dropout)

    if args.eval:
        test(model, (test_pos, test_x, test_y), x_normalizer, y_normalizer, args)
    else:
        train(model, train_loader, x_normalizer, y_normalizer, len(train_loader), args)
        test(model, (test_pos, test_x, test_y), x_normalizer, y_normalizer, args)


if __name__ == "__main__":
    main()
    