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
    """get parser for training and evaluation"""
    parser = argparse.ArgumentParser(description="Transolver Darcy 2D Training")

    # Execution context parameters
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"], help="context mode")
    parser.add_argument("--device_target", type=str, default="Ascend", help="target device")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--data_path", type=str, default="./piececonst_r421_N1024_smooth1.mat", help="data path")
    parser.add_argument("--save_name", type=str, default="transolver_darcy", help="checkpoint save name")

    # Model Architecture Hyperparameters
    parser.add_argument("--n_hidden", type=int, default=128, help="hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--n_head", type=int, default=8, help="number of heads")
    parser.add_argument("--unified_pos", type=int, default=1, help="whether to use unified position")
    parser.add_argument("--ref", type=int, default=8, help="reference dimension")
    parser.add_argument("--slice_num", type=int, default=32, help="number of slices")
    parser.add_argument("--mlp_ratio", type=int, default=1, help="MLP expansion ratio")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Training Strategy Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max gradient norm for clipping")
    parser.add_argument("--eval", type=int, default=0, help="evaluation mode")

    # Data Resolution and Sampling
    parser.add_argument("--ntrain", type=int, default=1000, help="number of training samples")
    parser.add_argument("--resolution", type=int, default=32, help="data resolution")
    parser.add_argument("--subsampling", type=int, default=13, help="subsampling rate")

    return parser


def train(model, train_loader, x_normalizer, y_normalizer, steps_per_epoch, args):
    """train process"""
    print(f"[INFO] Start Training: {args.save_name}")

    loss_fn = nn.MSELoss()
    # Use AdamWeightDecay to support weight_decay parameter
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
        # Apply gradient clipping if max_grad_norm > 0
        if args.max_grad_norm > 0:
            grads = ops.clip_by_global_norm(grads, args.max_grad_norm)
        optimizer(grads)
        return loss

    model.set_train(True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss_meter = 0.0

        for pos, x, y in train_loader:
            loss = train_step(pos, x, y)
            loss_meter += loss.asnumpy()

        epoch_time = time.time() - t0
        avg_loss = loss_meter / steps_per_epoch
        # Print logs every 10 epochs or at the first epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    save_path = os.path.join("./checkpoints", args.save_name + ".ckpt")
    ms.save_checkpoint(model, save_path)
    print(f"[INFO] Model saved to {save_path}")


def test(model, test_data_tuple, x_normalizer, y_normalizer, args):
    """test process"""
    print("[INFO] Start Testing & Visualization...")
    ckpt_path = os.path.join("./checkpoints", args.save_name + ".ckpt")
    ms.load_checkpoint(ckpt_path, model)
    model.set_train(False)

    pos_tensor, x_tensor, y_tensor = test_data_tuple
    x_enc = x_normalizer.encode(x_tensor)

    start_time = time.time()
    pred_enc = model(pos_tensor, x_enc)
    print(f"[INFO] Inference time: {(time.time() - start_time)*1000:.2f} ms")

    pred_phys = y_normalizer.decode(pred_enc)

    res = args.resolution
    label_np = y_tensor.asnumpy().reshape(res, res)
    pred_np = pred_phys.asnumpy().reshape(res, res)

    rmse = np.sqrt(np.mean((pred_np - label_np)**2))
    rel_l2 = np.linalg.norm(pred_np - label_np) / np.linalg.norm(label_np)
    print(f"[RESULT] Test RMSE: {rmse:.4e}")
    print(f"[RESULT] Relative L2: {rel_l2:.2%}")

    if not os.path.exists("./images"):
        os.makedirs("./images")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Label (Ground Truth)")
    plt.imshow(label_np, cmap='jet', origin='lower')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title(f"Prediction (Rel L2: {rel_l2:.1%})")
    plt.imshow(pred_np, cmap='jet', origin='lower')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title(f"Abs Error (RMSE: {rmse:.1e})")
    plt.imshow(np.abs(label_np - pred_np), cmap='jet', origin='lower')
    plt.colorbar()

    save_img_path = "./images/result_darcy_hd.png"
    plt.savefig(save_img_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[INFO] Visualization saved to {save_img_path}")


def main():
    """main function for script entry"""
    parser = get_parser()
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode == "GRAPH" else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    # Use args.ntrain to load dataset instead of hardcoded values
    raw_dataset = DarcyDataset(args.data_path, ntrain=args.ntrain,
                               subsampling=args.subsampling, resolution=args.resolution)

    # Use float32 to avoid data type mismatch
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
    steps_per_epoch = train_loader.get_dataset_size()

    test_idx = 0
    test_pos = ms.Tensor(raw_dataset.pos[test_idx:test_idx+1].astype(np.float32), ms.float32)
    test_x = ms.Tensor(raw_dataset.coeff[test_idx:test_idx+1].astype(np.float32), ms.float32)
    test_y = ms.Tensor(raw_dataset.solution[test_idx:test_idx+1].astype(np.float32), ms.float32)
    test_data_tuple = (test_pos, test_x, test_y)

    model = Transolver(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        slice_num=args.slice_num,
        fun_dim=1,
        out_dim=1,
        H=args.resolution,
        W=args.resolution,
        unified_pos=bool(args.unified_pos),
        ref=args.ref,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout
    )

    if args.eval:
        test(model, test_data_tuple, x_normalizer, y_normalizer, args)
    else:
        train(model, train_loader, x_normalizer, y_normalizer, steps_per_epoch, args)
        test(model, test_data_tuple, x_normalizer, y_normalizer, args)


if __name__ == "__main__":
    main()
