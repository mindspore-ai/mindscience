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
# ==============================================================================
"exp_elas"
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from tqdm import tqdm

from src.datasets.dataset import RandomAccessDataset
from src.utils.testloss import TestLoss
from src.models.model_dict import get_model
from src.utils.normalizer import UnitTransformer


def get_parser():
    """get parser"""
    parser = argparse.ArgumentParser('Training Transolver')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='TransolverIrregular')
    parser.add_argument('--n-hidden', type=int, default=128, help='hidden dim')
    parser.add_argument('--n-layers', type=int, default=8, help='layers')
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--downsample', type=int, default=5)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--unified_pos', type=int, default=0)
    parser.add_argument('--ref', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=64)
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--save_name', type=str, default='elas_Transolver')
    parser.add_argument('--data_path', type=str, default='./data')
    return parser


def count_parameters(model):
    """count"""
    total_params = sum(p.size for p in model.trainable_params())
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train(args, model, train_loader, test_loader, y_normalizer, myloss, train_step, step_per_epoch,
          ntrain, ntest):
    """train"""
    print('Training...')
    for ep in range(args.epochs):

        model.set_train()
        train_loss = 0

        b_i, total_steps = 0, step_per_epoch
        for pos, _, y in (pbar := tqdm(train_loader.create_tuple_iterator(), total=total_steps)):
            loss = train_step(pos, y)
            train_loss += loss.asnumpy()
            if (b_i + 1) % 5 == 0 or (b_i + 1) == total_steps:
                pbar.set_description(f'Epoch {ep+1} Iter {b_i+1}/{total_steps} Loss {loss.asnumpy():.5f}')
            b_i += 1

        train_loss = train_loss / ntrain
        print(f"Epoch {ep + 1} Train loss : {train_loss:.5f}")

        model.set_train(False)
        rel_err = 0.0
        for pos, _, y in tqdm(test_loader):
            out = model(pos, None).squeeze(-1)
            out = y_normalizer.decode(out)
            tl = myloss(out, y).asnumpy()
            rel_err += tl

        rel_err /= ntest
        print(f"rel_err : {rel_err}")

        if (ep + 1) == 1 or (ep + 1)% 5 == 0:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            print('save model')
            ms.save_checkpoint(model, os.path.join('./checkpoints', args.save_name + '.ckpt'))

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    print('save model')
    ms.save_checkpoint(model, os.path.join('./checkpoints', args.save_name + '.ckpt'))


def test(args, model, test_loader, y_normalizer, myloss, ntest):
    """test"""
    ms.load_param_into_net(model, ms.load_checkpoint("./checkpoints/" + args.save_name + ".ckpt"))
    model.set_train(False)
    if not os.path.exists('./results/' + args.save_name + '/'):
        os.makedirs('./results/' + args.save_name + '/')
    rel_err = 0.0
    showcase = 2
    cnt = 0

    for pos, fx, y in test_loader.create_tuple_iterator():
        cnt += 1
        out = model(pos, None).squeeze(-1)
        out = y_normalizer.decode(out)
        tl = myloss(out, y).asnumpy()
        rel_err += tl
        if cnt < showcase:
            print(cnt)
            plt.axis('off')
            plt.scatter(x=fx[0, :, 0], y=fx[0, :, 1],
                        c=y[0, :], cmap='coolwarm')
            plt.colorbar()
            plt.clim(0, 1000)
            plt.savefig(
                os.path.join('./results/' + args.save_name + '/',
                                "gt_" + str(cnt) + ".pdf"), bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.axis('off')
            plt.scatter(x=fx[0, :, 0], y=fx[0, :, 1],
                        c=out[0, :], cmap='coolwarm')
            plt.colorbar()
            plt.clim(0, 1000)
            plt.savefig(
                os.path.join('./results/' + args.save_name + '/',
                                "pred_" + str(cnt) + ".pdf"), bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.axis('off')
            plt.scatter(x=fx[0, :, 0], y=fx[0, :, 1],
                        c=((y[0, :] - out[0, :])), cmap='coolwarm')
            plt.clim(-8, 8)
            plt.colorbar()
            plt.savefig(
                os.path.join('./results/' + args.save_name + '/',
                                "error_" + str(cnt) + ".pdf"), bbox_inches='tight', pad_inches=0)
            plt.close()

    rel_err /= ntest
    print(f"rel_err : {rel_err}")


def main():
    """main"""
    parser = get_parser()
    args = parser.parse_args()
    ntrain = args.ntrain
    ntest = 200
    # ms.set_context(mode=ms.PYNATIVE_MODE)
    # ms.set_device(device_target='CPU')

    path_sigma = args.data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
    path_xy = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'

    input_s = np.load(path_sigma).astype(np.float32)
    input_s = np.transpose(input_s, (1, 0))
    input_xy = np.load(path_xy).astype(np.float32)
    input_xy = np.transpose(input_xy, (2, 0, 1))

    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]

    print(input_s.shape, input_xy.shape)

    y_normalizer = UnitTransformer(train_s)
    train_s = y_normalizer.encode(train_s)

    train_dataset = RandomAccessDataset(train_xy, train_xy, train_s)
    test_dataset = RandomAccessDataset(test_xy, test_xy, test_s)
    train_loader = GeneratorDataset(source=train_dataset, column_names=['x', 'fx', 'y'],
                                    shuffle=True).batch(args.batch_size)
    test_loader = GeneratorDataset(source=test_dataset, column_names=['x', 'fx', 'y'],
                                   shuffle=False).batch(args.batch_size)

    print("Dataloading is over.")

    model = get_model(args.model)(space_dim=2,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  time_input=False,
                                  mlp_ratio=args.mlp_ratio,
                                  fun_dim=0,
                                  out_dim=1,
                                  slice_num=args.slice_num,
                                  ref=args.ref,
                                  unified_pos=args.unified_pos)
    step_per_epoch = len(train_loader)
    cosine_decay_lr = nn.cosine_decay_lr(min_lr=0., max_lr=args.lr,
                                         total_step=args.epochs * step_per_epoch,
                                         step_per_epoch=step_per_epoch, decay_epoch=args.epochs)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=cosine_decay_lr,
                                   weight_decay=args.weight_decay)

    print(args)
    print(model)
    count_parameters(model)

    myloss = TestLoss(size_average=False)

    def forward_fn(x, y):
        """forward fn"""
        out = model(x, None).squeeze(-1)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        loss = myloss(out, y)
        return loss, y

    grad_fn = ops.value_and_grad(forward_fn, None, model.trainable_params(), has_aux=True)

    def train_step(x, y):
        """train _step"""
        (loss, _), grads = grad_fn(x, y)
        grads = ops.clip_by_global_norm(grads, clip_norm=args.max_grad_norm)
        optimizer(grads)
        return loss

    if args.eval:
        test(
            args,
            model=model,
            test_loader=test_loader,
            y_normalizer=y_normalizer,
            myloss=myloss,
            ntest=ntest
        )
    else:
        train(
            args,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            y_normalizer=y_normalizer,
            myloss=myloss,
            train_step=train_step,
            step_per_epoch=step_per_epoch,
            ntrain=ntrain,
            ntest=ntest
        )

if __name__ == "__main__":
    main()
