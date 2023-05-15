"""Train"""
import argparse
import time
import pickle
import stat
import os.path
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from src.utils import LossSmoothed, loss_nll, featurize, LRLIST
from src.model import ProteinMPNN
from src.datasets import StructureDatasetPDB, Definebatch


def main(args):
    """MAIN"""

    class CustomTrainOneStepCell(nn.Cell):
        """自定义训练网络"""

        def __init__(self, network_, optimizer_):
            """入参有两个：训练网络，优化器"""
            super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
            self.network = network_  # 定义前向网络
            self.network.set_grad()  # 构建反向网络
            self.optimizer_ = optimizer_  # 定义优化器
            self.weights = self.optimizer_.parameters  # 待更新参数
            self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度
            self.time = time.time()

        def construct(self, *inputs):
            x, s, mask, chain_m, residue_idx, chain_encoding_all, mask_for_loss = inputs
            input_ = x, s, mask, chain_m, residue_idx, chain_encoding_all
            loss_ = self.network(*inputs)
            grads = self.grad(self.network, self.weights)(*input_, mask_for_loss)
            self.optimizer_(grads)
            return loss_

    class CustomWithLossCell(nn.Cell):
        """前向传播"""

        def __init__(self, backbone, loss_fn):
            """参数有两个"""
            super(CustomWithLossCell, self).__init__(auto_prefix=False)
            self.backbone = backbone
            self.loss_fn = loss_fn

        def construct(self, x, s, mask, chain_m, residue_idx, chain_encoding_all, mask_for_loss):
            output = self.backbone(x, s, mask, chain_m, residue_idx, chain_encoding_all)
            loss_av = self.loss_fn(s, output, mask_for_loss)
            return loss_av

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    path = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not path:
        with os.fdopen(os.open(logfile, os.O_RDWR | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR), 'w') \
                    as f:
            f.write('Epoch\tTrain\tValidation\n')


    model = ProteinMPNN(num_letters=21,
                        node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        augment_eps=args.backbone_noise)
    loss = LossSmoothed()
    net_with_loss = CustomWithLossCell(model, loss)
    if path:
        checkpoint = ms.load_checkpoint(path)
        total_step = checkpoint['step']  # write total_step from the checkpoint
        epoch = checkpoint['epoch']  # write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0
    lrlist = LRLIST(args.hidden_dim, 2, 4000)
    lr = lrlist.cal_lr(args.num_epochs * 5436)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=ms.Tensor(lr), beta1=0.9, beta2=0.98, eps=1e-9)
    if path:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
    with open(args.path_for_pkl, 'rb') as f_read:
        pdb_dict_train = pickle.load(f_read)

    dataset_train = StructureDatasetPDB(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
    loader_train = Definebatch(dataset_train, batch_size=args.batch_size)

    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        train_net.set_train()
        train_sum, train_weights = 0., 0.
        train_acc = 0.
        for _, batch in enumerate(loader_train):
            x, s, mask, chain_m, residue_idx, chain_encoding_all = featurize(batch)
            mask_for_loss = mask * chain_m
            train_net(x, s, mask, chain_m, residue_idx, chain_encoding_all, mask_for_loss)
            total_step += 1

        log_probs = train_net.network.backbone(x, s, mask, chain_m, residue_idx, chain_encoding_all)
        loss, _, true_false = loss_nll(s, log_probs, mask_for_loss)
        train_sum += ops.ReduceSum()(loss * mask_for_loss).asnumpy()
        train_acc += ops.ReduceSum()(true_false * mask_for_loss).asnumpy()
        train_weights += ops.ReduceSum()(mask_for_loss).asnumpy()
        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
        with os.fdopen(os.open(logfile, os.O_RDWR | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR), 'w') \
                    as f:
            f.write(
                f'epoch: {e + 1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, \
                train_acc: {train_accuracy_}\n')
        print(
            f'epoch: {e + 1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, train_acc: {train_accuracy_}')

        checkpoint_filename_last = base_folder + 'model_weights/epoch_last.ckpt'
        if (e + 1) % args.save_model_every_n_epochs == 0:
            ms.save_checkpoint(model, checkpoint_filename_last)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str,
                           default="pdb_2021aug02_sample",
                           help="path for loading training data")
    argparser.add_argument("--path_for_outputs", type=str,
                           default="./exp_020/",
                           help="path for logs and model weights")
    argparser.add_argument("--path_for_pkl", type=str,
                           default="pdb_dict_train.pkl",
                           help="path for loading pkl")
    argparser.add_argument("--previous_checkpoint", type=str, default="",
                           help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=5,
                           help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2,
                           help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000,
                           help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=1000,  # 10000
                           help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2,
                           help="amount of noise added to backbone during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0,
                           help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=False, help="train with mixed precision")
    argparser.add_argument('--device_id', help='device id', type=int, default=0)
    argparser.add_argument('--device_target', help='device target', type=str, default="Ascend")

    args_ = argparser.parse_args()
    ms.set_context(device_target=args_.device_target, device_id=args_.device_id, mode=ms.GRAPH_MODE)
    main(args_)
