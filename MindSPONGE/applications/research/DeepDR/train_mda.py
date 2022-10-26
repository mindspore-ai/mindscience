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
"""Training of MDA"""
import os.path as Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import scipy.io as sio
import numpy as np
from model import MDA
from mindspore import nn, ops
import mindspore.dataset as ds
import mindspore as ms


class MDAWithLossCell(nn.Cell):
    """MDAWithLossCell"""
    def __init__(self, backbone, loss_fn):
        super(MDAWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    @property
    def backbone_network(self):
        return self._backbone

    def construct(self, data, label):
        _, out = self._backbone(data)
        return self._loss_fn(out, label)


class MDACustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(MDACustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        self.network.set_grad()  # 构建反向网络
        self.optimizer = optimizer  # 定义优化器
        self.weights = self.optimizer.parameters  # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss = self.network(*inputs)  # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        self.optimizer(grads)  # 使用优化器更新权重参数
        return loss


class MDALoss(nn.Cell):
    """MDALoss"""
    def __init__(self):
        """入参有两个：训练网络，优化器"""
        super(MDALoss, self).__init__()
        self.loss = ops.BinaryCrossEntropy()

    def construct(self, logits, labels):
        labels = labels.view((logits.shape[0], 9, -1))
        output = self.loss(logits, labels, None)
        return output


def build_mda_model(input_dims, encoding_dims):
    net = MDA(input_dims=input_dims, encoding_dims=encoding_dims)
    sgd = nn.SGD(params=net.trainable_params(), learning_rate=0.005, momentum=0.9, weight_decay=0.0, nesterov=False)
    loss = MDALoss()
    net_with_loss = MDAWithLossCell(net, loss_fn=loss)
    train_net = MDACustomTrainOneStepCell(net_with_loss, sgd)
    return train_net


def build_dataset(x, nf=0.5, std=1.0):
    """"build_dataset"""
    noise_factor = nf
    if isinstance(x, list):
        xs = train_test_split(*x, test_size=0.2)
        x_train = []
        x_test = []
        for jj in range(0, len(xs), 2):
            x_train.append(xs[jj])
            x_test.append(xs[jj + 1])
            x_train_noisy = list(x_train)
            x_test_noisy = list(x_test)
        for ii, _ in enumerate(x_train):
            x_train_noisy[ii] = x_train_noisy[ii] + noise_factor * np.random.normal(loc=0.0, scale=std,
                                                                                    size=x_train[ii].shape)
            x_test_noisy[ii] = x_test_noisy[ii] + noise_factor * np.random.normal(loc=0.0, scale=std,
                                                                                  size=x_test[ii].shape)
            x_train_noisy[ii] = np.clip(x_train_noisy[ii], 0, 1)
            x_test_noisy[ii] = np.clip(x_test_noisy[ii], 0, 1)
    else:
        x_train, x_test = train_test_split(x, test_size=0.2)
        x_train_noisy = x_train.copy()
        x_test_noisy = x_test.copy()
        x_train_noisy = x_train_noisy + noise_factor * np.random.normal(loc=0.0, scale=std, size=x_train.shape)
        x_test_noisy = x_test_noisy + noise_factor * np.random.normal(loc=0.0, scale=std, size=x_test.shape)
        x_train_noisy = np.clip(x_train_noisy, 0, 1)
        x_test_noisy = np.clip(x_test_noisy, 0, 1)
    output = (x_train_noisy, x_train, x_test_noisy, x_test)
    return output


class MyDataset:
    """MyDataset"""
    def __init__(self, x, nf=0.5, std=1.0, train=True):
        train_data, train_label, test_data, test_label = build_dataset(x, nf, std)
        if train:
            self._data = train_data
            self._label = train_label
        else:
            self._data = test_data
            self._label = test_label

    def __getitem__(self, index):
        """自定义随机访问函数"""
        data = np.concatenate([self._data[i][index] for i in range(9)], axis=0)
        label = np.concatenate([self._label[i][index] for i in range(9)], axis=0)
        return (data, label)

    def __len__(self):
        return len(self._data[0])


def build_model(x, input_dims, archs, save_models, nf=0.5, std=1.0, mtype='mda', epochs=80, batch_size=64):
    """build_model"""
    if mtype == 'mda':
        train_net = build_mda_model(input_dims=input_dims, encoding_dims=archs)
    else:
        print("### Wrong model.")
    train_data = MyDataset(x, nf=nf, std=std, train=True)
    test_data = MyDataset(x, nf=nf, std=std, train=False)
    train_dataset = ds.GeneratorDataset(train_data, column_names=['data', 'label'], shuffle=True)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    valid_dataset = ds.GeneratorDataset(test_data, column_names=['data', 'label'], shuffle=True)
    valid_dataset = valid_dataset.batch(batch_size=batch_size)
    for epoch in range(epochs):
        step = 0
        train_net.set_train()
        train_steps = train_dataset.get_dataset_size()
        val_steps = valid_dataset.get_dataset_size()
        for d in train_dataset.create_dict_iterator():
            train_result = train_net(ops.cast(d["data"], ms.float32), ops.cast(d["label"], ms.float32))
            print(f"train_Epoch: [{epoch} / {epochs}],"
                  f"step: [{step} / {train_steps}],"
                  f"train_loss: {train_result}")
            step = step + 1
        train_net.set_train(False)
        step = 0
        for d in valid_dataset.create_dict_iterator():
            val_result = train_net(ops.cast(d["data"], ms.float32), ops.cast(d["label"], ms.float32))
            print(f"val_Epoch: [{epoch} / {epochs}],"
                  f"step: [{step} / {val_steps}],"
                  f"val_loss: {val_result}")
            step = step + 1
    ms.save_checkpoint(train_net, save_models)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--mda_select_nets', help='select nets of mda', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 10])
    parser.add_argument('--ppmi_dir', help='PPMI directory', type=str,
                        default='dataset/PPMI/')
    parser.add_argument('--model_path', help='directory with models', type=str,
                        default='checkpoint/')
    parser.add_argument('--select_arch', help='select architecture of mda', type=list,
                        default=[3])
    parser.add_argument('--epochs', help='epochs of mda', type=int, default=150)
    parser.add_argument('--batch_size', help='batch size of mda', type=int, default=64)
    parser.add_argument('--noise_factor', help='noise factor of mda', type=float, default=0.5)
    parser.add_argument('--device_id', help='device id', type=int, default=1)
    args = parser.parse_args()
    ms.set_context(device_target='GPU', device_id=args.device_id)

    ORG = 'drug'
    MODELTYPE = 'mda'  # {mda or ae}

    MODELPATH = args.model_path  # directory with models
    SELECTARC = args.select_arch  # a number 1-10 (see below)
    SELECTNETS = args.mda_select_nets  # a number 1-10 (see below)
    EPOCHS = args.epochs
    BATCHSIZE = args.batch_size
    NF = args.noise_factor  # nf > 0 for denoising AE/MDA

    # all possible combinations for architectures
    arch = {}
    arch['mda'] = {}
    arch['mda']['drug'] = {}
    arch['mda']['drug'] = {1: [9 * 100],
                           2: [9 * 1000, 9 * 100, 9 * 1000],
                           3: [9 * 1000, 9 * 500, 9 * 100, 9 * 500, 9 * 1000],
                           4: [9 * 1000, 9 * 500, 9 * 200, 9 * 100, 9 * 200, 9 * 500, 9 * 1000],
                           5: [9 * 1000, 9 * 800, 9 * 500, 9 * 200, 9 * 100, 9 * 200, 9 * 500, 9 * 800, 9 * 1000],
                           }

    arch['ae'] = {}
    arch['ae']['drug'] = {}
    arch['ae']['drug'] = {1: [1000],
                          2: [2000, 1000, 2000],
                          3: [2000, 1500, 1000, 1500, 2000],
                          }

    # load PPMI matrices
    NETS = []
    input_dims_ = []
    for i in SELECTNETS:
        print("### [%d] Loading network..." % (i))
        N = sio.loadmat(args.ppmi_dir + ORG + '_net_' + str(i) + '.mat', squeeze_me=True)
        Net = N['Net'].todense()
        print("Net %d, NNofile_keywords=%d \n" % (i, np.count_nonzero(Net)))
        NETS.append(minmax_scale(Net))
        input_dims_.append(Net.shape[1])

    # Training MDA
    model_names = []
    for a in SELECTARC:
        print("### [%s] Running for architecture: %s" % (MODELTYPE, str(arch.get(MODELTYPE).get(ORG).get(a))))
        MODELNAME = 'mda.ckpt'
        if not Path.isfile(MODELPATH + MODELNAME):
            save_model = MODELPATH + MODELNAME
            build_model(NETS, input_dims_, arch.get(MODELTYPE).get(ORG).get(a), save_model, NF, 1.0, MODELTYPE,
                        EPOCHS, BATCHSIZE)
