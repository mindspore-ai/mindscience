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
"""Training of cVAE"""
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.nn as nn
from model import VAE

ms.set_context(mode=ms.PYNATIVE_MODE)


class CvaeDataset:
    """自定义数据集类"""

    def __init__(self, dataset):
        """自定义初始化操作"""
        self.data = dataset
        self.label = dataset  # 自定义数据

    def __getitem__(self, index):
        """自定义随机访问函数"""
        return self.data[index], self.label[index]

    def __len__(self):
        """自定义获取样本数据量函数"""
        return len(self.data)


class CvaeLoss(nn.Cell):
    """自定义损失函数Loss"""

    def __init__(self, option, alpha, beta):
        """初始化"""
        super(CvaeLoss, self).__init__()
        self.option = option
        self.alpha = alpha
        self.beta = beta

    def regularization(self, mu, logvar):
        return -0.5 * ops.ReduceSum()(1 + logvar - ops.Pow()(mu, 2) - ops.Exp()(logvar))

    def guassian_loss(self, recon_x, x):
        weights = x * self.alpha + (1 - x)
        loss_ = x - recon_x
        loss_ = ops.ReduceSum()(weights * loss_ * loss_)
        return loss_

    def bec_loss(self, recon_x, x):
        eps = 1e-8
        loss_ = -ops.ReduceSum()(self.alpha * ops.log(recon_x + eps) * x + ops.log(1 - recon_x + eps) * (1 - x))
        return loss_

    def construct(self, input_, label):
        recon_x, _, mu, logvar, _ = input_
        if self.option == 1:
            loss_ = self.guassian_loss(recon_x, label) + self.regularization(mu, logvar) * self.beta
        else:
            loss_ = self.bec_loss(recon_x, label) + self.regularization(mu, logvar) * self.beta
        return loss_


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

    def construct(self, *inputs):
        loss_ = self.network(*inputs)  # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        self.optimizer_(grads)  # 使用优化器更新权重参数
        return loss_


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--batch', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--m', type=int, default=300, help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dir', help='dataset directory',
                        default='dataset')
    parser.add_argument('--vae_encoder_layer_sizes', type=list, default=[1519, 1000], help='encoder layer sizes')
    parser.add_argument('--vae_latent_size', help='latent size', type=int, default=100)
    parser.add_argument('--vae_decoder_layer_sizes', help='decoder layer sizes', type=list, default=[1000, 1519])
    parser.add_argument('--learn_rate', help='learning rate', type=float, default=0.001)  # side0.0001,rating0.01
    parser.add_argument('--a', help='parameter alpha', type=float, default=15)
    parser.add_argument('--b', help='parameter beta', type=float, default=3)
    parser.add_argument('--rating', help='feed input as rating', action="store_true")
    parser.add_argument('--save', help='save model', action="store_true")
    parser.add_argument('--load', help='load model, 1 for cvae', type=int, default=0)
    parser.add_argument('--device_id', help='device id', type=int, default=0)

    args = parser.parse_args()
    # whether to ran with cuda
    ms.set_context(device_target='GPU', device_id=args.device_id)

    print('dataset directory: ' + args.dir)
    directory = args.dir

    PATH = '{}/drugDisease.txt'.format(directory)
    print('train data path: ' + PATH)
    R = np.loadtxt(PATH)
    RTENSOR = R.transpose()
    if args.rating:  # feed in with rating
        whole_positive_index = []
        whole_negative_index = []
        for i in range(np.shape(RTENSOR)[0]):
            for j in range(np.shape(RTENSOR)[1]):
                if int(RTENSOR[i][j]) == 1:
                    whole_positive_index.append([i, j])
                elif int(RTENSOR[i][j]) == 0:
                    whole_negative_index.append([i, j])
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)
        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
        count = 0
        for i in whole_positive_index:
            data_set[count][0] = i[0]
            data_set[count][1] = i[1]
            data_set[count][2] = 1
            count += 1
        for i in negative_sample_index:
            data_set[count][0] = whole_negative_index[i][0]
            data_set[count][1] = whole_negative_index[i][1]
            data_set[count][2] = 0
            count += 1
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0, 1000, 1)[0]
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        kf = fold.split(data_set[:, 2], data_set[:, 2])

        for train_index, test_index in kf:
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            Xtrain = np.zeros((np.shape(RTENSOR)[0], np.shape(RTENSOR)[1]))
            for ele in DTItrain:
                Xtrain[ele[0], ele[1]] = ele[2]
            RTENSOR = ms.Tensor(Xtrain.astype('float32'))

            # data
            rtensor = CvaeDataset(RTENSOR.asnumpy())
            train_dataset = ds.GeneratorDataset(rtensor, column_names=['data', 'label'], shuffle=True)
            train_dataset = train_dataset.batch(args.batch)
            # loss
            loss = CvaeLoss(option=2, alpha=args.a, beta=args.b)
            # net
            network = VAE(args.vae_encoder_layer_sizes, args.vae_latent_size, args.vae_decoder_layer_sizes)
            print(network)
            net_with_loss = nn.WithLossCell(network, loss)
            optimizer = nn.Adam(params=network.trainable_params(), learning_rate=args.learn_rate, weight_decay=1e-1)

            if args.load > 0:
                NAME = 'cvae'
                PATH = 'checkpoint/' + NAME + '.ckpt'
                print('load model from path: ' + PATH)
                ms.load_checkpoint(PATH, net=network)

            train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
            # 设置网络为训练模式
            train_net.set_train()
            epochs = args.m
            steps = train_dataset.get_dataset_size()
            for epoch in range(epochs):
                step = 0
                for d in train_dataset.create_dict_iterator():
                    result = train_net(d['data'], d['label'])
                    print(f"Epoch: [{epoch} / {epochs}], "
                          f"step: [{step} / {steps}], "
                          f"loss: {result / len(d['data'])}, ")
                    step = step + 1

            train_net.set_train(False)
            score, _, _, _, _ = network(ms.Tensor(rtensor.data))
            print(score.shape)
            ZSCORE = score.asnumpy()

            pred_list = []
            ground_truth = []
            for ele in DTItrain:
                pred_list.append(ZSCORE[ele[0], ele[1]])
                ground_truth.append(ele[2])
            train_auc = roc_auc_score(ground_truth, pred_list)
            train_aupr = average_precision_score(ground_truth, pred_list)
            print('train auc aupr,', train_auc, train_aupr)
            pred_list = []
            ground_truth = []
            for ele in DTItest:
                pred_list.append(ZSCORE[ele[0], ele[1]])
                ground_truth.append(ele[2])
            test_auc = roc_auc_score(ground_truth, pred_list)
            test_aupr = average_precision_score(ground_truth, pred_list)
            print('test auc aupr', test_auc, test_aupr)
            test_auc_fold.append(test_auc)
            test_aupr_fold.append(test_aupr)
        avg_auc = np.mean(test_auc_fold)
        avg_pr = np.mean(test_aupr_fold)
        print('mean auc aupr', avg_auc, avg_pr)
        if args.save:
            NAME = 'cvae'
            PATH = 'checkpoint/' + NAME
            ms.save_checkpoint(network, PATH)
            score, _, _, _, _ = network(ms.Tensor(RTENSOR))
            ZSCORE = score.asnumpy()
            np.save('biozscore.npy', ZSCORE)
            np.save('Rtensor.npy', RTENSOR)
    else:  # feed in with side information
        PATH = 'dataset/drugFeatures.txt'
        print('feature data path: ' + PATH)
        fea = np.loadtxt(PATH)
        X = fea.transpose()
        X[X > 0] = 1
        X = ms.Tensor(X.astype('float32'), ms.float32)
        X = CvaeDataset(X.asnumpy())
        train_dataset = ds.GeneratorDataset(X, column_names=['data', 'label'], shuffle=True)
        train_dataset = train_dataset.batch(args.batch)
        # loss
        loss = CvaeLoss(option=1, alpha=args.a, beta=args.b)
        # net
        network = VAE(args.vae_encoder_layer_sizes, args.vae_latent_size, args.vae_decoder_layer_sizes)
        net_with_loss = nn.WithLossCell(network, loss)

        if args.load > 0:
            NAME = 'cvae'
            PATH = 'checkpoint/' + NAME
            print('load model from path: ' + PATH)
            ms.load_checkpoint(PATH, net=network)

        optimizer = nn.Adam(params=network.trainable_params(), learning_rate=args.learn_rate)

        train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
        train_net.set_train()
        epochs = args.m
        steps = train_dataset.get_dataset_size()
        for epoch in range(epochs):
            step = 0
            for d in train_dataset.create_dict_iterator():
                result = train_net(d['data'], d['label'])
                print(f"Epoch: [{epoch} / {epochs}], "
                      f"step: [{step} / {steps}], "
                      f"loss: {result / len(d['data'])}")
                step = step + 1

        if args.save:
            NAME = 'cvae'
            PATH = 'checkpoint/' + NAME
            ms.save_checkpoint(network, PATH)
