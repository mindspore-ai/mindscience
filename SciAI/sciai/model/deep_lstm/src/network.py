""" network definition"""
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import stop_gradient


class DeepLSTM(nn.Cell):
    """DeepLSTM network definition"""
    def __init__(self, input_dim, num_classes, embed_dim=100):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, embed_dim, 1, has_bias=True, batch_first=True, bidirectional=False)
        self.act = nn.ReLU()
        self.lstm2 = nn.LSTM(embed_dim, embed_dim, 1, has_bias=True, batch_first=True, bidirectional=False)
        self.dense1 = nn.Dense(embed_dim, embed_dim)
        self.dense2 = nn.Dense(embed_dim, num_classes)

    def construct(self, x, h0, c0):
        """forward function"""
        output, (_, _) = self.lstm1(x, (h0, c0))
        output = self.act(output)
        output, (_, _) = self.lstm2(output, (h0, c0))
        output = self.act(output)
        output = self.dense1(output)
        output = self.act(output)
        output = self.dense2(output)
        return output


class NetWithLoss(nn.Cell):
    def __init__(self, network, loss):
        super(NetWithLoss, self).__init__()
        self.network = network
        self.loss = loss

    def construct(self, x, h0, c0, y):
        logits = self.network(x, h0, c0)
        loss = self.loss(logits, y)
        return loss


class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell definition"""
    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network  # 带loss的网络结构
        self.network.set_grad()  # PYNATIVE模式时需要，如果为True，则在执行正向网络时，将生成需要计算梯度的反向网络。
        self.optimizer = optimizer  # 优化器，用于参数更新
        self.weights = self.optimizer.parameters

    def construct(self, x, h0, c0, y):
        loss = self.network(x, h0, c0, y)  # 运行正向网络，获取loss
        grad_fn = ms.grad(self.network, grad_position=None, weights=self.weights)
        grads = grad_fn(x, h0, c0, y)
        grads = ms.ops.clip_by_global_norm(grads, clip_norm=1e-3)
        self.optimizer(grads)  # 优化器更新参数
        return loss


class EvalNet(nn.Cell):
    def __init__(self, netwithloss):
        super(EvalNet, self).__init__()
        self.network = netwithloss

    def construct(self, x, h0, c0, label):
        loss = self.network(x, h0, c0, label)
        loss = stop_gradient(loss)
        return loss
