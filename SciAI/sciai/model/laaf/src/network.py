from mindspore import ops, nn

from sciai.architecture.basic_block import MSE


class RecoverySlopeLoss(nn.Cell):
    def __init__(self, mlp):
        super(RecoverySlopeLoss, self).__init__()
        self.mlp = mlp
        self.mse = MSE()

    def construct(self, x_train, y_train):
        """Network forward pass"""
        y_pred = self.mlp(x_train)
        loss = self.mse(y_pred - y_train) + \
               1.0 / (ops.reduce_mean(sum(ops.exp(ops.reduce_mean(a)) for a in self.mlp.a_value())))
        return loss, y_pred

    def a_values_np(self):
        return ops.stack(self.mlp.a_value())
