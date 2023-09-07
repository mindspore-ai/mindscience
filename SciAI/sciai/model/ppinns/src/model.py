from mindspore import nn, ops, Tensor, float32


class Net(nn.Cell):
    def __init__(self, fnn, odenn):
        super().__init__()
        self.fnn = fnn
        self.odenn = odenn

    def construct(self, t_train, t_bc_train):
        """Network forward pass"""
        u_pred = self.fnn(t_train)
        u_0_pred = self.fnn(t_bc_train)
        f_pred = self.odenn(t_train)
        return u_pred, u_0_pred, f_pred


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, t_train, t_bc_train, u_0_train):
        """Network forward pass"""
        u_pred, u_0_pred, f_pred = self._backbone(t_train, t_bc_train)
        return ops.add(self._loss_fn(f_pred, Tensor(0, dtype=float32)),
                       self._loss_fn(u_0_train, u_0_pred))
