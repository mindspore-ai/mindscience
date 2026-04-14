"""Sympnet Modules Definition"""
import mindspore as ms
from mindspore import ops, nn

from .module import Module, StructureNN


class LinearModule(Module):
    """Linear symplectic module. Si is distributed N(0, 0.01), and b is set to zero."""

    def __init__(self, dim, layers):
        super(LinearModule, self).__init__()
        self.dim = dim
        self.layers = layers
        self.randn = ops.StandardNormal()
        self.zeros = ops.Zeros()
        self.matmul = ops.MatMul()
        d = int(self.dim / 2)
        param_list = [ms.Parameter(self.randn((d, d)) * 0.01, name='S{}'.format(i + 1)) for i in range(self.layers)]
        self.param_tuple = ms.ParameterTuple(param_list)
        if d == 1:
            self.bp = ms.Parameter(ms.Tensor(0, ms.float32))
            self.bq = ms.Parameter(ms.Tensor(0, ms.float32))
        else:
            self.bp = ms.Parameter(ms.Tensor([0] * d, ms.float32))
            self.bq = ms.Parameter(ms.Tensor([0] * d, ms.float32))

    def construct(self, pqh):
        """Network forward pass"""
        p, q, h = pqh[0], pqh[1], pqh[2]
        for i in range(self.layers):
            s = self.param_tuple[i]
            if i % 2 == 0:
                p = p + self.matmul_uniform(q, (s + s.transpose())) * h
            else:
                q = q + self.matmul_uniform(p, (s + s.transpose())) * h
        return p + self.bp * h, q + self.bq * h, h

    def matmul_uniform(self, a, b):
        if len(a.shape) == 1:
            res = self.matmul(a.expand_dims(0), b)
            return res.squeeze()
        return self.matmul(a, b)


class ActivationModule(Module):
    """Activation symplectic module."""

    def __init__(self, dim, activation, mode):
        super(ActivationModule, self).__init__()
        self.dim = dim
        self.activation = activation
        self.mode = mode
        self.randn = ops.StandardNormal()
        d = int(self.dim / 2)
        self.a = ms.Parameter(self.randn((d,)) * 0.01)

    def construct(self, pqh):
        """Network forward pass"""
        p, q, h = pqh[0], pqh[1], pqh[2]
        if self.mode == 'up':  # pylint: disable=R1705
            return p + self.act(q) * self.a * h, q, h
        elif self.mode == 'low':
            return p, self.act(p) * self.a * h + q, h
        raise ValueError()


class GradientModule(Module):
    """Gradient symplectic module."""

    def __init__(self, dim, width, activation, mode):
        super(GradientModule, self).__init__()
        self.dim = dim
        self.width = width
        self.activation = activation
        self.mode = mode
        self.randn = ops.StandardNormal()
        self.matmul = ops.MatMul()
        d = int(self.dim / 2)
        self.k = ms.Parameter(self.randn((d, self.width)) * 0.01)
        self.a = ms.Parameter(self.randn((self.width,)) * 0.01)
        self.b = ms.Parameter(ops.zeros(self.width, ms.float32))

    def construct(self, pqh):
        """Network forward pass"""
        p, q, h = pqh[0], pqh[1], pqh[2]
        if self.mode == 'up':  # pylint: disable=R1705
            grad_h = self.matmul_uniform(self.act(self.matmul_uniform(q, self.k + self.b)) * self.a,
                                         self.k.transpose())
            return p + grad_h * h, q, h
        if self.mode == 'low':
            grad_h = self.matmul_uniform(self.act(self.matmul_uniform(p, self.k + self.b)) * self.a,
                                         self.k.transpose())
            return p, grad_h * h + q, h
        raise ValueError()

    def matmul_uniform(self, a, b):
        if len(a.shape) == 1:
            res = self.matmul(a.expand_dims(0), b)
            return res.squeeze()
        return self.matmul(a, b)


class SympNet(StructureNN):
    """Sympnet module."""
    def __init__(self):
        super(SympNet, self).__init__()
        self.dim = None
        self.cat = ops.Concat(axis=-1)
        self.cast = ops.Cast()

    @property
    def third_parameter(self):
        pass

    def predict(self, xh, steps=1):  # pylint: disable=W0221
        dim = xh.shape[-1]
        size = len(xh.shape)
        if dim == self.dim:
            pred = [xh]
            for _ in range(steps):
                pred.append(self(pred[-1]))
        else:
            x0, h = xh[..., :-1], xh[..., -1:]
            pred = [x0] + []
            for _ in range(steps):
                pred.append(self(self.cat([pred[-1], h])))
        pred = pred[1:]
        res = self.cat(pred).view((-1, steps, self.dim)[2 - size:])
        return res.asnumpy()


class LASympNet(SympNet):
    """LA-SympNet.
    Input: [num, dim] or [num, dim + 1]
    Output: [num, dim]
    """

    def __init__(self, dim, layers=3, sublayers=2, activation='sigmoid'):
        super(LASympNet, self).__init__()
        self.dim = dim
        self.layers = layers
        self.sublayers = sublayers
        self.activation = activation
        self.cat = ops.Concat(axis=-1)
        self.__init_modules()

    @property
    def third_parameter(self):
        return self.sublayers

    def construct(self, pqh):
        """Network forward pass"""
        d = int(self.dim / 2)
        if pqh.shape[-1] == self.dim + 1:
            p, q, h = pqh[..., :d], pqh[..., d:-1], pqh[..., -1:]
        elif pqh.shape[-1] == self.dim:
            p, q, h = pqh[..., :d], pqh[..., d:], ops.ones_like(pqh[..., -1:])
        else:
            raise ValueError
        return self.cat(self.modules([p, q, h])[:2])

    def __init_modules(self):
        self.modules = nn.SequentialCell()
        for i in range(self.layers - 1):
            self.modules.append(LinearModule(self.dim, self.sublayers))
            mode = 'up' if i % 2 == 0 else 'low'
            self.modules.append(ActivationModule(self.dim, self.activation, mode))
        self.modules.append(LinearModule(self.dim, self.sublayers))


class GSympNet(SympNet):
    """G-SympNet.
    Input: [num, dim] or [num, dim + 1]
    Output: [num, dim]
    """

    def __init__(self, dim, layers=3, width=20, activation='sigmoid'):
        super(GSympNet, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width
        self.activation = activation
        self.cat = ops.Concat(axis=-1)
        self.gradms = nn.SequentialCell()
        for i in range(self.layers):
            mode = 'up' if i % 2 == 0 else 'low'
            self.gradms.append(GradientModule(self.dim, self.width, self.activation, mode))

    @property
    def third_parameter(self):
        return self.width

    def construct(self, pqh):
        """Network forward pass"""
        d = int(self.dim / 2)
        if pqh.shape[-1] == self.dim + 1:
            p, q, h = pqh[..., :d], pqh[..., d:-1], pqh[..., -1:]
        elif pqh.shape[-1] == self.dim:
            p, q, h = pqh[..., :d], pqh[..., d:], ops.ones_like(pqh[..., -1:])
        else:
            raise ValueError
        p, q, h = self.gradms([p, q, h])
        return self.cat((p, q))
