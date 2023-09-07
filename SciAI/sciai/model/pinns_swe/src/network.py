from mindspore import ops, nn
from sciai.architecture import MLP, Normalize


class PeriodicSphere(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, inputs0, inputs1):
        """Network forward pass"""
        return ops.concat((ops.cos(inputs1) * ops.cos(inputs0),
                           ops.cos(inputs1) * ops.sin(inputs0),
                           ops.sin(inputs1)), axis=1)


# Define the network
class Model(nn.Cell):
    def __init__(self, t0, tfinal, layers):
        super().__init__()
        self.normalize = Normalize(t0, tfinal)
        self.periodic_sphere = PeriodicSphere()
        self.concat = ops.Concat(axis=1)
        self.mlp = MLP(layers, weight_init="XavierNormal", bias_init="zeros", activation="tanh")

    def construct(self, inp1, inp2, inp3):
        """Network forward pass"""
        b1 = self.normalize(inp1)
        b23 = self.periodic_sphere(inp2, inp3)
        b = self.concat((b1, b23))
        out = self.mlp(b)
        return out
