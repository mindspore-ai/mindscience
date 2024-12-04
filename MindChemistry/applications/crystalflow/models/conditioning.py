"""condition utils"""
import mindspore as ms
from mindspore import ops, nn

class GaussianExpansion(nn.Cell):
    r"""Expansion layer using a set of Gaussian functions.
    https://github.com/atomistic-machine-learning/cG-SchNet/blob/53d73830f9fb1158296f060c2f82be375e2bb7f9/nn_classes.py#L687)
    """
    def __init__(self, start, stop, n_gaussians=50, trainable=False, width=None):
        super(GaussianExpansion, self).__init__()
        offset = ops.linspace(start, stop, n_gaussians)
        self.n_out = n_gaussians
        if width is None:
            widths = (offset[1] - offset[0]) * ops.ones_like(offset)
        else:
            widths = width * ops.ones_like(offset)
        if trainable:
            self.widths = ms.Parameter(widths)
            self.offsets = ms.Parameter(offset)
        else:
            self.widths = ms.Parameter(widths, requires_grad=False)
            self.offsets = ms.Parameter(offset, requires_grad=False)

    def construct(self, prop):
        """Compute expanded gaussian property values.
        Args:
            prop (Tensor): property values of (N_b x 1) shape.
        Returns:
            Tensor: layer output of (N_b x N_g) shape.
        """
        prop = prop.reshape(prop.shape[0], -1)
        coeff = -0.5 / ops.pow(self.widths, 2)[None, :]
        diff = prop - self.offsets[None, :]
        return ops.exp(coeff * ops.pow(diff, 2))
