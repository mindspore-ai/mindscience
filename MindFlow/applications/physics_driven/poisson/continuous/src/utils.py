"""Utilities for the Poisson example."""
import math
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import numpy as ms_np
from mindspore import nn, ops


class AnalyticSolution(nn.Cell):
    """Analytic solution."""

    def __init__(self, n_dim):
        super(AnalyticSolution, self).__init__()
        self.prod = ops.ReduceProd(keep_dims=True)
        self.factor = ms.Tensor(1/(16.*n_dim*math.pi*math.pi))

    def construct(self, x):
        return self.factor*self.prod(ms_np.sin(4*math.pi*x), 1)


def relative_l2(x, y):
    """Calculate the relative L2 error."""
    return ops.sqrt(ops.reduce_mean(ops.square(x - y))) / ops.sqrt(
        ops.reduce_mean(ops.square(y))
    )


def calculate_l2_error(model, ds_test, n_dim):
    """Calculate the relative L2 error."""
    # Create solution
    solution = AnalyticSolution(n_dim)

    # Evaluate
    for x_domain, x_bc in ds_test:
        y_pred_domain = model(x_domain)
        y_test_domain = solution(x_domain)

        y_pred_bc = model(x_bc)
        y_test_bc = solution(x_bc)
        print(
            "Relative L2 error (domain): {:.4f}".format(
                relative_l2(y_pred_domain, y_test_domain).asnumpy()
            )
        )
        print(
            "Relative L2 error (bc): {:.4f}".format(
                relative_l2(y_pred_bc, y_test_bc).asnumpy()
            )
        )
        print("")


def draw2d(model, nmaps):
    '''draw the cloud map'''
    # Create the dataset
    x = np.linspace(0., 1., nmaps)
    y = np.linspace(0., 1., nmaps)
    x, y = np.meshgrid(x, y)
    x = np.reshape(x, (nmaps*nmaps, 1))
    y = np.reshape(y, (nmaps*nmaps, 1))
    z = np.concatenate((x, y), axis=1)
    z = model(ms.Tensor(z, dtype=ms.float32)).asnumpy()

    plt.figure()
    plt.subplot(111)
    plt.scatter(x, y, c=z, cmap=plt.cm.rainbow,
                vmin=min(z), vmax=max(z))
    plt.text(50, 50, r'DNN', {'color': 'b', 'fontsize': 20})
    plt.colorbar()
    plt.show()
