"""lattice transform utils"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops


def matrix_exp(m, n=20):  # m: (B,3,3)
    s = current_m = ops.eye(3, dtype=m.dtype)[None, :, :]
    for i in range(1, n + 1):
        current_m = (current_m @ m) / i  # m^n/n!
        s += current_m
    return s


class LatticePolarDecomp(nn.Cell):
    """class for transformation between lattice and lattice_polar"""
    def decompose(self, matrix):  # matrix as row vectors
        """transform lattice to lattice_polar"""
        a, u = ops.eig(matrix @ matrix.swapaxes(-1, -2))
        a, u = a.real(), u.real()
        s = u @ (ops.diag_embed(a.log()) / 2) @ u.swapaxes(-1, -2)

        k0 = s[:, 0, 1]
        k1 = s[:, 0, 2]
        k2 = s[:, 1, 2]
        k3 = (s[:, 0, 0] - s[:, 1, 1]) / 2
        k4 = (s[:, 0, 0] + s[:, 1, 1] - 2 * s[:, 2, 2]) / 6
        k5 = (s[:, 0, 0] + s[:, 1, 1] + s[:, 2, 2]) / 3
        k = ops.vstack([k0, k1, k2, k3, k4, k5]).swapaxes(-1, -2)
        return k

    def build(self, vector):
        k = vector
        s0 = ops.stack([k[:, 3] + k[:, 4] + k[:, 5], k[:, 0], k[:, 1]], 1)  # (B, 3)
        s1 = ops.stack([k[:, 0], -k[:, 3] + k[:, 4] + k[:, 5], k[:, 2]], 1)  # (B, 3)
        s2 = ops.stack([k[:, 1], k[:, 2], -2 * k[:, 4] + k[:, 5]], 1)  # (B, 3)
        s = ops.stack([s0, s1, s2], 1)  # (B, 3, 3)
        exp_s = matrix_exp(s)  # (B, 3, 3)
        return exp_s

    def sample(self, batch_size, sigma, dtype=None):
        v = ops.randn([batch_size, 6], dtype=dtype) * sigma
        v[:, -1] = v[:, -1] + 1
        return v

    def sample_like(self, vector, sigma):
        v = ops.randn_like(vector) * sigma
        v[:, -1] = v[:, -1] + 1
        return v

    def sample_numpy(self, batch_size, sigma, dtype=ms.float32):
        v = np.random.randn(batch_size, 6) * sigma
        v[:, -1] = v[:, -1] + 1
        v = ms.Tensor(v, dtype=dtype)
        return v

    def rand_like_numpy(self, vector, dtype=ms.float32):
        #the numpy version of ops.rand_like
        v = np.random.rand(*vector.shape)
        v = ms.Tensor(v, dtype=dtype)
        return v
