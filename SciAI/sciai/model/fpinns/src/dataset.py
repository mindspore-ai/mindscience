import numpy as np
import skopt


def generate_sample(n_samples):
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    # 1st point: [0, 0, ...]
    sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
    space = [(0.0, 1.0)]
    return np.asarray(
        sampler.generate(dimensions=space, n_samples=n_samples)[0:], dtype=np.float32
    )


def random_points(n, diam, l):
    x = generate_sample(n)
    return (diam * x + l).astype(np.float32)
