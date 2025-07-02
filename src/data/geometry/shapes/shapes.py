"""Basic classes for the shape module."""
from abc import ABC, abstractmethod

import numpy as np


class Shape(ABC):
    """Base Shape class.

    Args:
        volume (float): Volume, area or length of the shape.
        boundary (Shape): Boundary of the shape.
    """
    def __init__(self, volume, boundary=None):
        self._volume = volume
        self._boundary = boundary

    @property
    def volume(self):
        """Volume of the shape (length for 1D shape and area for 2D shape)."""
        return self._volume

    @abstractmethod
    def is_inside(self, pts):
        """Check if the points is inside the shape.

        Should return a flattened array.
        """

    def is_on_boundary(self, pts):
        if self._boundary is None:
            raise ValueError("The shape has no boundary.")
        return self._boundary.is_inside(pts)

    @abstractmethod
    def sample(self, num_samps):
        """Uniformly sample points in the shape."""

    def sample_boundary(self, num_samps):
        if self._boundary is None:
            raise ValueError("The shape has no boundary.")
        return self._boundary.sample(num_samps)


class Union(Shape):
    """Union of a group of non-overlapping shapes.

    Args:
        shapes (list): A list of shape objects.
        boundary (Shape): Boundary of the union.
        use_probs (bool): If True, the expected number of samples in each
        boundary is proportional to the area (length) of the boundary;
        otherwise the expected number of samples in each boundary is the same.
    """
    def __init__(self, shapes, boundary=None, use_probs=True):
        probs = np.array([sh.volume for sh in shapes])
        volume = sum(probs)
        if use_probs:
            probs /= volume
        else:
            probs = [1./len(shapes)]*len(shapes)

        super(Union, self).__init__(volume, boundary)
        self._probs = probs
        self._shapes = shapes

    def is_inside(self, pts):
        return np.any([sh.is_inside(pts) for sh in self._shapes], axis=0)

    def sample(self, num_samps):
        num_each = np.random.multinomial(num_samps, self._probs)
        samps = np.vstack([sh.sample(num) for num, sh in zip(num_each, self._shapes)])
        np.random.shuffle(samps)
        return samps


def derive_union_boundary(shapes, boundary_type):
    """Create of a union based on the boundary type.

    Args:
        shapes (list): A list of shape objects.
        boundary_type (str): this can be ``'uniform'`` or ``'unweighted'``.

            - ``'uniform'``, the expected number of samples in each boundary is
              proportional to the area (length) of the boundary.
            - ``'unweighted'``, the expected number of samples in each boundary is
              the same.

    Returns:
        Union, Union of the input shapes.
    """
    if boundary_type == 'none':
        boundary = None
    else:
        if boundary_type == 'uniform':
            use_probs = True
        elif boundary_type == 'unweighted':
            use_probs = False
        else:
            raise ValueError
        boundary = Union(shapes, use_probs=use_probs)
    return boundary
