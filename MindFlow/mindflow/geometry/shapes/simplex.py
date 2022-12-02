"""Sampler of simplex shapes."""
import math
from itertools import combinations

import numpy as np

from .shapes import Shape, derive_union_boundary


class Segment(Shape):
    """Segment sampler."""
    def __init__(self, vertices):
        x0, y0, x1, y1 = np.ravel(vertices)
        self._slope = (y1 - y0)/(x1 - x0)
        self._intercpet = (y0*x1 - y1*x0)/(x1 - x0)
        self._x_min = min(x0, x1)
        self._x_max = max(x0, x1)
        super(Segment, self).__init__(volume=math.sqrt((x0 - x1)*(x0 - x1) + (y0 - y1)*(y0 - y1)))

    def is_inside(self, pts):
        x = pts[:, 0]
        return (x >= self._x_min) & (x <= self._x_max) & self.is_on(pts)

    def is_on(self, pts):
        """Check if ``x`` is on the line defined by the segment."""
        x, y = pts.T
        return np.isclose(y, self.line_equation(x))

    def sample(self, num_samps):
        x = self._x_min + (self._x_max - self._x_min)*np.random.rand(num_samps, 1)
        return np.hstack([x, self.line_equation(x)])

    def line_equation(self, x):
        return self._intercpet + self._slope*x


class Simplex(Shape):
    """Simplex (triangle in 2D and tetrahedron in 3D) sampler."""
    def __init__(self, vertices, boundary_type='none'):
        vertices = self._validate(vertices)
        # Compute volume
        volume = .5*abs(np.linalg.det(np.hstack([np.ones([len(vertices), 1]), vertices])))
        # Prepare boundary
        num_vertices = len(vertices)
        if num_vertices == 3:
            cls = Segment
        elif num_vertices == 4:
            cls = Triangle3D
        else:
            raise ValueError
        shapes = [
            cls(vertices[list(indices)])
            for indices in combinations(range(num_vertices), num_vertices - 1)
        ]
        boundary = derive_union_boundary(shapes, boundary_type)
        super(Simplex, self).__init__(volume, boundary)
        # Prepare transform matrix
        self._x0 = vertices[0]
        matrix = vertices[1:] - vertices[0]
        if np.linalg.det(matrix) < 0:
            # Ensure the vertices are in an anti-clockwise order
            tmp = matrix[0].copy()
            matrix[0] = matrix[1]
            matrix[1] = tmp
        self._matrix = matrix
        self._matrix_inv = np.linalg.inv(matrix)

    def is_inside(self, pts):
        pts = self.inverse_transform(pts)
        return np.all(pts > 0, axis=1) & (pts.sum(axis=1) < 1.)

    def sample(self, num_samps):
        return self.transform(self.sample_unit_triangle(num_samps))

    def sample_unit_triangle(self, num_samps):
        """Uniformly sample points inside a unit simplex."""
        samps = np.random.rand(num_samps, 2)
        cond = np.sum(samps, axis=1) > 1
        samps[cond] = 1 - samps[cond][:, [1, 0]]

        if len(self._x0) == 3:
            # Notice the area of the triangle at z is .5*(1 - z)^2
            z = 1 - np.cbrt(np.random.rand(num_samps, 1))
            samps = np.hstack([(1 - z)*samps, z])

        return samps

    def transform(self, pts):
        """Transform the unit triangle into the given triangle."""
        return np.matmul(pts, self._matrix) + self._x0

    def inverse_transform(self, pts):
        """Transform the given triangle into the unit triangle."""
        return np.matmul(pts - self._x0, self._matrix_inv)

    def _validate(self, vertices):
        vertices = np.asarray(vertices).copy()
        if vertices.shape == (3, 2):
            if Segment(vertices[:2]).is_on(vertices[2]):
                raise ValueError("Three points lie on a line.")
        elif vertices.shape == (4, 3):
            pass
        else:
            raise ValueError
        return vertices


class Triangle3D(Shape):
    """3D Triangle sampler."""
    def __init__(self, vertices):
        vertices = self._validate(vertices)
        # Check if the given triangle can be reduced into 2D.
        dim_reduced = None
        for i_col in range(vertices.shape[1]):
            if np.allclose(vertices[:, i_col], vertices[0, i_col]):
                dim_reduced = i_col
                val_reduced = vertices[0, i_col]
                break

        if dim_reduced is None:
            # Compute area
            cross_product = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            volume = .5*abs(np.linalg.norm(cross_product, ord=2))
            super(Triangle3D, self).__init__(volume)
            # Prepare transform matrix
            matrix = vertices
            if np.linalg.det(matrix) < 0:
                matrix = np.asarray([matrix[1], matrix[0], matrix[2]])
            self._matrix = matrix
            self._matrix_inv = np.linalg.inv(matrix)
            #
            self._tri_reduced = None
        else:
            self._tri_reduced = Simplex(
                np.delete(vertices, dim_reduced, axis=1), boundary_type='none')
            self._dim_reduced = dim_reduced
            self._val_reduced = val_reduced
            super(Triangle3D, self).__init__(self._tri_reduced.volume)

    def is_inside(self, pts):
        if self._tri_reduced is None:
            pts = self.inverse_transform(pts)
            return np.all((pts > 0.) & (pts < 1.), axis=1) & np.isclose(pts.sum(axis=1), 1.)

        pts_reduced = np.delete(pts, self._dim_reduced, axis=1)
        return self._tri_reduced.is_inside(pts_reduced) \
            & np.isclose(pts[:, self._dim_reduced], self._val_reduced)

    def sample(self, num_samps):
        if self._tri_reduced is None:
            return self.transform(self.sample_unit_triangle(num_samps))

        samps_reduced = self._tri_reduced.sample(num_samps)
        samps = np.insert(samps_reduced, self._dim_reduced, self._val_reduced, axis=1)
        return samps

    def sample_unit_triangle(self, num_samps):
        """Uniformly sample points in the plane defined by (0, 0, 1), (1, 0, 0)
        and (0, 1, 0).

        This is equivalent to sample from a flat dirichlet distribution.
        """
        samps = np.log(np.random.rand(num_samps, 3))
        samps /= samps.sum(axis=1, keepdims=True)
        return samps

    def transform(self, pts):
        """Transform the unit triangle into the given triangle."""
        return np.matmul(pts, self._matrix)

    def inverse_transform(self, pts):
        """Transform the given triangle into the unit triangle."""
        return np.matmul(pts, self._matrix_inv)

    def _validate(self, vertices):
        vertices = np.asarray(vertices).copy()
        assert vertices.shape == (3, 3)
        # TODO Check the cases that three points lie on a line
        return vertices
