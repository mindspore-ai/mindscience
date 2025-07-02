"""Pentagon sampler."""
import math

import numpy as np

from .shapes import Union, derive_union_boundary
from .simplex import Segment, Simplex


class Pentagon(Union):
    """Pentagon.

    Aussme that the adjacent input vertices form en edge of the Pentagon. The
    code does not check the case that three points lie on a line.
    """
    def __init__(self, vertices, boundary_type='none'):
        vertices = self._validiate(vertices)
        shapes = self._create_triangles(vertices)
        segments = [
            Segment([vertices[idx], vertices[self._c_idx(idx + 1)]])
            for idx in range(len(vertices))
        ]
        boundary = derive_union_boundary(segments, boundary_type)
        super(Pentagon, self).__init__(shapes, boundary)

    def _create_triangles(self, vertices):
        """Divide the given pentagon into three triangles."""
        def compute_angle(vec0, vec1):
            """Compute the angle of two vectors."""
            cos_t = np.inner(vec0, vec1)/(np.linalg.norm(vec0, 2)*np.linalg.norm(vec1, 2))
            return math.acos(cos_t)

        # Check if the given pentagon is convex
        num_angles = 5
        angles = [0.]*num_angles
        for i_v in range(num_angles):
            vec0 = vertices[self._c_idx(i_v - 1)] - vertices[i_v]
            vec1 = vertices[self._c_idx(i_v + 1)] - vertices[i_v]
            angles[i_v] = compute_angle(vec0, vec1)

        theta_tot = sum(angles)
        if math.isclose(theta_tot, 3*math.pi):
            idx_start = 0
        else:
            for i_theta, theta in enumerate(angles):
                if math.isclose(theta_tot - 2*theta + 2*math.pi, 3*math.pi):
                    idx_start = i_theta
                    break

        vertices_tri = [
            np.asarray([
                vertices[idx_start],
                vertices[self._c_idx(idx_start + 1)],
                vertices[self._c_idx(idx_start + 2)]]),
            np.asarray([
                vertices[idx_start],
                vertices[self._c_idx(idx_start - 1)],
                vertices[self._c_idx(idx_start - 2)]]),
            np.asarray([
                vertices[idx_start],
                vertices[self._c_idx(idx_start + 2)],
                vertices[self._c_idx(idx_start - 2)]]),
        ]
        return [Simplex(ver, boundary_type='none') for ver in vertices_tri]

    def _c_idx(self, idx):
        """Get cyclic indices."""
        idx_max = 5
        if idx >= idx_max:
            idx = idx - idx_max
        elif idx < 0:
            idx = idx + idx_max
        return idx

    def _validiate(self, vertices):
        vertices = np.asarray(vertices).copy()
        assert len(vertices) == 5
        return vertices
