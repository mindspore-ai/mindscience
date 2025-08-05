"""Polygon shapes."""
import numpy as np

from .shapes import Union, derive_union_boundary
from .simplex import Segment, Simplex


class Polygon(Union):
    """Polygon.

    Assume that the adjacent input vertices form en edge of the Polygon. The
    code does not check the case that three points lie on a line.
    """

    def __init__(self, vertices, boundary_type="none"):
        vertices = self._validiate(vertices)
        self._vertices = vertices
        shapes = self._create_triangles()
        segments = [
            Segment([vertices[idx], vertices[(idx + 1) % len(vertices)]])
            for idx in range(len(vertices))
        ]
        boundary = derive_union_boundary(segments, boundary_type)
        super(Polygon, self).__init__(shapes, boundary)

    def _create_triangles(self):
        """Create triangles from the vertices."""

        def is_ear(triangle: Simplex, vertices):
            # Check if the triangle is an ear of the polygon.
            # ref: https://en.wikipedia.org/wiki/Polygon_triangulation
            return np.all(triangle.is_inside(vertices) != 1)

        def is_convex(vertices):
            """Check if the given triangle is convex."""
            x0 = vertices[1]
            return np.cross(vertices[0] - x0, vertices[2] - x0) > 0

        vertices = self._vertices.copy()
        triangles = []
        while len(vertices) > 3:
            for i, vertex in enumerate(vertices):
                try:
                    triangle_vertices = [
                        vertices[i - 1],
                        vertex,
                        vertices[(i + 1) % len(vertices)],
                    ]
                    if not is_convex(triangle_vertices):
                        continue
                    triangle = Simplex(
                        triangle_vertices,
                        "uniform",
                    )
                    if is_ear(triangle, vertices):
                        triangles.append(triangle)
                        vertices = np.delete(vertices, i, axis=0)
                        break
                except ValueError:  # Three points lie on a line
                    pass
            else:
                raise RuntimeError("Cannot find an ear.")
        triangles.append(Simplex(vertices))
        return triangles

    def _validiate(self, vertices):
        vertices = np.asarray(vertices).copy()
        if vertices.ndim != 2:
            raise ValueError("vertices should be a 2D array.")
        if vertices.shape[1] != 2:
            raise ValueError("vertices should have two columns.")
        return vertices
