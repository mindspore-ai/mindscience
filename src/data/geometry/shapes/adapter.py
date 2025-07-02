"""Adapter."""
import numpy as np

from .. import geometry_base


class Geometry(geometry_base.Geometry):
    r"""
    Adapter class for using the ``shapes`` module.

    Args:
        shape (Shape): an object for sampling the points.
        name (str): name of the geometry.
        dim (int): number of dimensions.
        coord_min (Union[int, float, list[int, float], tuple[int, float], numpy.ndarray]):
            minimal coordinate of the geometry.
        coord_max (Union[int, float, list[int, float], tuple[int, float], numpy.ndarray]):
            maximal coordinate of the geometry.
        dtype (numpy.dtype): Data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, shape, name, dim, coord_min, coord_max,
                 dtype=np.float32, sampling_config=None):
        self._shape = shape
        self.columns_dict = {}
        super(Geometry, self).__init__(name, dim, coord_min, coord_max, dtype, sampling_config)

    def sampling(self, geom_type="domain"):
        config = self.sampling_config
        if geom_type.lower() == "domain":
            self.columns_dict["domain"] = [self.name + "_domain_points"]
            if config.domain.random_sampling:
                data = self._shape.sample(config.domain.size).astype(self.dtype)
            else:
                raise NotImplementedError("Sampling grid points is not implemented.""")
        elif geom_type.lower() == "bc":
            self.columns_dict["BC"] = [self.name + "_BC_points"]
            if config.bc.random_sampling:
                data = self._shape.sample_boundary(config.bc.size).astype(self.dtype)
            else:
                raise NotImplementedError("Sampling grid points is not implemented.""")
        else:
            raise ValueError
        return data

    def _inside(self, points, strict=False):
        cond = self._shape.is_inside(points)
        if strict:
            cond |= self._on_boundary(points)
        return cond

    def _on_boundary(self, points):
        return self._shape.is_on_boundary(points)

    def _boundary_normal(self, points):
        raise NotImplementedError("{}._boundary_normal not implemented".format(self.geom_type))
