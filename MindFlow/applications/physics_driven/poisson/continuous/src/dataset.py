"""Create dataset."""
from copy import deepcopy

from mindflow.data import Dataset
from mindflow.geometry import (
    Interval,
    Rectangle,
    Disk,
    Triangle,
    Pentagon,
    Polygon,
    Cylinder,
    Cone,
    Tetrahedron,
    generate_sampling_config,
)


shape_factory = {
    "interval": Interval,
    "rectangle": Rectangle,
    "disk": Disk,
    "triangle": Triangle,
    "pentagon": Pentagon,
    "polygon": Polygon,
    "cylinder": Cylinder,
    "cone": Cone,
    "tetrahedron": Tetrahedron,
}


def create_dataset(geom_name, config, n_samps=None):
    """Create dataset."""
    if n_samps is not None:
        config = deepcopy(config)
        config["data"]["domain"]["size"] = n_samps
        config["data"]["BC"]["size"] = n_samps
        config["data"]["train"]["batch_size"] = n_samps
    sampling_config = generate_sampling_config(config["data"])
    try:
        cls_shape = shape_factory[geom_name]
    except KeyError:
        raise ValueError("Wrong geometry name.")
    region = cls_shape(
        geom_name, **config["geometry"][geom_name], sampling_config=sampling_config
    )
    dataset = Dataset({region: ["domain", "BC"]})
    ds_create = dataset.create_dataset(
        batch_size=config["data"]["train"]["batch_size"],
        shuffle=True,
        prebatched_data=True,
        drop_remainder=True,
    )
    return ds_create, region.dim
