"""Create dataset."""
from mindflow.data import Dataset
from mindflow.geometry import (
    Rectangle, Disk, Triangle, Pentagon,
    Cylinder, Cone, Tetrahedron,
    generate_sampling_config,
)


shape_factory = {
    "rectangle": Rectangle,
    "disk": Disk,
    "triangle": Triangle,
    "pentagon": Pentagon,
    "cylinder": Cylinder,
    "cone": Cone,
    "tetrahedron": Tetrahedron,
}


def create_dataset(geom_name, config):
    """Create dataset."""
    sampling_config = generate_sampling_config(config['data'])
    try:
        cls_shape = shape_factory[geom_name]
    except KeyError:
        raise ValueError("Wrong geometry name.")
    region = cls_shape(geom_name, **config['geometry'][geom_name], sampling_config=sampling_config)
    dataset = Dataset({region: ['domain', 'BC']})
    ds_create = dataset.create_dataset(
        batch_size=config['batch_size'], shuffle=True, prebatched_data=True, drop_remainder=True
    )
    steps_per_epoch = config['data']['domain']['size']//config['batch_size']
    return (dataset, ds_create), (steps_per_epoch, region.dim)
