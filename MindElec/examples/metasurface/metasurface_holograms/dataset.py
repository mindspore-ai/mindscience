"""
dataset module
"""
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision

IMAGE_SIZE = 40


def create_dataset_mnist(batch_size, root, num_parallel_workers=None, shuffle=True):
    """
    create mnist dataset
    Args:
        batch_size: batch size
        root: path root
        num_parallel_workers: number of parallel workers
        shuffle: whether shuffle

    Returns: mindspore mnist dataset

    """
    data_set = ds.ImageFolderDataset(root, num_parallel_workers=num_parallel_workers, shuffle=shuffle)
    transform_img = [
        vision.Decode(),
        vision.Resize(IMAGE_SIZE),
        vision.CenterCrop(IMAGE_SIZE),
        vision.HWC2CHW()
    ]

    # 数据映射操作
    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers, operations=transform_img,
                            output_columns="image")
    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers,
                            operations=lambda x: x.astype("float32"))

    # 批量操作
    data_set = data_set.batch(batch_size)

    return data_set
