"""
utils module
"""
import os
import matplotlib.pyplot as plt
from mindspore import ops, Tensor


def write_log(txt_file_name, message):
    print(message)
    with os.fdopen(txt_file_name, 'a') as txt_file:
        txt_file.write(message + "\n")


def save_tensor_imgs(imgs, save_path, file_name):
    """
    save tensor as png file
    Args:
        imgs: image
        save_path: path to save
        file_name: file name

    Returns:

    """
    squeeze = ops.Squeeze(0)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgs_num = imgs.shape[0]

    fig = plt.figure(figsize=(8, 8))
    columns = 8
    rows = 8
    for i in range(1, imgs_num + 1):
        img = squeeze(imgs[i - 1]).asnumpy()
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    plt.savefig(save_path + file_name)


def get_infinite_batches(data_loader):
    while True:
        for d in data_loader:
            img = Tensor(d['image'])
            yield img
