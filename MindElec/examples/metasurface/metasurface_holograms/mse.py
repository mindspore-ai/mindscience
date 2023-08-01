"""
mse module
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
from mindspore import context, ops, nn, Tensor

from dataset import create_dataset_mnist
from wgan_gradient_penalty import Generator

parser = argparse.ArgumentParser(description="Mindspore implementation of GAN models.")

parser.add_argument('--model', type=str, default='cWGAN-GP', choices=['GAN', 'DCGAN', 'WGAN-CP', 'cWGAN-GP'])
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--is_train', type=str, default='False')
parser.add_argument('--is_finetune', type=str, default='True',
                    help='False: Train from scratch; True: Train from a saved checkpint(load_D, load_G)')
parser.add_argument('--is_evaluate', type=str, default='False',
                    help='True: Will test the images loaded from test_data_path.')
parser.add_argument('--is_test_single_image', type=str, default='True',
                    help='True: Will test the single image from single_image_path')

parser.add_argument('--data_path', type=str, default='./data/', help='path to dataset')
parser.add_argument('--train_data_path', type=str, default='./data/MNIST/binary_images/trainimages',
                    help='path to dataset')
parser.add_argument('--test_data_path', type=str, default='./data/MNIST/binary_images/testimages',
                    help='path to dataset')
parser.add_argument('--valid_data_path', type=str, default='./data/MNIST/binary_images/validimages',
                    help='path to dataset')
parser.add_argument('--single_image_path', type=str,
                    default='./MNIST/binary_images/testimages/9/38.png',
                    help='The image path when is_test_single_image==True')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                    help='The name of dataset')
parser.add_argument('--download', type=str, default='False')
parser.add_argument('--epochs', type=int, default=200, help='The number of epochs to run')
parser.add_argument('--Gen_learning_rate', type=float, default=3e-5, help='The learning rate of Generator')
parser.add_argument('--Dis_learning_rate', type=float, default=3e-5, help='The learning rate of Discriminator')
parser.add_argument('--save_per_times', type=int, default=500, help='Save model per generator update times')
parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
parser.add_argument('--Dis_per_train', type=int, default=1, help='Train Discriminator times per training iteration')
parser.add_argument('--Gen_per_train', type=int, default=3, help='Train Generator times per training iteration')

parser.add_argument('--train_iters', type=int, default=2001,
                    help='The number of iterations for training in WGAN model.')

args = parser.parse_args()

# 1. loss: output loss, without penalty, with mse

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=args.device_id)  # PYNATIVE_MODE  GRAPH_MODE
# , save_graphs = True, save_graphs_path = './simple_net_mse_graph'

LOGS_FOLDER = "./logs/"
rec_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
TXT_FILE_NAME = LOGS_FOLDER + "_simple_net_only_mse.txt"  # + rec_time

# WGAN values from paper
g_learning_rate = args.Gen_learning_rate
d_learning_rate = args.Dis_learning_rate
B1 = 0.5
B2 = 0.999
batch_size = args.batch_size
# 把不训练的参数去掉

# # Set the logger
logger_num = int(args.train_iters / args.save_per_times) + 1
print('logger_num:', logger_num)

ones = ops.Ones()
zeros = ops.Zeros()


# 计算loss必须用继承nn.Cell的类，在construct方法中计算
# G的adv loss和mse loss都算了再更新参数

def write_log(log_file_name, log_message):
    print(log_message)
    with os.fdopen(log_file_name, 'a') as txt_file:
        txt_file.write(log_message + "\n")


class GLoss(nn.Cell):
    """连接生成器和损失"""

    def __init__(self, net_g):
        super(GLoss, self).__init__(auto_prefix=True)
        self.net_g = net_g
        self.mse = nn.MSELoss()

    def construct(self, real_img_in):
        """构建生成器损失计算结构"""
        fake_img = self.net_g(real_img_in)
        mse_loss = self.mse(fake_img, real_img_in)
        return mse_loss


def validate(valid_imgs, net_g, to_save_path, train_iter):
    """
    validate and save result into image
    Args:
        valid_imgs: image to validate
        net_g: generator net
        to_save_path: the path to save png
        train_iter: training iteration number

    Returns:

    """
    gen_valid = net_g(valid_imgs)
    mse = nn.MSELoss()
    valid_mse_loss = mse(valid_imgs, gen_valid)
    print('mse loss: ', valid_mse_loss)
    file_name = '_simple_net_only_mse_iter_{}.png'.format(train_iter)  # rec_time +
    save_tensor_imgs(gen_valid, to_save_path, file_name)


def save_tensor_imgs(imgs, to_save_path, file_name):
    """
    save tensor as image
    Args:
        imgs: tensor of image
        to_save_path: path to save png
        file_name: save file name

    Returns:

    """
    squeeze = ops.Squeeze(0)
    if not os.path.exists(to_save_path):
        os.makedirs(to_save_path)

    imgs_num = imgs.shape[0]

    fig = plt.figure(figsize=(8, 8))
    columns = 8
    rows = 8
    for i in range(1, imgs_num + 1):
        img = squeeze(imgs[i - 1]).asnumpy()
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    plt.savefig(to_save_path + file_name)


def get_infinite_batches(data_loader):
    """
    return infinite batch of image
    Args:
        data_loader: data loader

    Returns:

    """
    while True:
        for d in data_loader:
            img = Tensor(d['image'])
            yield img


generator = Generator(args.data_path)  # _trainable _trainable_sqrt

g_loss_cell = GLoss(generator)

g_optimizer = nn.Adam(filter(lambda p: p.requires_grad, generator.trainable_params()), learning_rate=g_learning_rate,
                      beta1=B1, beta2=B2)

# 实例化TrainOneStepCell
g_loss_step = nn.TrainOneStepCell(g_loss_cell, g_optimizer)  # TrainOneStep

train_dataset = create_dataset_mnist(args.batch_size, args.train_data_path)
valid_dataset = create_dataset_mnist(args.batch_size, args.valid_data_path, shuffle=False)

train_dataloader = get_infinite_batches(train_dataset.create_dict_iterator())
valid_dataloader = get_infinite_batches(valid_dataset.create_dict_iterator())

# 开始循环训练
print("Starting Training Loop...")

MESSAGE = 'output loss, only mse'
write_log(TXT_FILE_NAME, MESSAGE)

expand_dims = ops.ExpandDims()

# save original validation images
val_img = valid_dataloader.__next__()  # (bs, 1, 40, 40)
val_img = expand_dims(val_img[:, 0, :, :], 1)

save_path = os.path.join(os.getcwd(), 'results/training_result_images/')
save_tensor_imgs(val_img, save_path, 'orignal_val_img.png')

generator.set_train()

START_ITER = 0
for t_iter in range(START_ITER, START_ITER + args.train_iters):

    real_img = train_dataloader.__next__()
    real_img = real_img / 255.
    real_img = expand_dims(real_img[:, 0, :, :], 1)
    g_loss = g_loss_step(real_img)

    MESSAGE = '[%d/%d iters]\tLoss_G: %.4f' % (t_iter, args.train_iters, g_loss.asnumpy())
    write_log(TXT_FILE_NAME, MESSAGE)

    if t_iter % args.save_per_times == 0:
        val_img = valid_dataloader.__next__()  # (bs, 1, 40, 40)
        val_img = expand_dims(val_img[:, 0, :, :], 1)
        save_path = os.path.join(os.getcwd(), 'results/training_result_images/')
        validate(val_img, generator, save_path, t_iter)
