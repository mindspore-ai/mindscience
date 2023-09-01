"""
main module
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import mindspore
from mindspore import context, ops, nn, Tensor
from mindspore.ops import operations as P
from mindspore.train import amp

from dataset import create_dataset_mnist
from wgan_gradient_penalty import Discriminator, Generator


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Mindspore implementation of GAN models.")
    parser.add_argument('--model', type=str, default='cWGAN-GP', choices=['GAN', 'DCGAN', 'WGAN-CP', 'cWGAN-GP'])
    parser.add_argument('--mode', type=str, default=context.GRAPH_MODE)
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--device_target', type=int, default=None)
    parser.add_argument('--data_path', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--train_data_path', type=str, default='./data/MNIST/binary_images/trainimages',
                        help='path to dataset')
    parser.add_argument('--test_data_path', type=str, default='./data/MNIST/binary_images/testimages',
                        help='path to dataset')
    parser.add_argument('--valid_data_path', type=str, default='./data/MNIST/binary_images/validimages',
                        help='path to dataset')
    parser.add_argument('--log_folder_path', type=str, default='./logs', help='path to logs')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                        help='The name of dataset')
    parser.add_argument('--Gen_learning_rate', type=float, default=3e-5, help='The learning rate of Generator')
    parser.add_argument('--Dis_learning_rate', type=float, default=3e-5, help='The learning rate of Discriminator')
    parser.add_argument('--save_per_times', type=int, default=50, help='Save model per generator update times')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--Dis_per_train', type=int, default=1, help='Train Discriminator times per training iteration')
    parser.add_argument('--Gen_per_train', type=int, default=3, help='Train Generator times per training iteration')
    parser.add_argument('--train_iters', type=int, default=5001,
                        help='The number of iterations for training in WGAN model.')
    parser.add_argument("--download_data", type=str, default="metasurface_holograms", help="Project name of remote "
                                                                                           "data")
    parser.add_argument("--force_download", type=bool, default=False, help="Whether forcely download data in sciai")
    args = parser.parse_args()
    return args


def write_log(log_file_name, log_message):
    """
    write log into log file
    Args:
        log_file_name: log file name
        log_message: log message

    Returns: None

    """
    print(log_message)
    with open(log_file_name, mode='a') as txt_file:
        txt_file.write(log_message + "\n")


class GradNetWrtX(nn.Cell):
    """
    first grad of given net
    """

    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x)


class GLoss(nn.Cell):
    """连接生成器和损失"""

    def __init__(self, net_d, net_g):
        super(GLoss, self).__init__(auto_prefix=True)
        self.net_d = net_d
        self.net_g = net_g
        self.concat = P.Concat(axis=1)
        self.mse = nn.MSELoss()

    def construct(self, real_img_in):
        """构建生成器损失计算结构"""
        fake_img = self.net_g(real_img_in)
        real_fake_imgs = self.concat((real_img_in, fake_img))
        generator_loss = self.net_d(real_fake_imgs)
        generator_loss = generator_loss.mean()
        return -generator_loss


class GMse(nn.Cell):
    """连接生成器和损失"""

    def __init__(self, net_d, net_g):
        super(GMse, self).__init__(auto_prefix=True)
        self.net_d = net_d
        self.net_g = net_g
        self.concat = P.Concat(axis=1)
        self.mse = nn.MSELoss()

    def construct(self, real_img_in):
        """构建生成器损失计算结构"""
        fake_img = self.net_g(real_img_in)
        generator_mse_loss = self.mse(fake_img, real_img_in)
        return generator_mse_loss


class DLossReal(nn.Cell):
    """连接判别器和损失"""

    def __init__(self, net_d, net_g):
        super(DLossReal, self).__init__(auto_prefix=True)
        self.net_d = net_d
        self.net_g = net_g
        self.concat = P.Concat(axis=1)

    def construct(self, real_img_in):
        """构建判别器损失计算结构"""
        real_imgs = self.concat((real_img_in, real_img_in))
        discriminator_loss_real = self.net_d(real_imgs)
        discriminator_loss_real = discriminator_loss_real.mean()
        return -discriminator_loss_real


class DLossFake(nn.Cell):
    """连接判别器和损失"""

    def __init__(self, net_d, net_g):
        super(DLossFake, self).__init__()
        self.net_d = net_d
        self.net_g = net_g
        self.concat = P.Concat(axis=1)

    def construct(self, real_img_in):
        """构建判别器损失计算结构"""
        fake_img = self.net_g(real_img_in)
        fake_img = ops.stop_gradient(fake_img)
        real_fake_imgs = self.concat((real_img_in, fake_img))

        discriminator_loss_fake = self.net_d(real_fake_imgs)
        discriminator_loss_fake = discriminator_loss_fake.mean()
        return discriminator_loss_fake


class GradientPenaltyCell(nn.Cell):
    """连接判别器梯度和损失"""

    def __init__(self, net_d, net_g, batch_size):
        super(GradientPenaltyCell, self).__init__()
        self.net_d = net_d
        self.net_g = net_g

        self.concat = P.Concat(axis=1)
        self.mean = P.ReduceMean()
        self.norm = nn.Norm(axis=1)
        self.lambda_term = 10

        minvalue = Tensor(0, mindspore.float32)
        maxvalue = Tensor(1, mindspore.float32)
        self.eta = mindspore.ops.uniform((batch_size, 1, 1, 1), minvalue, maxvalue, dtype=mindspore.float32).astype(
            mindspore.float16)
        self.gradx = GradNetWrtX(self.net_d)

    def construct(self, real_img_in):
        """构建判别器梯度损失计算结构"""
        fake_img = self.net_g(real_img_in)
        fake_img = ops.stop_gradient(fake_img)

        interpolated = self.eta * real_img_in + ((1 - self.eta) * fake_img)  # [batch 1 40 40]
        interpolated = self.concat((real_img_in, interpolated))  # [batch 2 40 40]

        # calculate gradients of probabilities with respect to examples
        gradients = self.gradx(interpolated)
        grad_penalty = self.mean(((self.norm(gradients) - 1) ** 2)) * self.lambda_term
        return grad_penalty


def validate(valid_imgs, net_g, to_save_path, train_iter, start_time_format):
    """
    validate and save result into image
    Args:
        valid_imgs: image to validate
        net_g: generator net
        to_save_path: the path to save png
        train_iter: training iteration number
        start_time_format: start time in formatted string

    Returns:

    """
    gen_valid = net_g(valid_imgs)
    mse = nn.MSELoss()
    valid_mse_loss = mse(valid_imgs, gen_valid)
    print('mse loss: ', valid_mse_loss)
    file_name = start_time_format + '_with_pena_ln_iter_{}.png'.format(train_iter)
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
        img = squeeze(imgs[i - 1]).astype(mindspore.float32).asnumpy()
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


def train(args):
    """train"""
    os.makedirs(args.log_folder_path, exist_ok=True)
    rec_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    txt_file_name = os.path.join(args.log_folder_path, rec_time + "_with_pena.log")

    # WGAN values from paper
    g_learning_rate = args.Gen_learning_rate
    d_learning_rate = args.Dis_learning_rate
    b1 = 0.5
    b2 = 0.999
    batch_size = args.batch_size
    # 把不训练的参数去掉

    # # Set the logger
    logger_num = int(args.train_iters / args.save_per_times) + 1
    print('logger_num:', logger_num)

    # 计算loss必须用继承nn.Cell的类，在construct方法中计算
    # construct输入为data, label

    generator = Generator(args.data_path)
    discriminator = Discriminator(channels=2)

    g_loss_cell = GLoss(discriminator, generator)
    g_mse_cell = GMse(discriminator, generator)
    d_loss_real_cell = DLossReal(discriminator, generator)
    d_loss_fake_cell = DLossFake(discriminator, generator)
    gradient_penalty_cell = GradientPenaltyCell(discriminator, generator, batch_size)

    d_optimizer = nn.Adam(discriminator.trainable_params(), learning_rate=d_learning_rate, beta1=b1, beta2=b2)
    g_optimizer = nn.Adam(generator.trainable_params(), learning_rate=g_learning_rate, beta1=b1, beta2=b2)
    g_loss_step = amp.build_train_network(g_loss_cell, g_optimizer, level="O0")
    g_mse_step = amp.build_train_network(g_mse_cell, g_optimizer, level="O0")
    d_real_step = amp.build_train_network(d_loss_real_cell, d_optimizer, level="O0")
    d_fake_step = amp.build_train_network(d_loss_fake_cell, d_optimizer, level="O0")
    gradient_penalty_step = amp.build_train_network(gradient_penalty_cell, d_optimizer, level="O0")

    train_dataset = create_dataset_mnist(batch_size, args.train_data_path)
    valid_dataset = create_dataset_mnist(batch_size, args.valid_data_path, shuffle=False)

    train_dataloader = get_infinite_batches(train_dataset.create_dict_iterator())
    valid_dataloader = get_infinite_batches(valid_dataset.create_dict_iterator())

    # 开始循环训练
    print("Starting Training Loop...")

    start_message = 'output loss, with penalty'
    write_log(txt_file_name, start_message)

    expand_dims = ops.ExpandDims()

    # save original validation images
    val_img = valid_dataloader.__next__()  # (bs, 1, 40, 40)
    val_img = expand_dims(val_img[:, 0, :, :], 1)

    save_path = os.path.join(os.getcwd(), 'results/training_result_images/')
    save_tensor_imgs(val_img, save_path, 'orignal_val_img.png')

    discriminator.set_train()
    generator.set_train()

    start_iter = 0
    start_time = time.time()
    last_time = start_time
    for t_iter in range(start_iter, start_iter + args.train_iters):
        for _ in range(args.Dis_per_train):
            real_img = train_dataloader.__next__()
            real_img = real_img / 255.
            real_img = expand_dims(real_img[:, 0, :, :], 1)
            d_loss_real = d_real_step(real_img)
            d_loss_fake = d_fake_step(real_img)
            gradient_penalty_loss = gradient_penalty_step(real_img)

        for _ in range(args.Gen_per_train):
            real_img = train_dataloader.__next__()
            real_img = real_img / 255.
            real_img = expand_dims(real_img[:, 0, :, :], 1)
            g_loss = g_loss_step(real_img)
            mse_loss = g_mse_step(real_img)
        this_time = time.time()
        time_diff = this_time - last_time
        total_time = this_time - start_time
        last_time = this_time
        message = '[%d/%d iters]\tLoss_D_real: %.4f\tLoss_D_fake: %.4f\tLoss_Gradient: %.4f\tLoss_G: %.4f\t' \
                  'mse_G: %.4f\tstep time: %.4f\ttotal time: %.4f' \
                  % (t_iter, args.train_iters, d_loss_real.asnumpy(), d_loss_fake.asnumpy(),
                     gradient_penalty_loss.asnumpy(), g_loss.asnumpy(), mse_loss.asnumpy(), time_diff, total_time)
        write_log(txt_file_name, message)

        if t_iter % args.save_per_times == 0:
            val_img = valid_dataloader.__next__()  # (bs, 1, 40, 40)
            val_img = val_img / 255.
            val_img = expand_dims(val_img[:, 0, :, :], 1)
            save_path = os.path.join(os.getcwd(), 'results/training_result_images/')
            validate(val_img, generator, save_path, t_iter, rec_time)


if __name__ == "__main__":
    args_ = parse_args()
    # 1. loss: output loss, with penalty, with mse
    mindspore.set_seed(1234)
    if args_.device_target is not None:
        context.set_context(device_target=args_.device_target)
    if args_.device_id is not None:
        context.set_context(device_target=args_.device_id)
    context.set_context(mode=args_.mode)
    train(args_)
