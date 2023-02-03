"""
argument parser module
"""
import argparse


def parse_args():
    """
    arguments parser
    Returns: args parser
    """
    parser = argparse.ArgumentParser(description="Mindspore implementation of GAN models.")

    parser.add_argument('--model', type=str, default='cWGAN-GP', choices=['GAN', 'DCGAN', 'WGAN-CP', 'cWGAN-GP'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataroot', type=str, default='../MNIST/binary_images', help='path to dataset')
    parser.add_argument('--train_dataroot', type=str, default='./MNIST/binary_images/trainimages',
                        help='path to dataset')
    parser.add_argument('--test_dataroot', type=str, default='./MNIST/binary_images/testimages', help='path to dataset')
    parser.add_argument('--valid_dataroot', type=str, default='./MNIST/binary_images/validimages',
                        help='path to dataset')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                        help='The name of dataset')
    parser.add_argument('--Gen_learning_rate', type=float, default=3e-5, help='The learning rate of Generator')
    parser.add_argument('--Dis_learning_rate', type=float, default=3e-5, help='The learning rate of Discriminator')
    parser.add_argument('--save_per_times', type=int, default=50, help='Save model per generator update times')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--Dis_per_train', type=int, default=1, help='Train Discriminator times per training iteration')
    parser.add_argument('--Gen_per_train', type=int, default=3, help='Train Generator times per training iteration')
    parser.add_argument('--train_iters', type=int, default=2001,
                        help='The number of iterations for training in WGAN model.')

    args = parser.parse_args()

    return args
