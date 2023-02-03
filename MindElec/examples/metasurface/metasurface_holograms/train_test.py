"""
train test module
"""
import mindspore
from mindspore import ops, nn, Tensor
from mindspore.ops import operations as P

from utils import write_log, save_tensor_imgs


class GradNetWrtX(nn.Cell):
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
        return generator_loss


class GMse(nn.Cell):
    """连接生成器和损失"""

    def __init__(self, net_d, net_g):
        super(GMse, self).__init__(auto_prefix=True)
        self.net_d = net_d
        self.net_g = net_g
        self.concat = P.Concat(axis=1)
        self.mse = nn.MSELoss()

    def construct(self, real_img):
        """构建生成器损失计算结构"""
        fake_img = self.net_g(real_img)
        mse_loss = self.mse(fake_img, real_img)
        return mse_loss


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
        return discriminator_loss_real


class DLossFake(nn.Cell):
    """连接判别器和损失"""

    def __init__(self, net_d, net_g):
        super(DLossFake, self).__init__(auto_prefix=True)
        self.net_d = net_d
        self.net_g = net_g

        self.concat = P.Concat(axis=1)
        self.mean = P.ReduceMean()
        self.norm = nn.Norm(axis=1)
        self.lambda_term = 10

        self.minval = Tensor(0, mindspore.float32)
        self.maxval = Tensor(1, mindspore.float32)

    def construct(self, real_img_in):
        """构建判别器损失计算结构"""
        fake_img = self.net_g(real_img_in)
        fake_img = ops.stop_gradient(fake_img)
        real_fake_imgs = self.concat((real_img_in, fake_img))

        d_loss_fake = self.net_d(real_fake_imgs)
        d_loss_fake = d_loss_fake.mean()

        return d_loss_fake

    def calculate_gradient_penalty(self, real_imgs, fake_imgs):
        """
        calculate gradient penalty
        Args:
            real_imgs: real imagetensor
            fake_imgs: fake image tensor

        Returns: gradient penalty

        """
        batch_size = real_imgs.shape[0]
        eta = mindspore.ops.uniform((batch_size, 1, 1, 1), self.minval, self.maxval, dtype=mindspore.float32)

        interpolated = eta * real_imgs + ((1 - eta) * fake_imgs)  # [batch 1 40 40]
        interpolated = self.concat((real_imgs, interpolated))  # [batch 2 40 40]

        # calculate gradients of probabilities with respect to examples
        gradients = GradNetWrtX(self.net_d)(interpolated)[0]
        grad_penalty = self.mean(((self.norm(gradients) - 1) ** 2)) * self.lambda_term

        return grad_penalty


def validate(valid_imgs, net_g, save_path, t_iter, rec_time, txt_file_name):
    """
    validate and save result into image
    Args:
        valid_imgs: image to validate
        net_g: generator net
        save_path: the path to save png
        t_iter: training iteration number
        rec_time: start time in formatted string
        txt_file_name: log file name

    Returns: None

    """
    gen_valid = net_g(valid_imgs)
    mse = nn.MSELoss()
    for i in range(valid_imgs.shape[0]):
        mse_loss_i = mse(valid_imgs[i], gen_valid[i])
        message = 'MSE loss %d: %.5f' % (i, mse_loss_i.asnumpy())
        write_log(txt_file_name, message)

    file_name = rec_time + '_with_pena_ln_iter_{}.png'.format(t_iter)
    save_tensor_imgs(gen_valid, save_path, file_name)
    mindspore.save_checkpoint(net_g, "./saved_models/" + rec_time + "net_G_t" + str(t_iter) + ".ckpt")
