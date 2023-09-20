# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""loss functions for dgmr"""
from mindspore import nn, ops


def loss_hinge_disc(score_generated, score_real):
    r"""
    Discriminator hinge loss.
    .. math::
        L_D(\phi)=E[ReLU(1-D_{\phi}(X))+ReLU(1+D_{\phi}(G(Z)))]
    """
    l1 = ops.relu(1.0 - score_real)
    op = ops.ReduceMean()
    loss = op(l1)
    l2 = ops.relu(1.0 + score_generated)
    loss += op(l2)
    return loss


def loss_hinge_gen(score_generated):
    """Generator hinge loss."""
    op = ops.ReduceMean()
    loss = -op(score_generated)
    return loss


def grid_cell_regularizer(generated_samples, batch_targets):
    r"""
    Grid cell regularizer.
    .. math::
        L_R = \frac{1}{HWN}\mid\mid(E_Z[G_{\theta}(Z)]-X_{M+1:M+N})\bigodot\omega(X_{M+1:M+N})\mid\mid_1
    Args:
      generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
      batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

    Returns:
      loss: A tensor of shape [batch_size].
    """
    op = ops.ReduceMean()
    gen_mean = op(generated_samples, 0)
    loss = nn.MSELoss(reduction='mean')
    loss = loss(gen_mean, batch_targets)
    return loss


class GenWithLossCell(nn.Cell):
    r"""Generator loss cell"""
    def __init__(self, generator, discriminator, generation_steps, grid_lambda):
        super(GenWithLossCell, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.generation_steps = generation_steps
        self.grid_lambda = grid_lambda
        self.concat_op1 = ops.Concat(axis=1)
        self.concat_op2 = ops.Concat(axis=0)
        self.split = ops.Split(0, 2)

    def construct(self, images, future_images):
        """GenWithLossCell construct function"""
        predictions = [self.generator(images) for _ in range(self.generation_steps)]

        grid_cell_reg = grid_cell_regularizer(ops.stack(predictions, 0), future_images)

        generated_sequence = [self.concat_op1((images, x)) for x in predictions]
        real_sequence = self.concat_op1((images, future_images))
        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = self.concat_op2((real_sequence, g_seq))
            concatenated_outputs = self.discriminator(concatenated_inputs)
            # Split along the concatenated dimension, as discrimnator concatenates along dim=1
            _, score_generated = self.split(concatenated_outputs)
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(self.concat_op2(generated_scores))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg
        return generator_loss


class DiscWithLossCell(nn.Cell):
    r"""Discriminator loss cell"""
    def __init__(self, generator, discriminator):
        super(DiscWithLossCell, self).__init__()

        self.dis_loss = loss_hinge_disc
        self.generator = generator
        self.discriminator = discriminator
        self.concat_op1 = ops.Concat(axis=1)
        self.concat_op2 = ops.Concat(axis=0)
        self.split1 = ops.Split(0, 2)
        self.split2 = ops.Split(1, 2)

    def construct(self, images, future_images):
        """DiscWithLossCell construct function"""
        predictions = self.generator(images)

        generated_sequence = self.concat_op1((images, predictions))
        real_sequence = self.concat_op1((images, future_images))

        concatenated_inputs = self.concat_op2((real_sequence, generated_sequence))
        concatenated_outputs = self.discriminator(concatenated_inputs)

        score_real, score_generated = self.split1(concatenated_outputs)

        score_real_spatial, score_real_temporal = self.split2(score_real)
        score_generated_spatial, score_generated_temporal = self.split2(score_generated)
        discriminator_loss = loss_hinge_disc(
            score_generated_spatial, score_real_spatial
        ) + loss_hinge_disc(score_generated_temporal, score_real_temporal)

        return discriminator_loss
