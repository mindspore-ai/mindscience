# Copyright 2025 Huawei Technologies Co., Ltd
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
# ============================================================================
"""diffusion ST testcase"""
import time
import pytest

import numpy as np
from mindspore import Tensor, ops, amp, nn, jit
from mindspore import dtype as mstype

from mindflow.cell import DiffusionTransformer, DiffusionTrainer, DDPMScheduler, DDIMScheduler, DDPMPipeline, \
    DDIMPipeline, ConditionDiffusionTransformer
from dataset import get_latent_dataset
from ae import LATENT_DIM, generate_image


def count_params(model):
    """count_params"""
    count = 0
    for p in model.trainable_params():
        t = 1
        for i in range(len(p.shape)):
            t = t * p.shape[i]
        count += t
    print(model)
    print(f'model params: {count}')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_train_diffusion(diffusion_type='ddpm', latent_path='latent.npy', cond_path='labels.npy'):
    """
    Feature: diffusion model train
    Description: test DDPM/DDIM model generate MNIST image latent
    Expectation: success
    """
    compute_dtype = mstype.float32
    in_dim = LATENT_DIM

    model = DiffusionTransformer(in_channels=in_dim,
                                 out_channels=in_dim,
                                 hidden_channels=256,
                                 layers=1,
                                 heads=8,
                                 time_token_cond=True,
                                 compute_dtype=compute_dtype)
    count_params(model)
    num_train_timesteps = 500
    batch_size = 512
    infer_bs = 10
    if diffusion_type == 'ddpm':
        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
                                  beta_start=0.0001,
                                  beta_end=0.02,
                                  beta_schedule="squaredcos_cap_v2",
                                  clip_sample_range=1.0,
                                  thresholding=False,
                                  dynamic_thresholding_ratio=None,
                                  rescale_betas_zero_snr=False,
                                  timestep_spacing="leading",
                                  compute_dtype=compute_dtype)
        pipe = DDPMPipeline(model=model, scheduler=scheduler, batch_size=infer_bs,
                            seq_len=1, num_inference_steps=num_train_timesteps)
    else:
        scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps,
                                  beta_start=0.0001,
                                  beta_end=0.02,
                                  beta_schedule="squaredcos_cap_v2",
                                  clip_sample_range=1.0,
                                  thresholding=False,
                                  dynamic_thresholding_ratio=None,
                                  rescale_betas_zero_snr=False,
                                  timestep_spacing="leading",
                                  compute_dtype=compute_dtype)
        num_inference_steps = 50
        pipe = DDIMPipeline(model=model, scheduler=scheduler, batch_size=infer_bs,
                            seq_len=1, num_inference_steps=num_inference_steps)

    trainer = DiffusionTrainer(model,
                               scheduler,
                               objective='pred_noise',
                               p2_loss_weight_gamma=0,
                               p2_loss_weight_k=1,
                               loss_type='l2')

    def forward_fn(data, t, noise):
        loss = trainer.get_loss(data, noise, t)
        return loss
    lr = 0.0002
    optimizer = nn.Adam(model.trainable_params(), lr, 0.9, 0.99)
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    @jit
    def train_step(data, t, noise):
        loss, grads = grad_fn(data, t, noise)
        is_finite = amp.all_finite(grads)
        if is_finite:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    train_dataset = get_latent_dataset(
        latent_path=latent_path, cond_path=cond_path, batch_size=batch_size)
    epochs = 20
    print(f'scheduler.num_timesteps: {scheduler.num_timesteps}')

    for epoch in range(1, epochs+1):
        time_beg = time.time()
        for samples, _ in train_dataset:
            timesteps = Tensor(np.random.randint(
                0, num_train_timesteps, (samples.shape[0],)).astype(np.int32))
            noise = Tensor(np.random.randn(*samples.shape), mstype.float32)
            step_train_loss = train_step(samples, timesteps, noise)
        print(
            f"epoch: {epoch} train loss: {step_train_loss.asnumpy()} epoch time: {time.time() - time_beg:5.3f}s")
        if epoch % 10 == 0:
            result = pipe()
            generate_image(result, diffusion_type)
            print(f'save generated images')
    assert step_train_loss < 0.4


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_train_diffusion_cond(diffusion_type='ddpm', latent_path='latent.npy', cond_path='labels.npy'):
    """
    Feature: conditional diffusion model train
    Description: test DDPM/DDIM model generate MNIST image latent
    Expectation: success
    """
    compute_dtype = mstype.float32
    in_dim = LATENT_DIM
    model = ConditionDiffusionTransformer(in_channels=in_dim,
                                          out_channels=in_dim,
                                          cond_channels=1,
                                          hidden_channels=256,
                                          layers=1,
                                          heads=8,
                                          time_token_cond=True,
                                          compute_dtype=compute_dtype)

    count_params(model)
    num_train_timesteps = 500
    batch_size = 512
    infer_bs = 10
    if diffusion_type == 'ddpm':
        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
                                  beta_start=0.0001,
                                  beta_end=0.02,
                                  beta_schedule="squaredcos_cap_v2",
                                  clip_sample_range=1.0,
                                  thresholding=False,
                                  dynamic_thresholding_ratio=None,
                                  rescale_betas_zero_snr=False,
                                  timestep_spacing="leading",
                                  compute_dtype=compute_dtype)
        pipe = DDPMPipeline(model=model, scheduler=scheduler, batch_size=infer_bs,
                            seq_len=1, num_inference_steps=num_train_timesteps)
    else:
        scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps,
                                  beta_start=0.0001,
                                  beta_end=0.02,
                                  beta_schedule="squaredcos_cap_v2",
                                  clip_sample_range=1.0,
                                  thresholding=False,
                                  dynamic_thresholding_ratio=None,
                                  rescale_betas_zero_snr=False,
                                  timestep_spacing="leading",
                                  compute_dtype=compute_dtype)
        num_inference_steps = 50
        pipe = DDIMPipeline(model=model, scheduler=scheduler, batch_size=infer_bs,
                            seq_len=1, num_inference_steps=num_inference_steps)

    trainer = DiffusionTrainer(model,
                               scheduler,
                               objective='pred_noise',
                               p2_loss_weight_gamma=0,
                               p2_loss_weight_k=1,
                               loss_type='l2')

    def forward_fn(data, t, noise, cond):
        loss = trainer.get_loss(data, noise, t, cond)
        return loss
    lr = 0.0002
    optimizer = nn.Adam(model.trainable_params(), lr, 0.9, 0.99)
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    @jit
    def train_step(data, t, noise, cond):
        loss, grads = grad_fn(data, t, noise, cond)
        is_finite = amp.all_finite(grads)
        if is_finite:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    train_dataset = get_latent_dataset(
        latent_path=latent_path, cond_path=cond_path, batch_size=batch_size)
    epochs = 20
    print(f'scheduler.num_timesteps: {scheduler.num_timesteps}')
    infer_cond = ops.arange(0, 10, dtype=mstype.float32).reshape(10, 1)
    for epoch in range(1, epochs+1):
        time_beg = time.time()
        for samples, cond in train_dataset:
            timesteps = Tensor(np.random.randint(
                0, num_train_timesteps, (samples.shape[0],)).astype(np.int32))
            noise = Tensor(np.random.randn(*samples.shape), mstype.float32)
            step_train_loss = train_step(samples, timesteps, noise, cond)
        print(
            f"epoch: {epoch} train loss: {step_train_loss.asnumpy()} epoch time: {time.time() - time_beg:5.3f}s")
        if epoch % 10 == 0:
            result = pipe(infer_cond)
            generate_image(result, diffusion_type)
            print(f'save generated conditional images')
    assert step_train_loss < 0.4
