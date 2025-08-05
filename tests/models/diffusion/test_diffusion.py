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
"""diffusion api testcase"""
# pylint: disable=C0413
import os
import sys
import pytest

import numpy as np
from mindspore import Tensor, ops, context
from mindspore import dtype as mstype

from mindflow.cell import DiffusionScheduler, DDPMPipeline, DDIMPipeline, DDPMScheduler, DDIMScheduler, \
    DiffusionTransformer, ConditionDiffusionTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

from common.cell import FP32_RTOL, FP32_ATOL
from common.cell import compare_output

from ddim_gt import DDIMScheduler as DDIMSchedulerGt
from ddpm_gt import DDPMScheduler as DDPMSchedulerGt
from dataset import load_data

BATCH_SIZE, SEQ_LEN, IN_CHANNELS, OUT_CHANNELS, HIDDEN_CHANNELS, COND_CHANNELS = 8, 256, 16, 16, 64, 4
LAYERS, HEADS, TRAIN_TIMESTEPS = 3, 4, 100


def extract(a, t, x_shape):
    """calculate a[timestep]"""
    b = t.shape[0]
    out = Tensor(a).gather(t, -1)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPMSchedulerMock(DDPMScheduler):
    """modify `DDPMScheduler` step function"""

    def step(self, model_output: Tensor, sample: Tensor, timestep: Tensor, predicted_variance: Tensor = None):
        """denoise function"""
        pred_original_sample = self._pred_origin_sample(
            model_output, sample, timestep)

        pred_prev_sample = (
            extract(self.posterior_mean_coef1, timestep, sample.shape)*pred_original_sample +
            extract(self.posterior_mean_coef2, timestep, sample.shape)*sample
        )

        # 3. Add noise
        v = self._get_variance(sample, timestep, predicted_variance)
        variance = 0
        if timestep[0] > 0:
            variance_noise = model_output
            if self.variance_type == "fixed_small_log":
                variance = v * variance_noise
            elif self.variance_type == "learned_range":
                variance = ops.exp(0.5 * v) * variance_noise
            else:
                variance = (v ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample


def generate_inputs():
    """generate npy files"""
    original_samples = np.random.rand(BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    noise = np.random.rand(BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    condition = np.random.rand(BATCH_SIZE, COND_CHANNELS)
    timesteps = np.random.randint(30, 100, (BATCH_SIZE,))
    np.save('noise.npy', noise)
    np.save('original_samples.npy', original_samples)
    np.save('condition.npy', condition)
    np.save('timesteps.npy', timesteps)


def load_inputs(dtype=mstype.float32):
    """load npy"""
    sample = load_data('original_samples.npy')
    noise = load_data('noise.npy')
    cond = load_data('condition.npy')
    t = load_data('timesteps.npy')

    original_samples = Tensor(sample).astype(dtype)
    noise = Tensor(noise).astype(dtype)
    condition = Tensor(cond).astype(dtype)
    timesteps = Tensor(t).astype(mstype.int32)
    return original_samples, noise, condition, timesteps


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddim_step(mode):
    """
    Feature: DDIM step
    Description: test DDIM step function
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDIMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=False,
                              clip_sample_range=1.0,
                              thresholding=False,
                              prediction_type="epsilon",
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)
    scheduler_gt = DDIMSchedulerGt(num_train_timesteps=TRAIN_TIMESTEPS,
                                   beta_start=0.0001,
                                   beta_end=0.02,
                                   beta_schedule="squaredcos_cap_v2",
                                   clip_sample_range=1.0,
                                   trained_betas=None,
                                   thresholding=False,
                                   clip_sample=False,
                                   sample_max_value=1.0,
                                   prediction_type="epsilon",
                                   dynamic_thresholding_ratio=None,
                                   rescale_betas_zero_snr=False,
                                   timestep_spacing="leading")
    scheduler.set_timesteps(20)
    scheduler_gt.set_timesteps(20)
    original_samples, noise, _, _ = load_inputs()
    timesteps = Tensor(np.array([60]*BATCH_SIZE), dtype=mstype.int32)
    x_prev = scheduler.step(noise, original_samples, timesteps).numpy()
    x_prev_gt = scheduler_gt.step(noise, timesteps, original_samples).numpy()
    validate_ans = compare_output(x_prev, x_prev_gt, rtol=FP32_RTOL, atol=FP32_ATOL)
    assert validate_ans, f"test_ddim_step failed: {validate_ans}"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddpm_step(mode):
    """
    Feature: DDPM step
    Description: test DDPM step function
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDPMSchedulerMock(num_train_timesteps=TRAIN_TIMESTEPS,
                                  beta_start=0.0001,
                                  beta_end=0.02,
                                  beta_schedule="squaredcos_cap_v2",
                                  clip_sample=False,
                                  clip_sample_range=1.0,
                                  thresholding=False,
                                  dynamic_thresholding_ratio=None,
                                  rescale_betas_zero_snr=False,
                                  timestep_spacing="leading",
                                  compute_dtype=compute_dtype)
    scheduler_gt = DDPMSchedulerGt(num_train_timesteps=TRAIN_TIMESTEPS,
                                   beta_start=0.0001,
                                   beta_end=0.02,
                                   beta_schedule="squaredcos_cap_v2",
                                   clip_sample_range=1.0,
                                   trained_betas=None,
                                   thresholding=False,
                                   clip_sample=None,
                                   sample_max_value=1.0,
                                   prediction_type="epsilon",
                                   variance_type="fixed_small_log",
                                   dynamic_thresholding_ratio=None,
                                   rescale_betas_zero_snr=False,
                                   timestep_spacing="leading")
    original_samples, noise, _, _ = load_inputs()
    scheduler.set_timesteps(TRAIN_TIMESTEPS)
    scheduler_gt.set_timesteps(TRAIN_TIMESTEPS)
    timesteps = Tensor(np.array([60]*BATCH_SIZE), dtype=mstype.int32)
    x_prev = scheduler.step(noise, original_samples, timesteps).numpy()
    x_prev_gt = scheduler_gt.step(noise, timesteps, original_samples).numpy()
    validate_ans = compare_output(x_prev, x_prev_gt, rtol=FP32_RTOL, atol=FP32_ATOL)
    assert validate_ans, f'test_ddpm_step failed: {validate_ans}'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_set_timestep(mode):
    """
    Feature: diffusion set inference timesteps
    Description: test diffusion set timesteps function
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DiffusionScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                                   beta_start=0.0001,
                                   beta_end=0.02,
                                   beta_schedule="squaredcos_cap_v2",
                                   clip_sample=True,
                                   clip_sample_range=1.0,
                                   thresholding=False,
                                   dynamic_thresholding_ratio=None,
                                   rescale_betas_zero_snr=False,
                                   timestep_spacing="leading",
                                   compute_dtype=compute_dtype)

    try:
        scheduler.set_timesteps(1000)
    except ValueError:
        pass
    else:
        raise Exception("inference steps > train steps. Expected ValueError")


# pylint: disable=W0212
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_diffusion_pred_origin_sample(mode):
    """
    Feature: diffusion add noise
    Description: test diffusion pred origin sample
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DiffusionScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                                   beta_start=0.0001,
                                   beta_end=0.02,
                                   beta_schedule="squaredcos_cap_v2",
                                   clip_sample=False,
                                   clip_sample_range=1.0,
                                   thresholding=False,
                                   dynamic_thresholding_ratio=None,
                                   rescale_betas_zero_snr=False,
                                   timestep_spacing="leading",
                                   compute_dtype=compute_dtype)

    original_samples, noise, _, _ = load_inputs()
    scheduler.set_timesteps(TRAIN_TIMESTEPS)
    timesteps = Tensor(np.array([60]*BATCH_SIZE), dtype=mstype.int32)
    pred_original_sample = scheduler._pred_origin_sample(
        noise, original_samples, timesteps).numpy()
    pred_original_sample_gt = load_data('ddpm_pred_original_sample.npy')
    validate_ans = compare_output(pred_original_sample, pred_original_sample_gt, rtol=FP32_RTOL, atol=FP32_ATOL)
    assert validate_ans, f'test_diffusion_pred_origin_sample failed: {validate_ans}'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddpm_set_timestep(mode):
    """
    Feature: DDPM set timesteps
    Description: test DDPM set timesteps function
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              clip_sample_range=1.0,
                              thresholding=False,
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)

    try:
        scheduler.set_timesteps(50)
    except ValueError as e:
        print(e)
    else:
        raise Exception(
            "DDPM num_inference_steps defaults to num_train_timesteps. Expected ValueError")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddpm_pipe_no_cond(mode):
    """
    Feature: DDPM pipeline generation
    Description: test DDPM pipeline API with no condition input
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              clip_sample_range=1.0,
                              thresholding=False,
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)
    net = DiffusionTransformer(in_channels=IN_CHANNELS,
                               out_channels=IN_CHANNELS,
                               hidden_channels=HIDDEN_CHANNELS,
                               layers=LAYERS,
                               heads=HEADS,
                               time_token_cond=True,
                               compute_dtype=compute_dtype)
    pipe = DDPMPipeline(model=net, scheduler=scheduler,
                        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_inference_steps=TRAIN_TIMESTEPS)

    generated_samples = pipe()
    assert generated_samples.shape == (BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    assert generated_samples.dtype == compute_dtype


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddpm_pipe_cond(mode):
    """
    Feature: DDPM pipeline generation
    Description: test DDPM pipeline API with condition input
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              clip_sample_range=1.0,
                              thresholding=False,
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)
    _, _, cond, _ = load_inputs()
    net = ConditionDiffusionTransformer(in_channels=IN_CHANNELS,
                                        out_channels=IN_CHANNELS,
                                        cond_channels=COND_CHANNELS,
                                        hidden_channels=HIDDEN_CHANNELS,
                                        layers=LAYERS,
                                        heads=HEADS,
                                        time_token_cond=True,
                                        compute_dtype=compute_dtype)
    pipe = DDPMPipeline(model=net, scheduler=scheduler,
                        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_inference_steps=TRAIN_TIMESTEPS)

    generated_samples = pipe(cond)
    assert generated_samples.shape == (BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    assert generated_samples.dtype == compute_dtype


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddim_pipe_no_cond(mode):
    """
    Feature: DDIM pipeline generation
    Description: test DDIM pipeline API with no condition input
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDIMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              clip_sample_range=1.0,
                              thresholding=False,
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)
    net = DiffusionTransformer(in_channels=IN_CHANNELS,
                               out_channels=IN_CHANNELS,
                               hidden_channels=HIDDEN_CHANNELS,
                               layers=LAYERS,
                               heads=HEADS,
                               time_token_cond=True,
                               compute_dtype=compute_dtype)
    pipe = DDIMPipeline(model=net, scheduler=scheduler,
                        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_inference_steps=50)

    generated_samples = pipe()
    assert generated_samples.shape == (BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    assert generated_samples.dtype == compute_dtype


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddim_pipe_cond(mode):
    """
    Feature: DDIM pipeline generation
    Description: test DDIM pipeline API with condition input
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDIMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              clip_sample_range=1.0,
                              thresholding=False,
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)
    _, _, cond, _ = load_inputs()
    net = ConditionDiffusionTransformer(in_channels=IN_CHANNELS,
                                        out_channels=IN_CHANNELS,
                                        cond_channels=COND_CHANNELS,
                                        hidden_channels=HIDDEN_CHANNELS,
                                        layers=LAYERS,
                                        heads=HEADS,
                                        time_token_cond=True,
                                        compute_dtype=compute_dtype)
    pipe = DDIMPipeline(model=net, scheduler=scheduler,
                        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_inference_steps=50)

    generated_samples = pipe(cond)
    assert generated_samples.shape == (BATCH_SIZE, SEQ_LEN, IN_CHANNELS)
    assert generated_samples.dtype == compute_dtype


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_diffusion_addnoise_fp32(mode):
    """
    Feature: diffusion add noise
    Description: test diffusion-fp32 add noise
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DiffusionScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                                   beta_start=0.0001,
                                   beta_end=0.02,
                                   beta_schedule="squaredcos_cap_v2",
                                   prediction_type='epsilon',
                                   clip_sample=True,
                                   clip_sample_range=1.0,
                                   thresholding=False,
                                   sample_max_value=1.,
                                   dynamic_thresholding_ratio=None,
                                   rescale_betas_zero_snr=False,
                                   timestep_spacing="leading",
                                   compute_dtype=compute_dtype)

    original_samples, noise, _, timesteps = load_inputs()
    noised_sample = scheduler.add_noise(original_samples, noise, timesteps)
    noised_sample_gt = load_data('ddpm_noised_sample.npy')
    assert noised_sample.dtype == compute_dtype
    assert noised_sample.shape == (BATCH_SIZE, SEQ_LEN, OUT_CHANNELS)
    assert np.allclose(noised_sample.numpy(), noised_sample_gt)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_diffusion_addnoise_fp16(mode):
    """
    Feature: diffusion add noise
    Description: test diffusion-fp16 add noise
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float16
    scheduler = DiffusionScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                                   beta_start=0.0001,
                                   beta_end=0.02,
                                   beta_schedule="squaredcos_cap_v2",
                                   clip_sample=True,
                                   clip_sample_range=1.0,
                                   thresholding=False,
                                   dynamic_thresholding_ratio=None,
                                   rescale_betas_zero_snr=False,
                                   timestep_spacing="leading",
                                   compute_dtype=compute_dtype)

    original_samples, noise, _, timesteps = load_inputs(mstype.float16)
    noised_sample = scheduler.add_noise(original_samples, noise, timesteps)
    assert noised_sample.dtype == compute_dtype
    assert noised_sample.shape == (BATCH_SIZE, SEQ_LEN, OUT_CHANNELS)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddim_pipeline_eta(mode):
    """
    Feature: DDIM inference
    Description: test DDIM check eta
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDIMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              clip_sample_range=1.0,
                              thresholding=False,
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)
    net = DiffusionTransformer(in_channels=IN_CHANNELS,
                               out_channels=IN_CHANNELS,
                               hidden_channels=HIDDEN_CHANNELS,
                               layers=LAYERS,
                               heads=HEADS,
                               time_token_cond=True,
                               compute_dtype=compute_dtype)
    pipe = DDIMPipeline(model=net, scheduler=scheduler,
                        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_inference_steps=50)

    try:
        _ = pipe(condition=None, eta=2)
    except ValueError as e:
        print(e)
    else:
        raise Exception(
            "DDIM sample 0 <= eta <= 1. Expected ValueError")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('compute_dtype', [mstype.float16, mstype.float32])
def test_diffusion_transfomer(mode, compute_dtype):
    """
    Feature: Diffusion transformer
    Description: test diffusion transformer dtype and shape
    Expectation: success
    """
    context.set_context(mode=mode)
    net = DiffusionTransformer(in_channels=IN_CHANNELS,
                               out_channels=OUT_CHANNELS,
                               hidden_channels=HIDDEN_CHANNELS,
                               layers=LAYERS,
                               heads=HEADS,
                               time_token_cond=True,
                               compute_dtype=compute_dtype)
    x, _, _, timesteps = load_inputs()
    out = net(x, timesteps)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, OUT_CHANNELS)
    assert out.dtype == compute_dtype


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('compute_dtype', [mstype.float16, mstype.float32])
def test_cond_diffusion_transfomer(mode, compute_dtype):
    """
    Feature: Condition Diffusion transformer
    Description: test diffusion transformer dtype and shape
    Expectation: success
    """
    context.set_context(mode=mode)
    net = ConditionDiffusionTransformer(in_channels=IN_CHANNELS,
                                        out_channels=OUT_CHANNELS,
                                        cond_channels=COND_CHANNELS,
                                        hidden_channels=HIDDEN_CHANNELS,
                                        layers=LAYERS,
                                        heads=HEADS,
                                        cond_as_token=False,
                                        time_token_cond=True,
                                        compute_dtype=compute_dtype)
    x, _, condition, timesteps = load_inputs()
    out = net(x, timesteps, condition)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, OUT_CHANNELS)
    assert out.dtype == compute_dtype


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddim_pipe_type(mode):
    """
    Feature: DDIM pipeline generation
    Description: test DDIM pipeline API with DDPMScheduler
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              clip_sample_range=1.0,
                              thresholding=False,
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)
    net = DiffusionTransformer(in_channels=IN_CHANNELS,
                               out_channels=IN_CHANNELS,
                               hidden_channels=HIDDEN_CHANNELS,
                               layers=LAYERS,
                               heads=HEADS,
                               time_token_cond=True,
                               compute_dtype=compute_dtype)
    try:
        _ = DDIMPipeline(model=net, scheduler=scheduler,
                         batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_inference_steps=50)
    except TypeError:
        pass
    else:
        raise Exception("DDPMScheduler type. Expected TypeError")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ddpm_pipe_type(mode):
    """
    Feature: DDPM pipeline generation
    Description: test DDPM pipeline API with DDIMScheduler
    Expectation: success
    """
    context.set_context(mode=mode)
    compute_dtype = mstype.float32
    scheduler = DDIMScheduler(num_train_timesteps=TRAIN_TIMESTEPS,
                              beta_start=0.0001,
                              beta_end=0.02,
                              beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,
                              clip_sample_range=1.0,
                              thresholding=False,
                              dynamic_thresholding_ratio=None,
                              rescale_betas_zero_snr=False,
                              timestep_spacing="leading",
                              compute_dtype=compute_dtype)
    net = DiffusionTransformer(in_channels=IN_CHANNELS,
                               out_channels=IN_CHANNELS,
                               hidden_channels=HIDDEN_CHANNELS,
                               layers=LAYERS,
                               heads=HEADS,
                               time_token_cond=True,
                               compute_dtype=compute_dtype)
    try:
        _ = DDPMPipeline(model=net, scheduler=scheduler,
                         batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_inference_steps=50)
    except TypeError:
        pass
    else:
        raise Exception("DDIMScheduler type. Expected TypeError")
