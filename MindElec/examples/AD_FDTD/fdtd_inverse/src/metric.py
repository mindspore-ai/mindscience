# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Metrics for image reconstruction.
"""
import numpy as np


def psnr(evaluation, target):
    """
    Compute PSNR (Peak Signal to Noise Ratio) of the evaluation.

    Args:
        evaluation (numpy.ndarray, shape=(nx,ny)): The evaluated results.
        target (numpy.ndarray, shape=(nx,ny)): The ground truth.

    Returns:
        psnr_score (float): PSNR in dB
    """
    mse = np.mean((evaluation - target)**2)
    data_range = np.max(np.abs(target)) - np.min(np.abs(target))
    psnr_score = 20. * np.log10(data_range / np.sqrt(mse))
    return psnr_score


def ssim(evaluation, target):
    """
    Compute SSIM (Structural Similarity Index) between the evaluations and the targets.

    Args:
        evaluation (numpy.ndarray, shape=(nx,ny)): The evaluated results.
        target (numpy.ndarray, shape=(nx,ny)): The ground truth.

    Returns:
        ssim_score (float): SSIM
    """
    mu_x = np.mean(evaluation)
    mu_y = np.mean(target)
    sigma_x = np.sqrt(np.mean((evaluation - mu_x)**2))
    sigma_y = np.sqrt(np.mean((target - mu_y)**2))
    sigma = np.mean((evaluation - mu_x) * (target - mu_y))

    data_range = np.max(np.abs(target)) - np.min(np.abs(target))
    c1 = data_range * 1e-2
    c2 = data_range * 3e-2

    ssim_score = ((2 * mu_x * mu_y + c1) * (2. * sigma + c2)) / \
        ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
    return ssim_score
