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
# ============================================================================
"""utils for 1st phase generation"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

import mindspore as ms


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        u, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gm, latents):
    fig, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=200)
    ax = axs or plt.gca()
    ax.scatter(latents[:, 0], latents[:, 1], s=5, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gm.weights_.max()
    for pos, covar, w in zip(gm.means_, gm.covariances_, gm.weights_):
        draw_ellipse(pos, covar, alpha=0.75 * w * w_factor, facecolor='slategrey', zorder=-10)


def gaussian_mixture_model(latents, params):
    gm = GaussianMixture(n_components=4, random_state=0, init_params='kmeans').fit(latents)
    print('Average negative log likelihood:', -1 * gm.score(latents))
    if params['visualize']:
        plot_gmm(gm, latents)
        scores = []
        for i in range(1, 8):
            gm = GaussianMixture(n_components=i, random_state=0, init_params='kmeans').fit(latents)
            scores.append(-1 * gm.score(latents))
        sns.set_style("darkgrid")
        plt.figure()
        plt.scatter(range(1, 8), scores, color='green')
        plt.plot(range(1, 8), scores)
        plt.savefig(params['folder_dir'] + '/gaussian_mixture_model.png', format='png', dpi=300)
        plt.show()
    return gm


def sampler(gm, classifier, n_samples, sigma=0.1):
    sample_z = []
    z = gm.sample(1)[0]
    for i in range(n_samples):
        uniform_rand = np.random.uniform(size=1)
        z_next = np.random.multivariate_normal(z.squeeze(), sigma * np.eye(2)).reshape(1, -1)
        z_combined = np.concatenate((z, z_next), axis=0)
        scores = classifier(ms.Tensor(z_combined, ms.float32)).asnumpy().squeeze()
        z_score, z_next_score = np.log(scores[0]), np.log(scores[1])
        z_prob, z_next_prob = (gm.score(z) + z_score), (gm.score(z_next) + z_next_score)
        acceptance = min(0, (z_next_prob - z_prob))
        if i == 0:
            sample_z.append(z.squeeze())

        if np.log(uniform_rand) < acceptance:
            sample_z.append(z_next.squeeze())
            z = z_next
        else:
            pass
    return np.stack(sample_z)
