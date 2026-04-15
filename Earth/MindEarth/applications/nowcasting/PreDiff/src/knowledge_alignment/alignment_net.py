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
"alignment main model"
from typing import Dict, Any

import mindspore
from mindspore import mint, ops

from .alignment import NoisyCuboidTransformerEncoder


class AvgIntensityAlignment:
    """
    A class for intensity alignment using a NoisyCuboidTransformerEncoder model to guide latent space adjustments.
    """

    def __init__(
            self,
            guide_scale: float = 1.0,
            model_args: Dict[str, Any] = None,
            model_ckpt_path: str = None,
    ):
        r"""

        Parameters
        ----------
        alignment_type: str
        guide_scale:    float
        model_type: str
        model_args: Dict[str, Any]
        model_ckpt_path:    str
            if not None, load the model from the checkpoint
        """
        super().__init__()
        self.guide_scale = guide_scale
        if model_args is None:
            model_args = {}
        self.model = NoisyCuboidTransformerEncoder(**model_args)
        self.load_ckpt(model_ckpt_path)

    def load_ckpt(self, model_ckpt_path):
        if model_ckpt_path is not None:
            param_dict = mindspore.load_checkpoint(model_ckpt_path)
            param_not_load, _ = mindspore.load_param_into_net(self.model, param_dict)
            print("NoisyCuboidTransformerEncoder param_not_load:", param_not_load)

    def get_sample_align_fn(self, sample_align_model):
        """get_sample_align_fn"""

        def sample_align_fn(x, *args, **kwargs):
            def forward_fn(x_in):
                x_stop = ops.stop_gradient(x_in)
                return sample_align_model(x_stop, *args, **kwargs)

            grad_fn = mindspore.grad(forward_fn, grad_position=0)
            gradient = grad_fn(x)
            return gradient

        return sample_align_fn

    def alignment_fn(self, zt, t, **kwargs):
        r"""
        transform the learned model to the final guidance \mathcal{F}.

        Parameters
        ----------
        zt: ms.Tensor
            noisy latent z
        t:  ms.Tensor
            timestamp
        y:  ms.Tensor
            context sequence in pixel space
        zc: ms.Tensor
            encoded context sequence in latente space
        kwargs: Dict[str, Any]
            auxiliary knowledge for guided generation
            `avg_x_gt`: float is required.
        Returns
        -------
        ret:    ms.Tensor
        """
        pred = self.model(zt, t)
        target = kwargs.get("avg_x_gt")
        pred = pred.mean(axis=1)
        ret = mint.linalg.vector_norm(pred - target, ord=2)
        return ret

    def get_mean_shift(self, zt, t, y=None, zc=None, **kwargs):
        r"""
        Parameters
        ----------
        zt: ms.Tensor
            noisy latent z
        t:  ms.Tensor
            timestamp
        y:  ms.Tensor
            context sequence in pixel space
        zc: ms.Tensor
            encoded context sequence in latente space
        Returns
        -------
        ret:    ms.Tensor
            \nabla_zt U
        """
        grad_fn = self.get_sample_align_fn(self.alignment_fn)
        grad = grad_fn(zt, t, y=y, zc=zc, **kwargs)
        return self.guide_scale * grad
