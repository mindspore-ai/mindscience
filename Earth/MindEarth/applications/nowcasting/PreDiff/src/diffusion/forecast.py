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
'''diffusion inferrence'''
import time
import os
import json
from typing import Sequence, Union
import numpy as np
from einops import rearrange

import mindspore as ms
from mindspore import ops, mint, nn

from src.visual import vis_sevir_seq
from src.sevir_dataset import SEVIRDataset


def get_alignment_kwargs_avg_x(target_seq):
    """Generate alignment parameters for guided sampling"""
    batch_size = target_seq.shape[0]
    avg_intensity = mint.mean(target_seq.view(batch_size, -1), dim=1, keepdim=True)
    return {"avg_x_gt": avg_intensity * 2.0}


class DiffusionInferrence(nn.Cell):
    """
    Class managing model inference and evaluation processes. Handles loading checkpoints,
    generating predictions, calculating evaluation metrics, and saving visualization results.
    """
    def __init__(self, main_module, dm, logger, config):
        """
        Initialize inference manager with model, data module, logger, and configuration.
        Args:
            main_module: Main diffusion model for inference
            dm: Data module providing test dataset
            logger: Logging utility for evaluation progress
            config: Configuration dictionary containing evaluation parameters
        """
        super().__init__()
        self.ckpt_path = config["summary"].get(
            "ckpt_path",
            "./ckpt/diffusion.ckpt",
        )  # 删除
        self.num_samples = config["eval"].get("num_samples_per_context", 1)
        self.eval_example_only = config["eval"].get("eval_example_only", True)
        self.alignment_type = (
            config.get("model", {}).get("align", {}).get("alignment_type", "avg_x")
        )
        self.use_alignment = self.alignment_type is not None
        self.eval_aligned = config["eval"].get("eval_aligned", True)
        self.eval_unaligned = config["eval"].get("eval_unaligned", True)
        self.num_samples_per_context = config["eval"].get("num_samples_per_context", 1)
        self.logging_prefix = config["logging"].get("logging_prefix", "PreDiff")
        self.test_example_data_idx_list = config["eval"].get(
            "test_example_data_idx_list", [0, 16, 32, 48, 64, 72, 96, 108, 128]
        )
        self.main_module = main_module
        self.testdataset = dm.sevir_test
        self.logger = logger
        self.datasetprocessing = SEVIRDataset(
            data_types=["vil"],
            layout="NHWT",
            rescale_method=config.get("rescale_method", "01"),
        )
        self.example_save_dir = config["summary"].get("summary_dir", "./summary")

        self.fs = config["eval"].get("fs", 20)
        self.label_offset = config["eval"].get("label_offset", [-0.5, 0.5])
        self.label_avg_int = config["eval"].get("label_avg_int", False)

        self.current_epoch = 0

        self.learn_logvar = (
            config.get("model", {}).get("diffusion", {}).get("learn_logvar", False)
        )
        self.logvar = main_module.logvar
        self.maeloss = nn.MAELoss()
        self.test_metrics = {
            "step": 0,
            "mse": 0.0,
            "mae": 0.0,
            "ssim": 0.0,
            "mse_kc": 0.0,
            "mae_kc": 0.0,
        }

    def test(self):
        """Execute complete evaluation pipeline."""
        self.logger.info("============== Start Test ==============")
        self.start_time = time.time()
        for batch_idx, item in enumerate(self.testdataset.create_dict_iterator()):
            self.test_metrics = self._test_onestep(item, batch_idx, self.test_metrics)

        self._finalize_test(self.test_metrics)

    def _test_onestep(self, item, batch_idx, metrics):
        """Process one test batch and update evaluation metrics."""
        data_idx = int(batch_idx * 2)
        if not self._should_test_onestep(data_idx):
            return metrics
        data = item.get("vil")
        data = self.datasetprocessing.process_data(data)
        target_seq, cond, context_seq = self._get_model_inputs(data)
        aligned_preds, unaligned_preds = self._generate_predictions(
            cond, target_seq
        )
        metrics = self._update_metrics(
            aligned_preds, unaligned_preds, target_seq, metrics
        )
        self._plt_pred(
            data_idx,
            context_seq,
            target_seq,
            aligned_preds,
            unaligned_preds,
            metrics["step"],
        )

        metrics["step"] += 1
        return metrics

    def _should_test_onestep(self, data_idx):
        """Determine if evaluation should be performed on current data index."""
        return (not self.eval_example_only) or (
            data_idx in self.test_example_data_idx_list
        )

    def _get_model_inputs(self, data):
        """Extract and prepare model inputs from raw data."""
        target_seq, cond, context_seq = self.main_module.get_input(
            data, return_verbose=True
        )
        return target_seq, cond, context_seq

    def _generate_predictions(self, cond, target_seq):
        """Generate both aligned and unaligned predictions from the model."""
        aligned_preds = []
        unaligned_preds = []

        for _ in range(self.num_samples_per_context):
            if self.use_alignment and self.eval_aligned:
                aligned_pred = self._sample_with_alignment(
                    cond, target_seq
                )
                aligned_preds.append(aligned_pred)

            if self.eval_unaligned:
                unaligned_pred = self._sample_without_alignment(cond)
                unaligned_preds.append(unaligned_pred)

        return aligned_preds, unaligned_preds

    def _sample_with_alignment(self, cond, target_seq):
        """Generate predictions using alignment mechanism."""
        alignment_kwargs = get_alignment_kwargs_avg_x(target_seq)
        pred_seq = self.main_module.sample(
            cond=cond,
            batch_size=cond["y"].shape[0],
            return_intermediates=False,
            use_alignment=True,
            alignment_kwargs=alignment_kwargs,
            verbose=False,
        )
        if pred_seq.dtype != ms.float32:
            pred_seq = pred_seq.float()
        return pred_seq

    def _sample_without_alignment(self, cond):
        """Generate predictions without alignment."""
        pred_seq = self.main_module.sample(
            cond=cond,
            batch_size=cond["y"].shape[0],
            return_intermediates=False,
            verbose=False,
        )
        if pred_seq.dtype != ms.float32:
            pred_seq = pred_seq.float()
        return pred_seq

    def _update_metrics(self, aligned_preds, unaligned_preds, target_seq, metrics):
        """Update evaluation metrics with new predictions."""
        for pred in aligned_preds:
            metrics["mse_kc"] += ops.mse_loss(pred, target_seq)
            metrics["mae_kc"] += self.maeloss(pred, target_seq)
            self.main_module.test_aligned_score.update(pred, target_seq)

        for pred in unaligned_preds:
            metrics["mse"] += ops.mse_loss(pred, target_seq)
            metrics["mae"] += self.maeloss(pred, target_seq)
            self.main_module.test_score.update(pred, target_seq)

            pred_bchw = self._convert_to_bchw(pred)
            target_bchw = self._convert_to_bchw(target_seq)
            metrics["ssim"] += self.main_module.test_ssim(pred_bchw, target_bchw)[0]

        return metrics

    def _convert_to_bchw(self, tensor):
        """Convert tensor to batch-channel-height-width format for metrics."""
        return rearrange(tensor.asnumpy(), "b t h w c -> (b t) c h w")

    def _plt_pred(
            self, data_idx, context_seq, target_seq, aligned_preds, unaligned_preds, step
    ):
        """Generate and save visualization of predictions."""
        pred_sequences = [pred[0].asnumpy() for pred in aligned_preds + unaligned_preds]
        pred_labels = [
            f"{self.logging_prefix}_aligned_pred_{i}" for i in range(len(aligned_preds))
        ] + [f"{self.logging_prefix}_pred_{i}" for i in range(len(unaligned_preds))]

        self.save_vis_step_end(
            data_idx=data_idx,
            context_seq=context_seq[0].asnumpy(),
            target_seq=target_seq[0].asnumpy(),
            pred_seq=pred_sequences,
            pred_label=pred_labels,
            mode="test",
            suffix=f"_step_{step}",
        )

    def _finalize_test(self, metrics):
        """Complete test process and log final metrics."""
        total_time = (time.time() - self.start_time) * 1000
        self.logger.info(f"test cost: {total_time:.2f} ms")
        self._compute_total_metrics(metrics)
        self.logger.info("============== Test Completed ==============")

    def _compute_total_metrics(self, metrics):
        """log_metrics"""
        step_count = max(metrics["step"], 1)
        if self.eval_unaligned:
            self.logger.info(f"MSE: {metrics['mse'] / step_count}")
            self.logger.info(f"MAE: {metrics['mae'] / step_count}")
            self.logger.info(f"SSIM: {metrics['ssim'] / step_count}")
            test_score = self.main_module.test_score.eval()
            self.logger.info("SCORE:\n%s", json.dumps(test_score, indent=4))
        if self.use_alignment:
            self.logger.info(f"KC_MSE: {metrics['mse_kc'] / step_count}")
            self.logger.info(f"KC_MAE: {metrics['mae_kc'] / step_count}")
            aligned_score = self.main_module.test_aligned_score.eval()
            self.logger.info("KC_SCORE:\n%s", json.dumps(aligned_score, indent=4))

    def save_vis_step_end(
            self,
            data_idx: int,
            context_seq: np.ndarray,
            target_seq: np.ndarray,
            pred_seq: Union[np.ndarray, Sequence[np.ndarray]],
            pred_label: Union[str, Sequence[str]] = None,
            mode: str = "train",
            prefix: str = "",
            suffix: str = "",
    ):
        """Save visualization of predictions with context and target."""
        example_data_idx_list = self.test_example_data_idx_list
        if isinstance(pred_seq, Sequence):
            seq_list = [context_seq, target_seq] + list(pred_seq)
            label_list = ["context", "target"] + pred_label
        else:
            seq_list = [context_seq, target_seq, pred_seq]
            label_list = ["context", "target", pred_label]
        if data_idx in example_data_idx_list:
            png_save_name = f"{prefix}{mode}_data_{data_idx}{suffix}.png"
            vis_sevir_seq(
                save_path=os.path.join(self.example_save_dir, png_save_name),
                seq=seq_list,
                label=label_list,
                interval_real_time=10,
                plot_stride=1,
                fs=self.fs,
                label_offset=self.label_offset,
                label_avg_int=self.label_avg_int,
            )
