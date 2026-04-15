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
"diffusion model training"
import time
import os


import mindspore as ms
from mindspore import ops, nn
from mindspore.train.serialization import save_checkpoint

from src.sevir_dataset import SEVIRDataset


class DiffusionTrainer(nn.Cell):
    """
    Class managing the training pipeline for diffusion models. Handles dataset processing,
    optimizer configuration, gradient clipping, checkpoint saving, and logging.
    """
    def __init__(self, main_module, dm, logger, config):
        """
        Initialize trainer with model, data module, logger, and configuration.
        Args:
            main_module: Main diffusion model to be trained
            dm: Data module providing training dataset
            logger: Logging utility for training progress
            config: Configuration dictionary containing hyperparameters
        """
        super().__init__()
        self.main_module = main_module
        self.traindataset = dm.sevir_train
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
        self.optim_config = config["optim"]
        self.clip_norm = config.get("clip_norm", 2)
        self.ckpt_dir = os.path.join(self.example_save_dir, "ckpt")
        self.keep_ckpt_max = config["summary"].get("keep_ckpt_max", 100)
        self.ckpt_history = []
        self.grad_clip_fn = ops.clip_by_global_norm
        self.optimizer = nn.Adam(params=self.main_module.main_model.trainable_params(),
                                 learning_rate=config["optim"].get("lr", 1e-5))
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self, total_steps: int):
        """Execute complete training pipeline."""
        self.main_module.main_model.set_train(True)
        self.logger.info(f"total_steps: {total_steps}")
        self.logger.info("Initializing training process...")
        loss_processor = Trainonestepforward(self.main_module)
        grad_func = ms.ops.value_and_grad(loss_processor, None, self.main_module.main_model.trainable_params())
        for epoch in range(self.optim_config["max_epochs"]):
            epoch_loss = 0.0
            epoch_start = time.time()

            iterator = self.traindataset.create_dict_iterator()
            assert iterator, "dataset is empty"
            batch_idx = 0
            for batch_idx, batch in enumerate(iterator):
                processed_data = self.datasetprocessing.process_data(batch["vil"])
                loss_value, gradients = grad_func(processed_data)
                clipped_grads = self.grad_clip_fn(gradients, self.clip_norm)
                self.optimizer(clipped_grads)
                epoch_loss += loss_value.asnumpy()
                self.logger.info(
                    f"epoch: {epoch} step: {batch_idx}, loss: {loss_value}"
                )
            self._save_ckpt(epoch)
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                f"Avg Loss: {epoch_loss/(batch_idx+1):.4f}"
            )

    def _save_ckpt(self, epoch: int):
        """Save model ckpt with rotation policy"""
        ckpt_file = f"diffusion_epoch{epoch}.ckpt"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file)

        save_checkpoint(self.main_module.main_model, ckpt_path)
        self.ckpt_history.append(ckpt_path)

        if len(self.ckpt_history) > self.keep_ckpt_max:
            removed_ckpt = self.ckpt_history.pop(0)
            os.remove(removed_ckpt)
            self.logger.info(f"Removed outdated ckpt: {removed_ckpt}")


class Trainonestepforward(nn.Cell):
    """A neural network cell that performs one training step forward pass for a diffusion model.
    This class encapsulates the forward pass computation for training a diffusion model,
    handling the input processing, latent space encoding, conditioning, and loss calculation.
    Args:
        model (nn.Cell): The main diffusion model containing the necessary submodules
                         for encoding, conditioning, and loss computation.
    """

    def __init__(self, model):
        super().__init__()
        self.main_module = model

    def construct(self, inputs):
        """Perform one forward training step and compute the loss."""
        x, condition = self.main_module.get_input(inputs)
        x = x.transpose(0, 1, 4, 2, 3)
        n, t_, c_, h, w = x.shape
        x = x.reshape(n * t_, c_, h, w)
        z = self.main_module.encode_first_stage(x)
        _, c_z, h_z, w_z = z.shape
        z = z.reshape(n, -1, c_z, h_z, w_z)
        z = z.transpose(0, 1, 3, 4, 2)
        t = ops.randint(0, self.main_module.num_timesteps, (n,)).long()
        zc = self.main_module.cond_stage_forward(condition)
        loss = self.main_module.p_losses(z, zc, t, noise=None)
        return loss
