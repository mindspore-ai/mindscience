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
"""GTeam util"""
import os
import copy
import numpy as np
import sklearn.metrics as metrics
from geopy.distance import geodesic

import mindspore as ms
import mindspore.ops as ops
from mindearth import create_logger

from src import data
from src.data import generator_from_config, D2KM
from src.models import WaveformFullmodel

def predict_at_time(
        model,
        time,
        data_data,
        data_path,
        event_key,
        event_metadata,
        config,
        sampling_rate=100,
        pga=False,
):
    """Predict at a specific time point"""
    generator = generator_from_config(
        config,
        data_data,
        data_path,
        event_key,
        event_metadata,
        time,
        sampling_rate=sampling_rate,
        pga=pga,
    )

    pred_list_mag = []
    pred_list_loc = []
    pred_list_pga = []
    for i in range(len(generator)):
        x, _ = generator[i]

        pred = model(x[0], x[1], x[2])
        pred_list_mag.append(pred[0])
        pred_list_loc.append(pred[1])
        pred_list_pga.append(pred[2])

    pre_mag = ops.cat(pred_list_mag, axis=0)
    pre_loc = ops.cat(pred_list_loc, axis=0)
    pre_pga = ops.cat(pred_list_pga, axis=0)
    predictions = [pre_mag, pre_loc, pre_pga]

    mag_pred_filter = []
    loc_pred_filter = []
    pga_pred_filter = []

    for i, (start, end) in enumerate(zip(generator.reverse_index[:-1], generator.reverse_index[1:])):
        sample_mag_pred = predictions[0][start:end].reshape((-1,) + predictions[0].shape[-1:])
        sample_mag_pred = sample_mag_pred[:len(generator.pga[i])]
        mag_pred_filter += [sample_mag_pred]

        sample_loc_pred = predictions[1][start:end].reshape((-1,) + predictions[1].shape[-1:])
        sample_loc_pred = sample_loc_pred[:len(generator.pga[i])]
        loc_pred_filter += [sample_loc_pred]

        sample_pga_pred = predictions[2][start:end].reshape((-1,) + predictions[2].shape[-1:])
        sample_pga_pred = sample_pga_pred[:len(generator.pga[i])]
        pga_pred_filter += [sample_pga_pred]

    preds = [mag_pred_filter, loc_pred_filter, pga_pred_filter]

    return preds

def calc_mag_stats(mag_pred, event_metadata, key):
    """Calculate statistical information for magnitude predictions"""
    mean_mag = mag_pred
    true_mag = event_metadata[key].values
    # R^2
    r2 = metrics.r2_score(true_mag, mean_mag)
    # RMSE
    rmse = np.sqrt(metrics.mean_squared_error(true_mag, mean_mag))
    # MAE
    mae = metrics.mean_absolute_error(true_mag, mean_mag)
    return r2, rmse, mae

def calc_pga_stats(pga_pred, pga_true, suffix=""):
    """Calculate statistical information for PGA predictions"""
    if suffix:
        suffix += "_"
    valid_mask = np.isfinite(pga_true) & np.isfinite(pga_pred)
    pga_true_clean = pga_true[valid_mask]
    pga_pred_clean = pga_pred[valid_mask]
    r2 = metrics.r2_score(pga_true_clean, pga_pred_clean)
    rmse = np.sqrt(metrics.mean_squared_error(pga_true_clean, pga_pred_clean))
    mae = metrics.mean_absolute_error(pga_true_clean, pga_pred_clean)

    return [r2, rmse, mae]

def calc_loc_stats(loc_pred, event_metadata, pos_offset):
    """Calculate statistical information for location predictions"""
    coord_keys = data.detect_location_keys(event_metadata.columns)
    true_coords = event_metadata[coord_keys].values
    mean_coords = loc_pred
    mean_coords *= 100
    mean_coords[:, :2] /= D2KM
    mean_coords[:, 0] += pos_offset[0]
    mean_coords[:, 1] += pos_offset[1]

    dist_epi = np.zeros(len(mean_coords))
    dist_hypo = np.zeros(len(mean_coords))
    real_dep = np.zeros(len(mean_coords))
    pred_dep = np.zeros(len(mean_coords))
    for i, (pred_coord, true_coord) in enumerate(zip(mean_coords, true_coords)):
        dist_epi[i] = geodesic(pred_coord[:2], true_coord[:2]).km
        dist_hypo[i] = np.sqrt(dist_epi[i] ** 2 + (pred_coord[2] - true_coord[2]) ** 2)
        real_dep[i] = true_coord[2]
        pred_dep[i] = pred_coord[2]

    rmse_epi = np.sqrt(np.mean(dist_epi**2))
    mae_epi = np.mean(np.abs(dist_epi))

    rmse_hypo = np.sqrt(np.mean(dist_hypo**2))
    mae_hypo = np.mean(dist_hypo)

    return rmse_hypo, mae_hypo, rmse_epi, mae_epi


def seed_np_tf(seed):
    '''Set the random seed for numpy and manual seed for mindspore.'''
    np.random.seed(seed)
    ms.manual_seed(seed)


def evaluation(full_model, val_generator, losses, loss_weights):
    """
    Evaluates the performance of the full_model on the validation data provided by val_generator.
    Calculates the average validation loss by accumulating losses from different components (magnitude, location, pga)
    using the specified loss functions and weights.
    Args:
        full_model (nn.Cell): The complete model to be evaluated in inference mode.
        val_generator (generator): A generator that yields batches of validation data (x, y).
        Each x is expected to be a tuple of three input tensors, and y is a tuple of three target tensors.
        losses (dict): A dictionary mapping loss names to their respective loss functions.
        Supported keys: 'magnitude', 'location', 'pga'.
        loss_weights (dict): A dictionary mapping loss names to their corresponding weights.
    Returns:
        float: The average validation loss over the entire validation dataset.
    """
    full_model.set_train(False)
    epoch_val_loss = 0
    for i in range(len(val_generator)):
        x, y = val_generator[i]
        outputs = full_model(x[0], x[1], x[2])
        total_val_loss = ms.Tensor(0)

        for k, loss_fn in losses.items():
            if k == 'magnitude':
                mag_pre = outputs[0]
                mag_target = y[0]
                mag_loss = loss_fn(mag_target.squeeze(2), mag_pre) * loss_weights[k]
                total_val_loss += mag_loss
            elif k == 'location':
                loc_pre = outputs[1]
                loc_target = y[1]
                loc_loss = loss_fn(loc_target.squeeze(2), loc_pre) * loss_weights[k]
                total_val_loss += loc_loss
            elif k == 'pga':
                pga_pre = outputs[2]
                pga_target = ops.log(ops.abs(y[2]))
                pga_loss = loss_fn(pga_target.squeeze(3), pga_pre) * loss_weights[k]
                total_val_loss += pga_loss
        epoch_val_loss += total_val_loss.item()
    avg_val_loss = epoch_val_loss / len(val_generator)
    return avg_val_loss
def init_model(arg):
    """set model"""
    tmpcfg = copy.deepcopy(arg["model"])
    tmpcfg.pop("istraining")
    tmpcfg.pop("no_event_token")
    tmpcfg.pop("run_with_less_data")
    tmpcfg.pop("pga")
    tmpcfg.pop("mode")
    tmpcfg.pop("times")
    tmpcfg.pop("max_stations")
    model = WaveformFullmodel(**tmpcfg)
    if arg['model']['istraining']:
        model.set_train(True)
    else:
        param_dict = ms.load_checkpoint(arg["summary"].get("ckpt_path"))
        ms.load_param_into_net(model, param_dict)
        model.set_train(False)
    return model


def get_logger(config):
    """Get logger for saving log"""
    summary_params = config.get("summary")
    logger = create_logger(
        path=os.path.join(summary_params.get("summary_dir"), "results.log")
    )
    for key in config:
        logger.info(config[key])
    return logger
