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
"GTeam util"
import os
import copy
import numpy as np
import sklearn.metrics as metrics
from geopy.distance import geodesic

import mindspore as ms
import mindspore.ops as ops

from src import data
from src.data import generator_from_config, D2KM
from src.models import WaveformFullmodel
from mindearth.utils import create_logger


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

    for i, (start, end) in enumerate(
            zip(generator.dataset.reverse_index[:-1], generator.dataset.reverse_index[1:])
    ):
        sample_mag_pred = predictions[0][start:end].reshape(
            (-1,) + predictions[0].shape[-1:]
        )
        sample_mag_pred = sample_mag_pred[: len(generator.dataset.pga[i])]
        mag_pred_filter += [sample_mag_pred]

        sample_loc_pred = predictions[1][start:end].reshape(
            (-1,) + predictions[1].shape[-1:]
        )
        sample_loc_pred = sample_loc_pred[: len(generator.dataset.pga[i])]
        loc_pred_filter += [sample_loc_pred]

        sample_pga_pred = predictions[2][start:end].reshape(
            (-1,) + predictions[2].shape[-1:]
        )
        sample_pga_pred = sample_pga_pred[: len(generator.dataset.pga[i])]
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


def init_model(arg):
    """set model"""
    tmpcfg = copy.deepcopy(arg["model"])
    tmpcfg.pop("no_event_token")
    tmpcfg.pop("run_with_less_data")
    tmpcfg.pop("pga")
    tmpcfg.pop("mode")
    tmpcfg.pop("times")
    model = WaveformFullmodel(**tmpcfg)
    param_dict = ms.load_checkpoint(arg["summary"].get("ckpt_path"))
    # Load parameters into the network
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
