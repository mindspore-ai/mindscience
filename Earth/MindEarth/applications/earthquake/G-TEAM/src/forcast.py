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
"GTeam inference"
import numpy as np

from src.utils import (
    predict_at_time,
    calc_mag_stats,
    calc_loc_stats,
    calc_pga_stats,
)
from src.data import load_data
from src.visual import generate_true_pred_plot


class GTeamInference:
    """
    Initialize the GTeamInference class.
    """

    def __init__(self, model_ins, cfg, output_dir, logger):
        """
        Args:
            model_ins: The model instance used for inference.
            cfg: Configuration dictionary containing model and data parameters.
            output_dir: Directory to save the output results.
        Attributes:
            model: The model instance for inference.
            cfg: Configuration dictionary.
            output_dir: Directory to save outputs.
            pga: Flag indicating if PGA (Peak Ground Acceleration) is enabled.
            generator_params: Parameters for data generation.
            model_params: Parameters specific to the model.
            mag_key: Key for magnitude-related data.
            pos_offset: Position offset for location predictions.
            mag_stats: List to store magnitude prediction statistics.
            loc_stats: List to store location prediction statistics.
            pga_stats: List to store PGA prediction statistics.
        """
        self.model = model_ins
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = logger
        self.pga = cfg["model"].get("pga", "true")
        self.generator_params = cfg["data"]
        self.model_params = cfg["model"]
        self.output_dir = output_dir
        self.mag_key = self.generator_params["key"]
        self.pos_offset = self.generator_params["pos_offset"]
        self.mag_stats = []
        self.loc_stats = []
        self.pga_stats = []

    def _parse_predictions(self, pred):
        """
        Parse the raw predictions into magnitude, location, and PGA components.
        """
        mag_pred = pred[0]
        loc_pred = pred[1]
        pga_pred = pred[2] if self.pga else []
        return mag_pred, loc_pred, pga_pred

    def _process_predictions(
            self, mag_pred, loc_pred, pga_pred, time, evt_metadata, pga_true
    ):
        """
        Process the parsed predictions to compute statistics and generate plots.
        """
        mag_pred_np = [t[0].asnumpy() for t in mag_pred]
        mag_pred_reshaped = np.concatenate(mag_pred_np, axis=0)

        loc_pred_np = [t[0].asnumpy() for t in loc_pred]
        loc_pred_reshaped = np.array(loc_pred_np)

        pga_pred_np = [t.asnumpy() for t in pga_pred]
        pga_pred_reshaped = np.concatenate(pga_pred_np, axis=0)
        pga_true_reshaped = np.log(
            np.abs(np.concatenate(pga_true, axis=0).reshape(-1, 1))
        )

        if not self.model_params["no_event_token"]:
            self.mag_stats += calc_mag_stats(
                mag_pred_reshaped, evt_metadata, self.mag_key
            )

            self.loc_stats += calc_loc_stats(
                loc_pred_reshaped, evt_metadata, self.pos_offset
            )

            generate_true_pred_plot(
                mag_pred_reshaped,
                evt_metadata[self.mag_key].values,
                time,
                self.output_dir,
            )
        self.pga_stats = calc_pga_stats(pga_pred_reshaped, pga_true_reshaped)

    def _save_results(self):
        """
        Save the final results (magnitude, location, and PGA statistics) to a JSON file.
        """
        times = self.cfg["model"].get("times")
        self.logger.info("times: {}".format(times))
        self.logger.info("mag_stats: {}".format(self.mag_stats))
        self.logger.info("loc_stats: {}".format(self.loc_stats))
        self.logger.info("pga_stats: {}".format(self.pga_stats))

    def test(self):
        """
        Perform inference for all specified times, process predictions, and save results.
        This method iterates over the specified times, performs predictions, processes
        the results, and saves the final statistics.
        """
        data_data, evt_key, evt_metadata, meta_data, data_path = load_data(self.cfg)
        pga_true = data_data["pga"]
        for time in self.cfg["model"].get("times"):
            pred = predict_at_time(
                self.model,
                time,
                data_data,
                data_path,
                evt_key,
                evt_metadata,
                config=self.cfg,
                pga=self.pga,
                sampling_rate=meta_data["sampling_rate"],
            )
            mag_pred, loc_pred, pga_pred = self._parse_predictions(pred)
            self._process_predictions(
                mag_pred, loc_pred, pga_pred, time, evt_metadata, pga_true
            )
        self._save_results()
        print("Inference completed and results saved")
