# right 2020-2022 Huawei Technologies Co., Ltd
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
"""inference"""
import os
from math import radians, sin, cos, degrees, atan2

import cv2
import haversine.haversine
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm
from skimage import morphology

import mindspore as ms
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .segformer import SegFormer
from .utils import make_grid, warp, readbin


class Tester:
    """
    Sea Ice Segmentation Model Evaluator

    This class is used to evaluate the performance of sea ice segmentation models, including:
    - Loading models and checkpoints
    - Processing input data and making predictions
    - Calculating various evaluation metrics
    - Visualizing prediction results
    - Assessing the detection performance of Linear Kinematic Features (LKF)

    Attributes:
        config (dict): Configuration dictionary containing parameters such as model paths and data paths
        model (SegFormer): Instance of the segmentation model
        rmse_all (float): Cumulative Root Mean Square Error
        max_rmse (float): Maximum Root Mean Square Error
        output_dir_pred (str): Path for saving prediction results
        output_dir_mask (str): Path for saving label results
    """
    def __init__(self, config):
        """
        Initialize the IceSegmentationEvaluator class.

        Args:
            config (dict): Configuration dictionary containing:
                - model_checkpoint: Path to the model checkpoint file
                - test_input_root: Path to test input data
                - test_label_root: Path to test label data
                - output_path: Path to save output
        """
        self.config = config
        self.in_channels = self.config["model"].get("in_channels")
        # Initialize model
        self.model = SegFormer(
            in_channels=self.in_channels,
            num_classes=1,
            embedding_dim=256,
        )
        self._load_checkpoint()
        self.rmse_all = 0
        self.max_rmse = 0

    def _load_checkpoint(self):
        """Load model checkpoint."""
        param_dict = load_checkpoint(self.config["test"].get("model_checkpoint"))
        missing_keys, unexpected_keys = load_param_into_net(self.model, param_dict)
        print("=" * 50)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        print("Checkpoint loaded successfully!")
        print("=" * 50)
        # Print total parameters
        total_params = sum(param.size for param in self.model.get_parameters())
        print(f"Total Parameters: {total_params}")

    def _process_input(self, inp_path, label_path):
        """Process input and label data."""
        # Load data
        inp = np.load(inp_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True)
        # Transpose and reshape input
        inp = inp.transpose(2, 0, 1)
        ori_inp = inp[1, :, :]
        inp = np.expand_dims(inp[1:7, :, :], axis=0)
        inp = Tensor(inp, ms.float32)
        # Process labels
        labels = labels.transpose(2, 0, 1)
        labels = np.expand_dims(labels, axis=0)

        return inp, labels, ori_inp

    def _make_prediction(self, inp, labels):
        """Make prediction using the model."""
        intensity, motion = self.model(inp)
        batch, _, height, width = inp.shape
        # Reshape motion and intensity
        motion_ = motion.reshape(batch, 1, 2, height, width)
        intensity_ = intensity.reshape(batch, 1, 1, height, width)
        # Process last frames
        last_frames = Tensor(labels[:, 0, :, :], ms.float32).unsqueeze(dim=0)
        # Create grid for warping
        sample_tensor = np.zeros((1, 1, 2000, 2000)).astype(np.float32)
        grid = Tensor(make_grid(sample_tensor), ms.float32)
        my_grid = grid.tile((batch, 1, 1, 1))
        # Warp and predict
        last_frames = warp(
            last_frames, motion_[:, 0], my_grid, mode="nearest", padding_mode="border"
        )
        tmp = last_frames
        last_frames = last_frames + intensity_[:, 0]
        pred = last_frames

        return pred, tmp

    def _calculate_metrics(self, pred, labels, ori_inp):
        """Calculate evaluation metrics."""
        minus = pred[0, 0, :, :].asnumpy().T - labels[0, 1, :, :].T
        minus_consist = np.squeeze(pred[0, 0, :, :].asnumpy()).T - np.squeeze(ori_inp).T
        minus_label = np.squeeze(ori_inp).T - np.squeeze(labels[0, 1, :, :]).T
        rmse = np.sqrt(np.mean(np.square(minus)))
        return rmse, minus, minus_consist, minus_label

    def _save_results(self, pred_img, label_img, file_name):
        """Save prediction and label results."""
        self.output_dir_pred = os.path.join(self.config["test"].get("output_path"), "pred")
        os.makedirs(self.output_dir_pred, exist_ok=True)
        self.output_dir_mask = os.path.join(self.config["test"].get("output_path"), "mask")
        os.makedirs(self.output_dir_mask, exist_ok=True)
        np.save(os.path.join(self.output_dir_pred, file_name), pred_img)
        np.save(os.path.join(self.output_dir_mask, file_name), label_img)

    def _visualize_results(
            self, ori_inp, pred, labels, minus, minus_consist, file_name
    ):
        """Visualize and save results."""
        plt.figure(figsize=(15, 10))

        # Input
        plt.subplot(231)
        plt.pcolormesh(ori_inp.T, cmap=cm.gist_ncar_r, vmax=5, vmin=0)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title("input")

        # Prediction
        plt.subplot(232)
        plt.pcolormesh(
            pred[0, 0, :, :].asnumpy().T, cmap=cm.gist_ncar_r, vmax=5, vmin=0
        )
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title("pred")

        # Label
        plt.subplot(233)
        plt.pcolormesh(labels[0, 1, :, :].T, cmap=cm.gist_ncar_r, vmax=5, vmin=0)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title("label")

        # Prediction - Label
        plt.subplot(234)
        plt.pcolormesh(minus, cmap=cm.RdBu_r, vmax=0.2, vmin=-0.2)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title("pred - label")

        # Prediction - Input
        plt.subplot(235)
        plt.pcolormesh(minus_consist, cmap=cm.RdBu_r, vmax=0.05, vmin=-0.05)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title("pred - input")

        # Label - Input
        plt.subplot(236)
        plt.pcolormesh(
            labels[0, 1, :, :].T - ori_inp.T, cmap=cm.RdBu_r, vmax=0.2, vmin=-0.2
        )
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title("label - input")

        plt.tight_layout()
        output_dir = os.path.join(self.config["test"].get("output_path"), "pig")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, file_name[:-4] + ".png"), dpi=600)
        plt.close()

    def evaluate(self):
        """Evaluate the model on test dataset."""
        output_dir_input = os.path.join(self.config["data"].get("data_path"), "ice_input")
        os.makedirs(output_dir_input, exist_ok=True)
        output_dir_label = os.path.join(self.config["data"].get("data_path"), "ice_label")
        os.makedirs(output_dir_label, exist_ok=True)
        test_files = os.listdir(output_dir_input)
        print("Test dataset size:", len(test_files))

        for file in test_files:
            # Prepare paths
            inp_path = os.path.join(output_dir_input, file)
            label_path = os.path.join(
                output_dir_label, file.split(".")[0] + ".label.npy"
            )
            # Process input
            inp, labels, ori_inp = self._process_input(inp_path, label_path)
            # Make prediction
            pred, _ = self._make_prediction(inp, labels)
            # Calculate metrics
            rmse, minus, minus_consist, _ = self._calculate_metrics(
                pred, labels, ori_inp
            )
            # Update metrics
            self.rmse_all += rmse
            if rmse > self.max_rmse:
                self.max_rmse = rmse
            print(file, "RMSE:", rmse)
            # Save results
            pred_img = pred[0, 0, :, :].asnumpy()
            label_img = labels[0, 1, :, :]
            self._save_results(pred_img, label_img, file)
            # Visualize results
            self._visualize_results(ori_inp, pred, labels, minus, minus_consist, file)

        # Print final metrics
        avg_rmse = self.rmse_all / len(test_files)
        print("Average RMSE:", avg_rmse)
        print("Maximum RMSE:", self.max_rmse)
        print("Start evaluate!")
        self.evaluate_width_mask()
        self.evaluate_width_pred()
        self.evaluate_lead_all()
        self.evaluate_acc()
        print("Evaluation completed!")

    def to_lonlat_bin(self, lon, lat, x, y):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        return lon[x][y], lat[x][y]

    def calconnectivity(self, target, x, y):
        """calconnectivity"""
        connectivity = 0
        is_special = 0
        is_endpoint = 0
        l_neighbor = []
        p_neighbor = []
        p_neighbor.append(target[x - 1][y - 1])
        p_neighbor.append(target[x][y - 1])
        p_neighbor.append(target[x + 1][y - 1])
        p_neighbor.append(target[x - 1][y])
        p_neighbor.append(target[x + 1][y])
        p_neighbor.append(target[x - 1][y + 1])
        p_neighbor.append(target[x][y + 1])
        p_neighbor.append(target[x + 1][y + 1])
        for i in range(8):
            if p_neighbor[i] != 0:
                connectivity += 1
        if connectivity >= 3:
            is_special += 1
        elif connectivity == 1:
            is_endpoint += 1
        p_neighbor.sort()
        for j in range(len(p_neighbor) - 1):
            if p_neighbor[j + 1] != p_neighbor[j]:
                l_neighbor.append(p_neighbor[j + 1])
        return is_special, is_endpoint, l_neighbor

    def breakup(self, num_labels, stats, labels):
        """break up"""
        excption = 0
        for area in range(num_labels - 1):
            current_label_x = []
            current_label_y = []
            for area_x in range(stats[area + 1][2]):
                for area_y in range(stats[area + 1][3]):
                    if labels[stats[area + 1][1] + area_y][
                            stats[area + 1][0] + area_x
                    ] == (area + 1):
                        current_label_x.append(stats[area + 1][1] + area_y)
                        current_label_y.append(stats[area + 1][0] + area_x)
            special = 0
            special_points = []
            endpoint = 0
            for p in range(len(current_label_x)):
                is_special, is_endpoint, _ = self.calconnectivity(
                    labels, current_label_x[p], current_label_y[p]
                )
                if is_special == 1:
                    special += 1
                    special_points.append(p)
                if is_endpoint == 1:
                    endpoint += 1

            if special != 0:
                excption = special

            for p in special_points:
                labels[current_label_x[p]][current_label_y[p]] = 0

        return num_labels, stats, labels, excption

    def cal_distance(self, x1, y1, x2, y2):
        return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

    def sort_points(self, endlat, endlon, lat, lon):
        """sort_points"""
        sorted_lat = []
        sorted_lon = []
        sorted_lat.append(endlat)
        sorted_lon.append(endlon)
        ori_len = len(lon)
        while len(sorted_lon) != ori_len:
            minpos = 1
            min_dist = self.cal_distance(lat[0], lon[0], lat[1], lon[1])
            for i in range(len(lat) - 1):
                if self.cal_distance(lat[0], lon[0], lat[i + 1], lon[i + 1]) < min_dist:
                    min_dist = self.cal_distance(lat[0], lon[0], lat[i + 1], lon[i + 1])
                    minpos = i + 1
            sorted_lat.append(lat[minpos])
            sorted_lon.append(lon[minpos])
            del lat[0]
            del lon[0]
            lat.insert(0, lat[minpos - 1])
            lon.insert(0, lon[minpos - 1])
            del lat[minpos]
            del lon[minpos]
        return sorted_lat, sorted_lon

    def visualization(
            self, num_labels, stats, labels, file_lkf, fin_lkf_width_total, label
    ):
        """visualization"""
        m, lon_ori, lat_ori = self._initialize_map_and_coordinates()

        all_save_lons, all_save_lats, all_save_width = self._process_all_labels(
            num_labels, stats, labels, fin_lkf_width_total, lon_ori, lat_ori, m
        )

        self._save_results_and_visualize(
            all_save_lons, all_save_lats, all_save_width, file_lkf, label, m
        )

    def _initialize_map_and_coordinates(self):
        """initialize map_and coordinates"""
        m = Basemap(projection="npstere", boundinglat=70, lon_0=0, resolution="l")

        lon_path = os.path.join(self.config["data"].get("data_path"), "LONC.bin")
        lat_path = os.path.join(self.config["data"].get("data_path"), "LATC.bin")
        data_shape = self.config["data"].get("data_shape")
        lonc = readbin(lon_path, data_shape)
        lon_ori = lonc[1000:3000, 1:2001]
        latc = readbin(lat_path, data_shape)
        lat_ori = latc[1000:3000, 1:2001]

        return m, lon_ori, lat_ori

    def _process_all_labels(self, num_labels, stats, labels, fin_lkf_width_total, lon_ori, lat_ori, m):
        """process all labels"""
        all_save_lons = []
        all_save_lats = []
        all_save_width = []

        for i in range(num_labels - 1):
            if stats[i + 1][4] <= 10:
                continue

            lat, lon = self._process_single_label(i, stats, labels, lon_ori, lat_ori)

            if lat and lon:
                all_save_lats.append(np.array(lat))
                all_save_lons.append(np.array(lon))
                all_save_width.append(fin_lkf_width_total[i])

                xx, yy = m(np.array(lon), np.array(lat))
                plt.plot(xx, yy, ".", ms=1)

        return all_save_lons, all_save_lats, all_save_width

    def _process_single_label(self, label_index, stats, labels, lon_ori, lat_ori):
        """process single label"""
        lat = []
        lon = []
        endlat = 0
        endlon = 0

        current_stat = stats[label_index + 1]
        label_id = label_index + 1

        for j in range(current_stat[2]):
            for k in range(current_stat[3]):
                row_idx = current_stat[1] + k
                col_idx = current_stat[0] + j

                if labels[row_idx][col_idx] == label_id:
                    lat_point, lon_point, is_endpoint = self._process_label_point(
                        labels, row_idx, col_idx, lon_ori, lat_ori
                    )

                    if is_endpoint:
                        endlat = lat_point
                        endlon = lon_point
                        lat.insert(0, lat_point)
                        lon.insert(0, lon_point)
                    else:
                        lat.append(lat_point)
                        lon.append(lon_point)

        if lat and lon:
            lat, lon = self.sort_points(endlat, endlon, lat, lon)

        return lat, lon

    def _process_label_point(self, labels, row_idx, col_idx, lon_ori, lat_ori):
        """process label point"""
        _, fis_endpoint, _ = self.calconnectivity(labels, row_idx, col_idx)
        lon_point, lat_point = self.to_lonlat_bin(lon_ori, lat_ori, row_idx, col_idx)
        is_endpoint = (fis_endpoint == 1)

        return lat_point, lon_point, is_endpoint

    def _save_results_and_visualize(self, all_save_lons, all_save_lats, all_save_width,
                                    file_lkf, label, m):
        """save results and visualize"""
        vis = os.path.join(self.config["test"].get("output_path"), "vis")
        os.makedirs(vis, exist_ok=True)

        if label == "mask":
            self._save_mask_results(all_save_lons, all_save_lats, all_save_width, file_lkf)
            vis_label = vis + "/mask_"
        else:
            self._save_prediction_results(all_save_lons, all_save_lats, all_save_width, file_lkf)
            vis_label = vis + "/pred_"

        m.drawmapboundary()
        m.drawcoastlines()
        m.fillcontinents()

        plt.title("LKF detected")
        plt.savefig(vis_label + str(file_lkf) + ".png", dpi=1200)
        plt.clf()

    def _save_mask_results(self, all_save_lons, all_save_lats, all_save_width, file_lkf):
        """save mask results"""
        self.detect_result_label = os.path.join(
            self.config["test"].get("output_path"), "detect_result_label"
        )
        os.makedirs(self.detect_result_label, exist_ok=True)

        self.detect_result_label_width = os.path.join(
            self.config["test"].get("output_path"), "detect_result_label_width"
        )
        os.makedirs(self.detect_result_label_width, exist_ok=True)

        np.save(
            self.detect_result_label + "/detect_result_" + str(file_lkf) + ".npy",
            np.asarray([all_save_lons, all_save_lats], dtype=object),
        )
        np.save(
            self.detect_result_label_width + "/detect_result_width_" + str(file_lkf) + ".npy",
            np.asarray([all_save_width], dtype=object),
        )

    def _save_prediction_results(self, all_save_lons, all_save_lats, all_save_width, file_lkf):
        """save prediction results"""
        self.detect_result = os.path.join(
            self.config["test"].get("output_path"), "detect_result"
        )
        os.makedirs(self.detect_result, exist_ok=True)

        self.detect_result_width = os.path.join(
            self.config["test"].get("output_path"), "detect_result_width"
        )
        os.makedirs(self.detect_result_width, exist_ok=True)

        np.save(
            self.detect_result + "/detect_result_" + str(file_lkf) + ".npy",
            np.asarray([all_save_lons, all_save_lats], dtype=object),
        )
        np.save(
            self.detect_result_width + "/detect_result_width_" + str(file_lkf) + ".npy",
            np.asarray([all_save_width], dtype=object),
        )

    def evaluate_width_mask(self):
        """evaluate_width_mask"""
        kernel_size = 3
        padding = int((kernel_size - 1) / 2)
        filename_root = self.output_dir_mask
        for file in os.listdir(filename_root):
            print(file)
            plt.clf()
            filename = os.path.join(filename_root, file)
            gt = np.load(filename)
            gt_pad_img = np.pad(
                gt, ((2, 2), (2, 2)), "constant", constant_values=(np.nan, np.nan)
            )
            gt_dect_result = np.zeros((gt_pad_img.shape[0], gt_pad_img.shape[1]))
            for i in range(gt.shape[0]):
                for j in range(gt.shape[1]):
                    gt_local_mat = gt_pad_img[
                        i + padding - 6 : i + padding + 6,
                        j + padding - 6 : j + padding + 6,
                    ]
                    gt_local_sit = gt_local_mat.flatten()
                    gt_num_nan = len(gt_local_sit[np.isnan(gt_local_sit)])
                    gt_local_sit[np.isnan(gt_local_sit)] = 0
                    if len(gt_local_sit) > gt_num_nan:
                        gt_local_mean = np.sum(gt_local_sit) / (
                            len(gt_local_sit) - gt_num_nan
                        )
                        gt_local_std = np.std(gt_local_sit)
                        if gt_pad_img[i][j] < gt_local_mean - gt_local_std:
                            gt_dect_result[i][j] = 1
            gt_dect_result = gt_dect_result.astype(np.uint8) * 255
            _, _, gt_ori_stats, _ = (
                cv2.connectedComponentsWithStats(gt_dect_result)
            )
            gt_dect_result = gt_dect_result / 255
            gt_skeleton0 = morphology.skeletonize(gt_dect_result)
            gt_dect_result = gt_skeleton0.astype(np.uint8) * 255
            gt_num_labels, gt_labels, gt_stats, _ = (
                cv2.connectedComponentsWithStats(gt_dect_result)
            )
            gt_lkf_width_total = []
            for ii in range(1, len(gt_ori_stats)):
                gt_lkf_width = gt_ori_stats[ii][4] / gt_stats[ii][4]
                gt_lkf_width_total.append(gt_lkf_width)
            gt_exc = 1
            while gt_exc != 0:
                gt_num_labels, gt_stats, gt_labels, gt_exc = self.breakup(
                    gt_num_labels, gt_stats, gt_labels
                )
                print(
                    "========================================= break =================================================="
                )

            self.visualization(
                gt_num_labels,
                gt_stats,
                gt_labels,
                file[:-4],
                gt_lkf_width_total,
                "mask",
            )

    def evaluate_width_pred(self):
        """evaluate_width_pred"""
        kernel_size = 3

        padding = int((kernel_size - 1) / 2)

        filename_p_root = self.output_dir_pred
        hccf_path = os.path.join(self.config["data"].get("data_path"), "hFacC.data")
        landmask = readbin(hccf_path, self.config["data"].get("data_shape"))
        landmask = landmask[1000:3000, 1:2001]

        for file in os.listdir(filename_p_root):
            print(file)
            plt.clf()
            filename_p = os.path.join(filename_p_root, file)
            img = np.load(filename_p)
            img[landmask == 0] = 0
            img[img < 0] = 0
            pad_img = np.pad(
                img, ((2, 2), (2, 2)), "constant", constant_values=(np.nan, np.nan)
            )

            dect_result = np.zeros((pad_img.shape[0], pad_img.shape[1]))

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    local_mat = pad_img[
                        i + padding - 6 : i + padding + 6,
                        j + padding - 6 : j + padding + 6,
                    ]
                    local_sit = local_mat.flatten()
                    num_nan = len(local_sit[np.isnan(local_sit)])
                    local_sit[np.isnan(local_sit)] = 0
                    if len(local_sit) > num_nan:
                        local_mean = np.sum(local_sit) / (len(local_sit) - num_nan)
                        local_std = np.std(local_sit)
                        if pad_img[i][j] < local_mean - local_std:
                            dect_result[i][j] = 1
            dect_result = dect_result.astype(np.uint8) * 255
            _, _, ori_stats, _ = (
                cv2.connectedComponentsWithStats(dect_result)
            )
            dect_result = dect_result / 255

            skeleton0 = morphology.skeletonize(dect_result)
            dect_result = skeleton0.astype(np.uint8) * 255

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                dect_result
            )

            lkf_width_total = []
            for ii in range(1, len(ori_stats)):
                lkf_width = ori_stats[ii][4] / stats[ii][4]
                lkf_width_total.append(lkf_width)

            exc = 1
            while exc != 0:
                num_labels, stats, labels, exc = self.breakup(num_labels, stats, labels)
                print(
                    "========================================= break =================================================="
                )
            self.visualization(
                num_labels, stats, labels, file[:-4], lkf_width_total, "pred"
            )

    def calc_dis_v3(self, lkf_fcst, lkf_sar):
        """calc_dis"""
        dis_cutoff = 50.0
        imax = len(lkf_fcst[0])
        jmax = len(lkf_sar[0])
        dismin = np.nan * np.zeros((imax, jmax))
        disnrst = np.empty([])
        dis_index = []
        for j in tqdm(np.arange(jmax)):
            lon2, lat2 = lkf_sar[0][j], lkf_sar[1][j]
            lkf_sar_len = haversine.haversine((lat2[0], lon2[0]), (lat2[-1], lon2[-1]))
            if lkf_sar_len <= dis_cutoff:
                dismin[:, j] = np.nan
                continue
            for i in np.arange(imax):
                lon1, lat1 = lkf_fcst[0][i], lkf_fcst[1][i]
                lkf_fcst_len = haversine.haversine(
                    (lat1[0], lon1[0]), (lat1[-1], lon1[-1])
                )
                if lkf_fcst_len <= dis_cutoff:
                    dismin[i, j] = np.nan
                    continue
                dismin[i, j] = self._calculate_min_distance(lat1, lon1, lat2, lon2)
        disnrst = np.nanmin(dismin, axis=0)
        print(np.nanmax(disnrst), np.nanmin(disnrst), np.nanmean(disnrst))
        for index in range(jmax):
            dis_index_now = np.where(dismin[:, index] == disnrst[index])
            dis_index_now = np.squeeze(dis_index_now)
            if dis_index_now.size == 0:
                dis_index.append(999)
            elif dis_index_now.size > 1:
                dis_index.append(np.int32(dis_index_now[0]))
            else:
                dis_index.append(np.int32(dis_index_now))
        dis_index = np.array(dis_index)

        return disnrst, dis_index

    def _calculate_min_distance(self, lat1, lon1, lat2, lon2):
        """calculate_min_distance"""
        dis = np.zeros((len(lon1), len(lon2)))
        for ii in np.arange(len(lon1)):
            for jj in np.arange(len(lon2)):
                dis[ii, jj] = haversine.haversine(
                    (lat1[ii], lon1[ii]), (lat2[jj], lon2[jj])
                )
        dis1 = np.nanmin(dis, axis=1)
        dis2 = np.nanmin(dis, axis=0)
        dis1 = np.sort(dis1)
        dis2 = np.sort(dis2)
        n = min([np.sum(~np.isnan(dis1)), np.sum(~np.isnan(dis2))])
        return (np.sum(dis1[:n]) + np.sum(dis2[:n])) / (2 * n)

    def visual(self, pred, label, pair, lkf_name):
        """visual"""
        m = Basemap(projection="npstere", boundinglat=70, lon_0=0, resolution="l")
        plt.clf()
        for i in range(len(pair)):
            if pair[i] == 999:
                continue

            pred_lon = pred[0][pair[i]]
            pred_lat = pred[1][pair[i]]

            label_lon = label[0][i]
            label_lat = label[1][i]

            xx, yy = m(np.array(pred_lon), np.array(pred_lat))
            lxx, lyy = m(np.array(label_lon), np.array(label_lat))

            plt.subplot(121)
            plt.plot(xx, yy, ".", ms=1)
            m.drawmapboundary()
            m.drawcoastlines()
            m.fillcontinents()
            plt.title("pred")
            plt.subplot(122)
            plt.plot(lxx, lyy, ".", ms=1)
            m.drawmapboundary()
            m.drawcoastlines()
            m.fillcontinents()
            plt.title("label")
        output_path = self.config["test"].get("output_path")
        out_path = f"{output_path.rstrip('/')}/{lkf_name}.png"
        plt.savefig(out_path, dpi=600)

    def evaluate_dis(self, pred, label, pair):
        """evaluate_dis"""
        avg_diff = 0
        for i in range(len(pair)):
            if pair[i] == 999:
                continue
            pred_lon = pred[0][pair[i]]
            pred_lat = pred[1][pair[i]]

            label_lon = label[0][i]
            label_lat = label[1][i]

            lkf_pred_len = haversine.haversine(
                (pred_lat[0], pred_lon[0]), (pred_lat[-1], pred_lon[-1])
            )
            lkf_label_len = haversine.haversine(
                (label_lat[0], label_lon[0]), (label_lat[-1], label_lon[-1])
            )

            diff = abs(lkf_label_len - lkf_pred_len) / lkf_label_len
            avg_diff = avg_diff + diff

        avg_diff = avg_diff / len(pair)

        return avg_diff

    def get_degree(self, lata, lona, latb, lonb):
        """
        Args:
            point p1(latA, lonA)
            point p2(latB, lonB)
        Returns:
            bearing between the two GPS points,
            default: the basis of heading direction is north
        """
        radlata = radians(lata)
        radlona = radians(lona)
        radlatb = radians(latb)
        radlonb = radians(lonb)
        dlon = radlonb - radlona
        y = sin(dlon) * cos(radlatb)
        x = cos(radlata) * sin(radlatb) - sin(radlata) * cos(radlatb) * cos(dlon)
        brng = degrees(atan2(y, x))
        brng = (brng + 360) % 360
        return brng

    def evaluate_degree(self, pred, label, pair):
        """evaluate_degree"""
        avg_diff = 0
        for i in range(len(pair)):
            if pair[i] == 999:
                continue
            pred_lon = pred[0][pair[i]]
            pred_lat = pred[1][pair[i]]

            label_lon = label[0][i]
            label_lat = label[1][i]

            lkf_pred_degree = self.get_degree(
                pred_lat[0], pred_lon[0], pred_lat[2], pred_lon[2]
            )
            lkf_label_degree = self.get_degree(
                label_lat[0], label_lon[0], label_lat[2], label_lon[2]
            )

            diff = abs(lkf_pred_degree - lkf_label_degree)
            if diff >= 180:
                diff = 360 - diff
            avg_diff = avg_diff + diff

        avg_diff = avg_diff / len(pair)

        return avg_diff

    def evaluate_width(self, pred_width, label_width, pair):
        """evaluate_width"""
        avg_diff = 0
        for i in range(len(pair)):
            if pair[i] == 999:
                continue

            lkf_pred_width = pred_width[0][pair[i]]
            lkf_label_width = label_width[0][i]

            diff = abs(lkf_label_width - lkf_pred_width) / lkf_label_width
            avg_diff = avg_diff + diff

        avg_diff = avg_diff / len(pair)

        return avg_diff

    def evaluate_lead_all(self):
        """evaluate_lead_all"""
        fcst_root_path = self.detect_result
        model_root_path = self.detect_result_label
        fcst_width_path = self.detect_result_width
        model_width_path = self.detect_result_label_width
        lonc_path = os.path.join(self.config["data"].get("data_path"), "LONC.bin")
        lonc = readbin(lonc_path, self.config["data"].get("data_shape"))
        lonc = lonc[1000:3000, 1:2001]
        avg_dis_diff = 0
        avg_degree_diff = 0
        avg_width_diff = 0
        max_dis_diff = 0
        max_degree_diff = 0
        max_width_diff = 0
        for lkf_fcst_file in os.listdir(fcst_root_path):
            lkf_fcst_path = os.path.join(fcst_root_path, lkf_fcst_file)
            lkf_model_path = os.path.join(model_root_path, lkf_fcst_file)
            lkf_fcst_width_file = (
                "detect_result_width_ice_input_" + lkf_fcst_file.split("_")[4]
            )
            lkf_fcst_width_path = os.path.join(fcst_width_path, lkf_fcst_width_file)
            lkf_model_width_path = os.path.join(model_width_path, lkf_fcst_width_file)
            lkf_fcst = np.load(lkf_fcst_path, allow_pickle=True)
            lkf_model = np.load(lkf_model_path, allow_pickle=True)
            lkf_fcst_width = np.load(lkf_fcst_width_path, allow_pickle=True)
            lkf_model_width = np.load(lkf_model_width_path, allow_pickle=True)
            _, pair = self.calc_dis_v3(lkf_fcst, lkf_model)
            self.visual(lkf_fcst, lkf_model, pair, lkf_fcst_file[:-4])
            dis_width = self.evaluate_width(lkf_fcst_width, lkf_model_width, pair)
            print(lkf_fcst_file, "dis width is: ", dis_width)
            if dis_width > max_width_diff:
                max_width_diff = dis_width
            avg_width_diff = avg_width_diff + dis_width
            dis_diff = self.evaluate_dis(lkf_fcst, lkf_model, pair)
            print(lkf_fcst_file, "dis diff is: ", dis_diff)
            if dis_diff > max_dis_diff:
                max_dis_diff = dis_diff
            avg_dis_diff = avg_dis_diff + dis_diff
            degree_diff = self.evaluate_degree(lkf_fcst, lkf_model, pair)
            if degree_diff > max_degree_diff:
                max_degree_diff = degree_diff
            print(lkf_fcst_file, "degree diff is: ", degree_diff)
            avg_degree_diff = avg_degree_diff + degree_diff

        print("avg diff width: ", avg_width_diff / len(os.listdir(fcst_root_path)))
        print("avg diff dis: ", avg_dis_diff / len(os.listdir(fcst_root_path)))
        print("avg diff degree: ", avg_degree_diff / len(os.listdir(fcst_root_path)))

        print("max diff width: ", max_width_diff)
        print("max diff dis: ", max_dis_diff)
        print("max diff degree: ", max_degree_diff)

    def lonlat2xy2km(self, l_lon, lon):
        x = int(np.argwhere(l_lon == lon)[0][0])
        y = int(np.argwhere(l_lon == lon)[0][1])
        return x, y

    def evaluate_acc(self):
        """evaluate_acc"""
        fcst_root_path = self.detect_result
        model_root_path = self.detect_result_label
        lon_path = os.path.join(self.config["data"].get("data_path"), "LONC.bin")
        lon = readbin(lon_path, self.config["data"].get("data_shape"))
        lon = lon[1000:3000, 1:2001]

        avg_acc = 0

        for lkf_fcst_file in os.listdir(fcst_root_path):
            lkf_fcst_path = os.path.join(fcst_root_path, lkf_fcst_file)
            lkf_model_path = os.path.join(model_root_path, lkf_fcst_file)
            lkf_fcst = np.load(lkf_fcst_path, allow_pickle=True)
            lkf_model = np.load(lkf_model_path, allow_pickle=True)

            pred = np.zeros((2000, 2000))
            label = np.zeros((2000, 2000))
            for i in range(lkf_fcst.shape[1]):
                for j in range(len(lkf_fcst[0][i])):
                    if lkf_fcst[0][i][j] == 0:
                        continue
                    px, py = self.lonlat2xy2km(lon, lkf_fcst[0][i][j])
                    pred[px][py] = 255

            for ii in range(lkf_model.shape[1]):
                for jj in range(len(lkf_model[0][ii])):
                    if lkf_model[0][ii][jj] == 0:
                        continue
                    lx, ly = self.lonlat2xy2km(lon, lkf_model[0][ii][jj])
                    label[lx][ly] = 255

            acc_num = 0
            for w in range(label.shape[0]):
                for h in range(label.shape[1]):
                    if pred[w][h] == label[w][h]:
                        acc_num = acc_num + 1
            acc = acc_num / (label.shape[0] * label.shape[1])
            print("acc for ", lkf_fcst_file, " is:", acc)
            avg_acc = avg_acc + acc

        avg_acc = avg_acc / (len(os.listdir(fcst_root_path)))
        print("avg acc is: ", avg_acc)
