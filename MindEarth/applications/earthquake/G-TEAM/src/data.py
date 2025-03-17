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
"load diting data"
import os
import pickle
import glob
import h5py
import numpy as np

import mindspore
from mindspore.dataset import Dataset

# degrees to kilometers
D2KM = 111.19492664455874


def load_pickle_data(filename):
    """Load and deserialize data from a pickle file."""
    with open(filename, "rb") as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data


def save_pickle_data(filename, data):
    """Serialize and save data to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")


def load_data(cfg):
    """Load preprocessed seismic data from a configured pickle file."""
    data_path = glob.glob(os.path.join(cfg["data"].get("root_dir"), "*.hdf5"))[0]
    file_basename = os.path.basename(data_path).split(".")[0]
    filename = os.path.join(
        cfg["data"].get("root_dir"), f"{file_basename}_test_filter_pga.pkl"
    )
    loaded_pickle_data = load_pickle_data(filename)
    _, evt_metadata, meta_data, data_data, evt_key, _ = loaded_pickle_data
    return data_data, evt_key, evt_metadata, meta_data, data_path


def detect_location_keys(columns):
    """Identify standardized location keys from column headers."""
    candidates = [
        ["LAT", "Latitude(°)", "Latitude"],
        ["LON", "Longitude(°)", "Longitude"],
        ["DEPTH", "JMA_Depth(km)", "Depth(km)", "Depth/Km"],
    ]

    coord_keys = []
    for keyset in candidates:
        for key in keyset:
            if key in columns:
                coord_keys += [key]
                break

    if len(coord_keys) != len(candidates):
        raise ValueError("Unknown location key format")

    return coord_keys


class EarthquakeDataset(Dataset):
    """
    Dataset class for loading and processing seismic event data.
    Handles waveform loading, magnitude-based resampling, PGA target processing,
    and batch preparation for earthquake analysis models.
    Key Features:
        Batch processing of seismic waveforms and metadata
        Magnitude-based data resampling for class balance
        PGA (Peak Ground Acceleration) target handling
        HDF5 waveform data loading
        Flexible data shuffling and oversampling
    """

    def __init__(
            self,
            data_path,
            event_key,
            data,
            event_metadata,
            batch_size=32,
            shuffle=True,
            oversample=1,
            magnitude_resampling=3,
            min_upsample_magnitude=2,
            pga_targets=None,
            pga_mode=False,
            pga_key="pga",
            coord_keys=None,
            **kwargs,
    ):

        super(EarthquakeDataset, self).__init__()

        self.data_path = data_path
        self.event_key = event_key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.metadata = data["coords"]
        self.event_metadata = event_metadata
        self.pga = data[pga_key]
        self.triggers = data["p_picks"]
        self.oversample = oversample

        self.pga_mode = pga_mode
        self.pga_targets = pga_targets

        self.base_indexes = np.arange(self.event_metadata.shape[0])
        self.reverse_index = None

        if magnitude_resampling > 1:
            magnitude = self.event_metadata[kwargs["key"]].values
            for i in np.arange(min_upsample_magnitude, 9):
                ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                self.base_indexes = np.concatenate(
                    (
                        self.base_indexes,
                        np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1)),
                    )
                )

        if pga_mode and pga_targets is not None:
            new_base_indexes = []
            self.reverse_index = []
            c = 0
            for idx in self.base_indexes:
                num_samples = (len(self.pga[idx]) - 1) // pga_targets + 1
                new_base_indexes += [(idx, i) for i in range(num_samples)]
                self.reverse_index += [c]
                c += num_samples
            self.reverse_index += [c]
            self.base_indexes = new_base_indexes
        if coord_keys is None:
            self.coord_keys = detect_location_keys(event_metadata.columns)
        else:
            self.coord_keys = coord_keys
        self.use_shuffle()

    def __len__(self):
        """get length"""
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """Load data."""
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_data = {
            "indexes": batch_indexes,
            "waveforms": [],
            "metadata": [],
            "pga": [],
            "p_picks": [],
            "event_info": [],
        }
        if self.pga_mode:
            batch_data["pga_indexes"] = [x[1] for x in batch_indexes]
            batch_data["indexes"] = [x[0] for x in batch_indexes]
        for idx in batch_data["indexes"]:
            event = self.event_metadata.iloc[idx]
            event_name = str(event[self.event_key])
            waveform_data = self._load_waveform_data(event_name)
            batch_data["waveforms"].append(waveform_data)
            batch_data["metadata"].append(self.metadata[idx])
            batch_data["pga"].append(self.pga[idx])
            batch_data["p_picks"].append(self.triggers[idx])
            batch_data["event_info"].append(event)

        return batch_data

    def _load_waveform_data(self, event_name):
        """load waveform data"""
        with h5py.File(self.data_path, "r") as f:
            if "data" not in f or event_name not in f["data"]:
                return None
            g_event = f["data"][event_name]
            if "waveforms" not in g_event:
                return None
            return g_event["waveforms"][:, :, :]

    def use_shuffle(self):
        """shuffle index"""
        self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
        if self.shuffle:
            np.random.shuffle(self.indexes)


class DataProcessor:
    """
    A data processor for seismic event analysis that handles waveform preprocessing,
    station selection, and target preparation for machine learning models.
    Key functionalities:
        Batch processing of seismic waveforms and metadata
        Station selection strategies for efficient processing
        Multiple preprocessing techniques (cutout, integration, etc.)
        Coordinate transformations and target preparations
        PGA (Peak Ground Acceleration) target handling
        Data augmentation techniques (label smoothing, station blinding)
    """

    def __init__(
            self,
            waveform_shape=(3000, 6),
            max_stations=None,
            cutout=None,
            sliding_window=False,
            windowlen=3000,
            coords_target=True,
            pos_offset=(-21, -69),
            label_smoothing=False,
            station_blinding=False,
            pga_targets=None,
            adjust_mean=True,
            transform_target_only=False,
            trigger_based=None,
            disable_station_foreshadowing=False,
            selection_skew=None,
            pga_from_inactive=False,
            integrate=False,
            sampling_rate=100.0,
            select_first=False,
            scale_metadata=True,
            p_pick_limit=5000,
            pga_mode=False,
            no_event_token=False,
            pga_selection_skew=None,
            **kwargs,
    ):
        self.waveform_shape = waveform_shape
        self.max_stations = max_stations
        self.cutout = cutout
        self.sliding_window = sliding_window
        self.windowlen = windowlen
        self.coords_target = coords_target
        self.pos_offset = pos_offset
        self.label_smoothing = label_smoothing
        self.station_blinding = station_blinding
        self.pga_targets = pga_targets
        self.adjust_mean = adjust_mean
        self.transform_target_only = transform_target_only
        self.trigger_based = trigger_based
        self.disable_station_foreshadowing = disable_station_foreshadowing
        self.selection_skew = selection_skew
        self.pga_from_inactive = pga_from_inactive
        self.integrate = integrate
        self.sampling_rate = sampling_rate
        self.select_first = select_first
        self.scale_metadata = scale_metadata
        self.p_pick_limit = p_pick_limit
        self.pga_mode = pga_mode
        self.no_event_token = no_event_token
        self.pga_selection_skew = pga_selection_skew
        self.key = kwargs["key"]

    def process_batch(self, batch_data):
        """Main method to process a batch of data, now decomposed into smaller functions."""
        (
            indexes,
            waveforms_list,
            metadata_list,
            pga_list,
            p_picks_list,
            event_info_list,
            pga_indexes,
        ) = self._extract_batch_data(batch_data)

        true_batch_size = len(indexes)
        true_max_stations_in_batch = self._get_max_stations_in_batch(metadata_list)
        waveforms, metadata, pga, full_p_picks, p_picks, reverse_selections = (
            self._initialize_arrays(
                true_batch_size, true_max_stations_in_batch, metadata_list
            )
        )
        waveforms, metadata, pga, p_picks, full_p_picks, reverse_selections = (
            self._process_stations(
                waveforms_list,
                metadata_list,
                pga_list,
                p_picks_list,
                waveforms,
                metadata,
                pga,
                p_picks,
                full_p_picks,
            )
        )
        magnitude, target = self._process_magnitude_and_targets(event_info_list)
        org_waveform_length = waveforms.shape[2]
        waveforms, _ = self._process_waveforms(waveforms, org_waveform_length, p_picks)
        metadata, target = self._transform_locations(metadata, target)
        magnitude = self._apply_label_smoothing(magnitude)
        metadata, pga = self._adjust_metadata_and_pga(metadata, pga)
        pga_values, pga_targets_data = self._process_pga_targets(
            true_batch_size,
            pga,
            metadata,
            pga_indexes,
            reverse_selections,
            full_p_picks,
            indexes,
        )
        waveforms, metadata = self._apply_station_blinding(waveforms, metadata)
        waveforms, metadata = self._handle_stations_without_trigger(waveforms, metadata)
        waveforms, metadata = self._ensure_no_empty_arrays(waveforms, metadata)
        inputs, outputs = self._prepare_model_io(
            waveforms, metadata, magnitude, target, pga_targets_data, pga_values
        )

        return inputs, outputs

    def _extract_batch_data(self, batch_data):
        """Extract data from the batch dictionary."""
        indexes = batch_data["indexes"]
        waveforms_list = batch_data["waveforms"]
        metadata_list = batch_data["metadata"]
        pga_list = batch_data["pga"]
        p_picks_list = batch_data["p_picks"]
        event_info_list = batch_data["event_info"]
        pga_indexes = batch_data.get("pga_indexes", None)

        return (
            indexes,
            waveforms_list,
            metadata_list,
            pga_list,
            p_picks_list,
            event_info_list,
            pga_indexes,
        )

    def _get_max_stations_in_batch(self, metadata_list):
        """Calculate the maximum number of stations in the batch."""
        return max(
            [len(m) for m in metadata_list if m is not None] + [self.max_stations]
        )

    def _initialize_arrays(
            self, true_batch_size, true_max_stations_in_batch, metadata_list
    ):
        """Initialize arrays for batch processing."""
        waveforms = np.zeros([true_batch_size, self.max_stations] + self.waveform_shape)
        metadata = np.zeros(
            (true_batch_size, true_max_stations_in_batch) + metadata_list[0].shape[1:]
        )
        pga = np.zeros((true_batch_size, true_max_stations_in_batch))
        full_p_picks = np.zeros((true_batch_size, true_max_stations_in_batch))
        p_picks = np.zeros((true_batch_size, self.max_stations))
        reverse_selections = []

        return waveforms, metadata, pga, full_p_picks, p_picks, reverse_selections

    def _process_stations(
            self,
            waveforms_list,
            metadata_list,
            pga_list,
            p_picks_list,
            waveforms,
            metadata,
            pga,
            p_picks,
            full_p_picks,
    ):
        """Process stations and waveforms for each item in the batch."""
        reverse_selections = []

        for i, (waveform_data, meta, pga_data, p_pick_data) in enumerate(
                zip(waveforms_list, metadata_list, pga_list, p_picks_list)
        ):
            if waveform_data is None:
                continue

            num_stations = waveform_data.shape[0]

            if num_stations <= self.max_stations:
                waveforms[i, :num_stations] = waveform_data
                metadata[i, : len(meta)] = meta
                pga[i, : len(pga_data)] = pga_data
                p_picks[i, : len(p_pick_data)] = p_pick_data
                reverse_selections += [[]]
            else:
                selection = self._select_stations(num_stations, p_pick_data)

                metadata[i, : len(selection)] = meta[selection]
                pga[i, : len(selection)] = pga_data[selection]
                full_p_picks[i, : len(selection)] = p_pick_data[selection]

                tmp_reverse_selection = [0 for _ in selection]
                for j, s in enumerate(selection):
                    tmp_reverse_selection[s] = j
                reverse_selections += [tmp_reverse_selection]

                selection = selection[: self.max_stations]
                waveforms[i] = waveform_data[selection]
                p_picks[i] = p_pick_data[selection]

        return waveforms, metadata, pga, p_picks, full_p_picks, reverse_selections

    def _select_stations(self, num_stations, p_pick_data):
        """Select stations based on configured strategy."""
        if self.selection_skew is None:
            selection = np.arange(0, num_stations)
            np.random.shuffle(selection)
        else:
            tmp_p_picks = p_pick_data.copy()
            mask = np.logical_and(tmp_p_picks <= 0, tmp_p_picks > self.p_pick_limit)
            tmp_p_picks[mask] = min(np.max(tmp_p_picks), self.p_pick_limit)
            coeffs = np.exp(-tmp_p_picks / self.selection_skew)
            coeffs *= np.random.random(coeffs.shape)
            coeffs[p_pick_data == 0] = 0
            coeffs[p_pick_data > self.waveform_shape[0]] = 0
            selection = np.argsort(-coeffs)

        if self.select_first:
            selection = np.argsort(p_pick_data)

        return selection

    def _process_magnitude_and_targets(self, event_info_list):
        """Process magnitude and coordinate targets."""
        magnitude = np.array([e[self.key] for e in event_info_list], dtype=np.float32)
        target = None

        if self.coords_target:
            coord_keys = detect_location_keys(
                [col for e in event_info_list for col in e.index]
            )
            target = np.array(
                [[e[k] for k in coord_keys] for e in event_info_list], dtype=np.float32
            )

        magnitude = np.expand_dims(np.expand_dims(magnitude, axis=-1), axis=-1)
        return magnitude, target

    def _process_waveforms(self, waveforms, org_waveform_length, p_picks):
        """Apply cutout, sliding window, trigger-based, and integration transformations to waveforms."""
        cutout = org_waveform_length

        if self.cutout:
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = np.random.randint(
                    max(windowlen, self.cutout[0]),
                    min(waveforms.shape[2], self.cutout[1]) + 1,
                )
                waveforms = waveforms[:, :, window_end - windowlen : window_end]

                cutout = window_end
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms, axis=2, keepdims=True)
            else:
                cutout = np.random.randint(*self.cutout)
                if self.adjust_mean:
                    waveforms -= np.mean(
                        waveforms[:, :, : cutout + 1], axis=2, keepdims=True
                    )
                waveforms[:, :, cutout:] = 0

        if self.trigger_based:
            p_picks[p_picks <= 0] = org_waveform_length
            waveforms[cutout < p_picks, :, :] = 0

        if self.integrate:
            waveforms = np.cumsum(waveforms, axis=2) / self.sampling_rate

        return waveforms, cutout

    def _transform_locations(self, metadata, target):
        """Transform locations using the location_transformation method."""
        if self.coords_target:
            metadata, target = self.location_transformation(metadata, target)
        else:
            metadata = self.location_transformation(metadata)
        return metadata, target

    def _apply_label_smoothing(self, magnitude):
        """Apply label smoothing to magnitude if enabled."""
        if self.label_smoothing:
            magnitude += (
                (magnitude > 4)
                * np.random.randn(magnitude.shape[0]).reshape(magnitude.shape)
                * (magnitude - 4)
                * 0.05
            )
        return magnitude

    def _adjust_metadata_and_pga(self, metadata, pga):
        """Adjust metadata and PGA arrays based on configuration."""
        if not self.pga_from_inactive and not self.pga_mode:
            metadata = metadata[:, : self.max_stations]
            pga = pga[:, : self.max_stations]
        return metadata, pga

    def _process_pga_targets(
            self,
            true_batch_size,
            pga,
            metadata,
            pga_indexes,
            reverse_selections,
            full_p_picks,
            indexes,
    ):
        """Process PGA targets if enabled."""
        pga_values = None
        pga_targets_data = None

        if self.pga_targets:
            pga_values = np.zeros((true_batch_size, self.pga_targets))
            pga_targets_data = np.zeros((true_batch_size, self.pga_targets, 3))

            if self.pga_mode and pga_indexes is not None:
                self._process_pga_mode(
                    pga_values,
                    pga_targets_data,
                    pga,
                    metadata,
                    pga_indexes,
                    reverse_selections,
                )
            else:
                self._process_pga_normal(
                    pga_values, pga_targets_data, pga, metadata, full_p_picks, indexes
                )

            pga_values = pga_values.reshape((true_batch_size, self.pga_targets, 1, 1))

        return pga_values, pga_targets_data

    def _process_pga_mode(
            self,
            pga_values,
            pga_targets_data,
            pga,
            metadata,
            pga_indexes,
            reverse_selections,
    ):
        """Process PGA in PGA mode."""
        for i in range(len(pga_values)):
            pga_index = pga_indexes[i]
            if reverse_selections[i]:
                sorted_pga = pga[i, reverse_selections[i]]
                sorted_metadata = metadata[i, reverse_selections[i]]
            else:
                sorted_pga = pga[i]
                sorted_metadata = metadata[i]
            pga_values_pre = sorted_pga[
                pga_index * self.pga_targets : (pga_index + 1) * self.pga_targets
            ]
            pga_values[i, : len(pga_values_pre)] = pga_values_pre
            pga_targets_pre = sorted_metadata[
                pga_index * self.pga_targets : (pga_index + 1) * self.pga_targets,
                :,
            ]
            if pga_targets_pre.shape[-1] == 4:
                pga_targets_pre = pga_targets_pre[:, (0, 1, 3)]
            pga_targets_data[i, : len(pga_targets_pre), :] = pga_targets_pre

    def _process_pga_normal(
            self, pga_values, pga_targets_data, pga, metadata, full_p_picks, indexes
    ):
        """Process PGA in normal mode."""
        pga[np.logical_or(np.isnan(pga), np.isinf(pga))] = 0
        for i in range(pga_values.shape[0]):
            active = np.where(pga[i] != 0)[0]
            if not active:
                raise ValueError(f"Found event without PGA idx={indexes[i]}")
            while len(active) < self.pga_targets:
                active = np.repeat(active, 2)

            if self.pga_selection_skew is not None:
                active = self._select_pga_with_skew(active, full_p_picks[i])
            else:
                np.random.shuffle(active)

            samples = active[: self.pga_targets]
            if metadata.shape[-1] == 3:
                pga_targets_data[i] = metadata[i, samples, :]
            else:
                full_targets = metadata[i, samples]
                pga_targets_data[i] = full_targets[:, (0, 1, 3)]
            pga_values[i] = pga[i, samples]

    def _select_pga_with_skew(self, active, full_p_picks):
        """Select PGA with skew-based selection."""
        active_p_picks = full_p_picks[active]
        mask = np.logical_and(active_p_picks <= 0, active_p_picks > self.p_pick_limit)
        active_p_picks[mask] = min(np.max(active_p_picks), self.p_pick_limit)
        coeffs = np.exp(-active_p_picks / self.pga_selection_skew)
        coeffs *= np.random.random(coeffs.shape)
        return active[np.argsort(-coeffs)]

    def _apply_station_blinding(self, waveforms, metadata):
        """Apply station blinding if enabled."""
        if self.station_blinding:
            mask = np.zeros(waveforms.shape[:2], dtype=bool)

            for i in range(waveforms.shape[0]):
                active = np.where((waveforms[i] != 0).any(axis=(1, 2)))[0]
                if not active == 0:
                    active = np.zeros(1, dtype=int)
                blind_length = np.random.randint(0, len(active))
                np.random.shuffle(active)
                blind = active[:blind_length]
                mask[i, blind] = True

            waveforms[mask] = 0
            metadata[mask] = 0

        return waveforms, metadata

    def _handle_stations_without_trigger(self, waveforms, metadata):
        """Handle stations without trigger signal."""
        stations_without_trigger = (metadata != 0).any(axis=2) & (waveforms == 0).all(
            axis=(2, 3)
        )

        if self.disable_station_foreshadowing:
            metadata[stations_without_trigger] = 0
        else:
            waveforms[stations_without_trigger, 0, 0] += 1e-9

        return waveforms, metadata

    def _ensure_no_empty_arrays(self, waveforms, metadata):
        """Ensure there are no empty arrays in the batch."""
        mask = np.logical_and(
            (metadata == 0).all(axis=(1, 2)), (waveforms == 0).all(axis=(1, 2, 3))
        )
        waveforms[mask, 0, 0, 0] = 1e-9
        metadata[mask, 0, 0] = 1e-9

        return waveforms, metadata

    def _prepare_model_io(
            self, waveforms, metadata, magnitude, target, pga_targets_data, pga_values
    ):
        """Prepare model inputs and outputs."""
        inputs = [
            mindspore.tensor(waveforms, dtype=mindspore.float32),
            mindspore.tensor(metadata, dtype=mindspore.float32),
        ]
        outputs = []

        if not self.no_event_token:
            outputs += [mindspore.tensor(magnitude, dtype=mindspore.float32)]

            if self.coords_target:
                target = np.expand_dims(target, axis=-1)
                outputs += [mindspore.tensor(target, dtype=mindspore.float32)]

        if self.pga_targets and pga_values is not None and pga_targets_data is not None:
            inputs += [mindspore.tensor(pga_targets_data, dtype=mindspore.float32)]
            outputs += [mindspore.tensor(pga_values, dtype=mindspore.float32)]

        return inputs, outputs

    def location_transformation(self, metadata, target=None):
        """
        Apply transformations to the metadata and optionally to the target.
        Adjusts positions based on a positional offset and scales the data if required.
        """
        transform_target_only = self.transform_target_only
        metadata = metadata.copy()

        metadata_old = metadata
        metadata = metadata.copy()
        mask = (metadata == 0).all(axis=2)

        if target is not None:
            target[:, 0] -= self.pos_offset[0]
            target[:, 1] -= self.pos_offset[1]

        metadata[:, :, 0] -= self.pos_offset[0]
        metadata[:, :, 1] -= self.pos_offset[1]
        if self.scale_metadata:
            metadata[:, :, :2] *= D2KM
        if target is not None:
            target[:, :2] *= D2KM
        metadata[mask] = 0
        if self.scale_metadata:
            metadata /= 100
        if target is not None:
            target /= 100
        if transform_target_only:
            metadata = metadata_old
        if target is None:
            return metadata
        return metadata, target


class PreloadedEventGenerator(Dataset):
    """
    A custom PyTorch Dataset class designed to generate preloaded event data for training or evaluation.
    This class wraps an `EarthquakeDataset` and a `DataProcessor` to provide processed input-output pairs.
    Attributes:
        dataset (EarthquakeDataset): An instance of the EarthquakeDataset class, responsible for loading
                                      raw earthquake-related data.
        processor (DataProcessor): An instance of the DataProcessor class, responsible for processing
                                    the raw data into model-ready inputs and outputs.
    """

    def __init__(self, data_path, event_key, data, event_metadata, **kwargs):
        """
        Initializes the PreloadedEventGenerator.
        Args:
            data_path (str): The file path or directory where the dataset is stored.
            event_key (str): A key used to identify specific events within the dataset.
            data (dict or array-like): Raw data associated with the events.
            event_metadata (dict or DataFrame): Metadata describing the events in the dataset.
            **kwargs: Additional keyword arguments passed to both EarthquakeDataset and DataProcessor.
        """
        super(PreloadedEventGenerator, self).__init__()
        self.dataset = EarthquakeDataset(
            data_path=data_path,
            event_key=event_key,
            data=data,
            event_metadata=event_metadata,
            **kwargs,
        )
        self.processor = DataProcessor(**kwargs)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The length of the underlying EarthquakeDataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves and processes a single batch of data at the given index.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing two elements:
            inputs: Processed input data ready for model consumption.
            outputs: Corresponding target outputs for the model.
        """
        batch_data = self.dataset[index]
        inputs, outputs = self.processor.process_batch(batch_data)

        return inputs, outputs


def generator_from_config(
        config,
        data,
        data_path,
        event_key,
        event_metadata,
        time,
        sampling_rate=100,
        pga=False,
):
    """init generator"""
    generator_params = config["data"]
    cutout = int(sampling_rate * (generator_params["noise_seconds"] + time))
    cutout = (cutout, cutout + 1)

    n_pga_targets = config["model"].get("n_pga_targets", 0)
    if "data_path" in generator_params:
        del generator_params["data_path"]

    generator = PreloadedEventGenerator(
        data_path=data_path,
        event_key=event_key,
        data=data,
        event_metadata=event_metadata,
        coords_target=True,
        cutout=cutout,
        pga_targets=n_pga_targets,
        sampling_rate=sampling_rate,
        select_first=True,
        shuffle=False,
        pga_mode=pga,
        **generator_params,
    )
    return generator
