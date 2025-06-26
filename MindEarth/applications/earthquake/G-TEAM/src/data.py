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

import mindspore as ms
from mindspore.dataset import Dataset

# degrees to kilometers
D2KM = 111.19492664455874


def load_pickle_data(filename):
    """Load and deserialize data from a pickle file."""
    with open(filename, "rb") as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data

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


class DataGenerator(Dataset):
    """
    A PyTorch Dataset subclass for generating earthquake detection training data.
    Handles loading, preprocessing, and batching of seismic waveform data.
    """
    def __init__(self, data_path, event_metadata_index, event_key, mag_key='M_J', batch_size=32, cutout=None,
                 sliding_window=False, windowlen=3000, shuffle=True, label_smoothing=False, decimate=1):
        """
        Initialize the data generator.

        Args:
            data_path (str): Path to the HDF5 file containing seismic data.
            event_metadata_index (pd.DataFrame): DataFrame containing event metadata indices.
            event_key (str): Column name in metadata used to identify events.
            mag_key (str, optional): Column name containing magnitude values. Defaults to 'M_J'.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            cutout (tuple, optional): Time window for data augmentation. Defaults to None.
            sliding_window (bool, optional): Use sliding window for cutouts. Defaults to False.
            windowlen (int, optional): Length of time window for analysis. Defaults to 3000.
            shuffle (bool, optional): Shuffle data during epoch. Defaults to True.
            label_smoothing (bool, optional): Apply label smoothing. Defaults to False.
            decimate (int, optional): Decimation factor for waveform data. Defaults to 1.
        """
        super().__init__()
        self.data_path = data_path
        self.event_metadata_index = event_metadata_index
        self.event_key = event_key
        self.batch_size = batch_size
        self.mag_key = mag_key
        self.cutout = cutout
        self.sliding_window = sliding_window
        self.windowlen = windowlen
        self.shuffle = shuffle
        self.label_smoothing = label_smoothing
        self.decimate = decimate
        self.indexes = np.arange(len(self.event_metadata_index))
        self.on_epoch_end()

    def __len__(self):
        """
        Return the number of batches in the dataset.

        Returns:
            int: Number of batches = total samples / batch size (floor division)
        """
        return int(np.floor(len(self.event_metadata_index) / self.batch_size))

    def __getitem__(self, index):
        """
        Get a batch of data by index.

        Args:
            index (int): Batch index

        Returns:
            tuple: (x, y) where X is input tensor, y is target tensor
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_metadata = self.event_metadata_index.iloc[indexes]
        x, y = self.__data_generation(batch_metadata)
        return x, y

    def on_epoch_end(self):
        """
        Called when an epoch ends. Resets indexes and shuffles if required.
        """
        self.indexes = np.arange(len(self.event_metadata_index))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_metadata):
        """
        Generate data for a batch of events.

        Args:
            batch_metadata (pd.DataFrame): Metadata for batch of events

        Returns:
            tuple: (x, y) where x is input tensor, y is target tensor
        """
        x = []
        y = []
        with h5py.File(self.data_path, 'r') as f:
            for _, event in batch_metadata.iterrows():
                event_name = str(int(event[self.event_key]))
                if event_name not in f['data']:
                    continue
                g_event = f['data'][event_name]
                waveform = g_event['waveforms'][event['index'], ::self.decimate, :]
                if self.cutout:
                    if self.sliding_window:
                        windowlen = self.windowlen
                        window_end = np.random.randint(max(windowlen, self.cutout[0]),
                                                       min(waveform.shape[1], self.cutout[1]) + 1)
                        waveform = waveform[:, window_end - windowlen:window_end]
                    else:
                        waveform[:, np.random.randint(*self.cutout):] = 0
                x.append(waveform)
                y.append(event[self.mag_key])
        x = np.array(x)
        y = np.array(y)
        if self.label_smoothing:
            y += (y > 4) * np.random.randn(y.shape[0]).reshape(y.shape) * (y - 4) * 0.05

        return (ms.tensor(x, dtype=ms.float32),
                ms.tensor(np.expand_dims(np.expand_dims(y, axis=1), axis=2), dtype=ms.float32))

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

        super().__init__()

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

class PreloadedEventGenerator(Dataset):
    """
    A custom dataset generator for preloading seismic events.
    """
    def __init__(self, data_path, event_key, data, event_metadata, waveform_shape=(3000, 6), key='MA', batch_size=32,
                 cutout=None, sliding_window=False, windowlen=3000, shuffle=True, coords_target=True, oversample=1,
                 pos_offset=(-21, -69), label_smoothing=False, station_blinding=False, magnitude_resampling=3,
                 pga_targets=None, adjust_mean=True, transform_target_only=False, max_stations=None, trigger_based=None,
                 min_upsample_magnitude=2, disable_station_foreshadowing=False, selection_skew=None,
                 pga_from_inactive=False, integrate=False, differentiate=False, sampling_rate=100., select_first=False,
                 fake_borehole=False, scale_metadata=True, pga_key='pga', pga_mode=False, p_pick_limit=5000,
                 coord_keys=None, upsample_high_station_events=None, no_event_token=False, pga_selection_skew=None,
                 **kwargs):
        '''
        Initializes the PreloadedEventGenerator.

        Args:
            data_path: Path to the HDF5 file containing waveform data.
            event_key: The key in the event metadata DataFrame identifying each event.
            data: Dictionary containing 'coords' and 'pga' keys for metadata and PGA values.
            event_metadata: Pandas DataFrame with event metadata.
            waveform_shape: Shape of each waveform (number of samples, number of channels).
            key: The key in event metadata to use for magnitude.
            batch_size: Number of events per batch.
            cutout: Tuple specifying the range for random cutout in the waveform.
            sliding_window: Whether to use a sliding window for cutout.
            windowlen: Length of the sliding window.
            shuffle: Whether to shuffle the events at the end of each epoch.
            coords_target: Whether to include event coordinates as targets.
            oversample: Factor by which to oversample the events.
            pos_offset: Offset to apply to event coordinates.
            label_smoothing: Whether to apply label smoothing to magnitudes.
            station_blinding: Whether to randomly blind stations in the waveforms.
            magnitude_resampling: Factor by which to resample events based on their magnitude.
            pga_targets: Number of PGA targets to sample per event.
            adjust_mean: Whether to adjust the mean of the waveforms.
            transform_target_only: Whether to apply transformations only to the target coordinates.
            max_stations: Maximum number of stations to include per event.
            trigger_based: Whether to zero out waveforms before the P-wave trigger.
            min_upsample_magnitude: Minimum magnitude for upsampling.
            disable_station_foreshadowing: Whether to disable station foreshadowing.
            selection_skew: Skew parameter for selecting stations when max_stations is reached.
            pga_from_inactive: Whether to sample PGA from inactive stations.
            integrate: Whether to integrate the waveforms.
            differentiate: Whether to differentiate the waveforms.
            sampling_rate: Sampling rate of the waveforms.
            select_first: Whether to select the first stations when max_stations is reached.
            fake_borehole: Whether to add fake borehole channels to the waveforms.
            scale_metadata: Whether to scale the metadata coordinates.
            pga_key: Key in the data dictionary for PGA values.
            pga_mode: Whether to operate in PGA mode.
            p_pick_limit: Limit for P-wave picks.
            coord_keys: Keys in the event metadata for coordinates.
            upsample_high_station_events: Whether to upsample events with high station counts.
            no_event_token: Whether to include an event token in the outputs.
            pga_selection_skew: Skew parameter for selecting PGA targets.
            **kwargs:
        '''
        super().__init__()
        if kwargs:
            print(f'Unused parameters: {", ".join(kwargs.keys())}')
        self.data_path = data_path
        self.event_key = event_key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.waveform_shape = waveform_shape
        self.metadata = data['coords']
        self.event_metadata = event_metadata
        self.pga = data[pga_key]
        self.key = key
        self.cutout = cutout
        self.sliding_window = sliding_window
        self.windowlen = windowlen
        self.coords_target = coords_target
        self.oversample = oversample
        self.pos_offset = pos_offset
        self.label_smoothing = label_smoothing
        self.station_blinding = station_blinding
        self.magnitude_resampling = magnitude_resampling
        self.pga_targets = pga_targets
        self.adjust_mean = adjust_mean
        self.transform_target_only = transform_target_only
        self.max_stations = max_stations
        self.trigger_based = trigger_based
        self.disable_station_foreshadowing = disable_station_foreshadowing
        self.selection_skew = selection_skew
        self.pga_from_inactive = pga_from_inactive
        self.pga_selection_skew = pga_selection_skew
        self.integrate = integrate
        self.differentiate = differentiate
        self.sampling_rate = sampling_rate
        self.select_first = select_first
        self.fake_borehole = fake_borehole
        self.scale_metadata = scale_metadata
        self.upsample_high_station_events = upsample_high_station_events
        self.no_event_token = no_event_token
        self.triggers = data['p_picks']
        self.pga_mode = pga_mode
        self.p_pick_limit = p_pick_limit
        self.base_indexes = np.arange(self.event_metadata.shape[0])
        self.reverse_index = None
        if magnitude_resampling > 1:
            magnitude = self.event_metadata[key].values
            for i in np.arange(min_upsample_magnitude, 9):
                ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                self.base_indexes = np.concatenate(
                    (self.base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))
        if pga_mode:
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
        self.indexes = np.arange(len(self.event_metadata))
        if coord_keys is None:
            self.coord_keys = detect_location_keys(event_metadata.columns)
        else:
            self.coord_keys = coord_keys
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """
        Retrieves a batch of events from the dataset.
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        true_batch_size = len(indexes)
        if self.pga_mode:
            self.pga_indexes = [x[1] for x in indexes]
            indexes = [x[0] for x in indexes]

        waveforms = np.zeros([true_batch_size, self.max_stations] + self.waveform_shape)
        true_max_stations_in_batch = max(max([self.metadata[idx].shape[0] for idx in indexes]), self.max_stations)
        metadata = np.zeros((true_batch_size, true_max_stations_in_batch) + self.metadata[0].shape[1:])
        pga = np.zeros((true_batch_size, true_max_stations_in_batch))
        full_p_picks = np.zeros((true_batch_size, true_max_stations_in_batch))
        p_picks = np.zeros((true_batch_size, self.max_stations))
        reverse_selections = []

        waveforms, metadata, pga, p_picks, reverse_selections, full_p_picks = (
            self.htpyfile_process(indexes, waveforms, metadata, pga,
                                  p_picks, reverse_selections, full_p_picks))

        magnitude = self.event_metadata.iloc[indexes][self.key].values.copy()
        magnitude = magnitude.astype(np.float32)

        target, waveforms, magnitude, metadata, pga_values, pga_targets = (
            self.data_preprocessing(indexes, waveforms, p_picks, magnitude, metadata,
                                    pga, true_batch_size, reverse_selections, full_p_picks))

        waveforms, metadata = self.data_processing(waveforms, metadata)

        return self.get_result(waveforms, metadata, magnitude, target, pga_targets, pga_values)

    def htpyfile_process(self, indexes, waveforms, metadata, pga,
                         p_picks, reverse_selections, full_p_picks):
        """
        Processes the HDF5 file to retrieve waveform data for a batch of events.
        """
        with h5py.File(self.data_path, 'r') as f:
            for i, idx in enumerate(indexes):
                event = self.event_metadata.iloc[idx]
                event_name = str(event[self.event_key])
                if event_name not in f['data']:
                    continue
                g_event = f['data'][event_name]
                waveform_data = g_event['waveforms'][:, :, :]

                num_stations = waveform_data.shape[0]

                if num_stations <= self.max_stations:
                    waveforms[i, :num_stations] = waveform_data
                    metadata[i, :len(self.metadata[idx])] = self.metadata[idx]
                    pga[i, :len(self.pga[idx])] = self.pga[idx]
                    p_picks[i, :len(self.triggers[idx])] = self.triggers[idx]
                    reverse_selections += [[]]
                else:
                    if self.selection_skew is None:
                        selection = np.arange(0, num_stations)
                        np.random.shuffle(selection)
                    else:
                        tmp_p_picks = self.triggers[idx].copy()
                        mask = np.logical_and(tmp_p_picks <= 0, tmp_p_picks > self.p_pick_limit)
                        tmp_p_picks[mask] = min(np.max(tmp_p_picks), self.p_pick_limit)
                        coeffs = np.exp(-tmp_p_picks / self.selection_skew)
                        coeffs *= np.random.random(coeffs.shape)
                        coeffs[self.triggers[idx] == 0] = 0
                        coeffs[self.triggers[idx] > self.waveform_shape[0]] = 0
                        selection = np.argsort(-coeffs)

                    if self.select_first:
                        selection = np.argsort(self.triggers[idx])

                    metadata[i, :len(selection)] = self.metadata[idx][selection]
                    pga[i, :len(selection)] = self.pga[idx][selection]
                    full_p_picks[i, :len(selection)] = self.triggers[idx][selection]

                    tmp_reverse_selection = [0 for _ in selection]
                    for j, s in enumerate(selection):
                        tmp_reverse_selection[s] = j
                    reverse_selections += [tmp_reverse_selection]

                    selection = selection[:self.max_stations]
                    waveforms[i] = waveform_data[selection]
                    p_picks[i] = self.triggers[idx][selection]
        return waveforms, metadata, pga, p_picks, reverse_selections, full_p_picks

    def pga_mode_process(self, waveforms, reverse_selections, metadata,
                         pga_values, pga_targets, pga, indexes, full_p_picks):
        """
        Processes the data in PGA mode.
        """
        if self.pga_mode:
            for i in range(waveforms.shape[0]):
                pga_index = self.pga_indexes[i]
                if reverse_selections[i]:
                    sorted_pga = pga[i, reverse_selections[i]]
                    sorted_metadata = metadata[i, reverse_selections[i]]
                else:
                    sorted_pga = pga[i]
                    sorted_metadata = metadata[i]
                pga_values_pre = sorted_pga[pga_index * self.pga_targets:(pga_index + 1) * self.pga_targets]
                pga_values[i, :len(pga_values_pre)] = pga_values_pre
                pga_targets_pre = sorted_metadata[pga_index * self.pga_targets:(pga_index + 1) * self.pga_targets, :]
                if pga_targets_pre.shape[-1] == 4:
                    pga_targets_pre = pga_targets_pre[:, (0, 1, 3)]
                pga_targets[i, :len(pga_targets_pre), :] = pga_targets_pre
        else:
            pga[np.logical_or(np.isnan(pga), np.isinf(pga))] = 0
            for i in range(waveforms.shape[0]):
                active = np.where(pga[i] != 0)[0]
                l = len(active)
                if l == 0:
                    raise ValueError(f'Found event without PGA idx={indexes[i]}')
                while len(active) < self.pga_targets:
                    active = np.repeat(active, 2)
                if self.pga_selection_skew is not None:
                    active_p_picks = full_p_picks[i, active]
                    mask = np.logical_and(active_p_picks <= 0, active_p_picks > self.p_pick_limit)
                    active_p_picks[mask] = min(np.max(active_p_picks), self.p_pick_limit)
                    coeffs = np.exp(-active_p_picks / self.pga_selection_skew)
                    coeffs *= np.random.random(coeffs.shape)
                    active = active[np.argsort(-coeffs)]
                else:
                    np.random.shuffle(active)

                samples = active[:self.pga_targets]
                if metadata.shape[-1] == 3:
                    pga_targets[i] = metadata[i, samples, :]
                else:
                    full_targets = metadata[i, samples]
                    pga_targets[i] = full_targets[:, (0, 1, 3)]
                pga_values[i] = pga[i, samples]
        return pga_values, pga_targets

    def data_preprocessing(self, indexes, waveforms, p_picks, magnitude, metadata,
                           pga, true_batch_size, reverse_selections, full_p_picks):
        """
        Data preprocessing.
        """
        target = None
        if self.coords_target:
            target = self.event_metadata.iloc[indexes][self.coord_keys].values
            target = target.astype(np.float32)
        org_waveform_length = waveforms.shape[2]
        if self.cutout:
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = np.random.randint(max(windowlen, self.cutout[0]),
                                               min(waveforms.shape[2], self.cutout[1]) + 1)
                waveforms = waveforms[:, :, window_end - windowlen: window_end]
                cutout = window_end
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms, axis=2, keepdims=True)
            else:
                cutout = np.random.randint(*self.cutout)
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms[:, :, :cutout + 1], axis=2, keepdims=True)
                waveforms[:, :, cutout:] = 0
        else:
            cutout = waveforms.shape[2]

        if self.trigger_based:
            p_picks[p_picks <= 0] = org_waveform_length
            waveforms[cutout < p_picks, :, :] = 0
        if self.integrate:
            waveforms = np.cumsum(waveforms, axis=2) / self.sampling_rate
        if self.differentiate:
            waveforms = np.diff(waveforms, axis=2)

        magnitude = np.expand_dims(np.expand_dims(magnitude, axis=-1), axis=-1)
        if self.coords_target:
            metadata, target = self.location_transformation(metadata, target)
        else:
            metadata = self.location_transformation(metadata)

        if self.label_smoothing:
            magnitude += (magnitude > 4) * np.random.randn(magnitude.shape[0]).reshape(magnitude.shape) * (
                magnitude - 4) * 0.05
        if not self.pga_from_inactive and not self.pga_mode:
            metadata = metadata[:, :self.max_stations]
            pga = pga[:, :self.max_stations]
        pga_values = ()
        pga_targets = ()
        if self.pga_targets:
            pga_values = np.zeros((true_batch_size, self.pga_targets))
            pga_targets = np.zeros((true_batch_size, self.pga_targets, 3))

            pga_values, pga_targets = self.pga_mode_process(waveforms, reverse_selections, metadata,
                                                            pga_values, pga_targets, pga, indexes, full_p_picks)

            pga_values = pga_values.reshape((true_batch_size, self.pga_targets, 1, 1))

        return target, waveforms, magnitude, metadata, pga_values, pga_targets

    def data_processing(self, waveforms, metadata):
        """
        Data process.
        """
        metadata = metadata[:, :self.max_stations]
        if self.station_blinding:
            mask = np.zeros(waveforms.shape[:2], dtype=bool)

            for i in range(waveforms.shape[0]):
                active = np.where((waveforms[i] != 0).any(axis=(1, 2)))[0]
                l = len(active)
                if l == 0:
                    active = np.zeros(1, dtype=int)
                blind_length = np.random.randint(0, len(active))
                np.random.shuffle(active)
                blind = active[:blind_length]
                mask[i, blind] = True

            waveforms[mask] = 0
            metadata[mask] = 0

        stations_without_trigger = (metadata != 0).any(axis=2) & (waveforms == 0).all(axis=(2, 3))
        if self.disable_station_foreshadowing:
            metadata[stations_without_trigger] = 0
        else:
            waveforms[stations_without_trigger, 0, 0] += 1e-9

        mask = np.logical_and((metadata == 0).all(axis=(1, 2)), (waveforms == 0).all(axis=(1, 2, 3)))
        waveforms[mask, 0, 0, 0] = 1e-9
        metadata[mask, 0, 0] = 1e-9

        return waveforms, metadata

    def get_result(self, waveforms, metadata, magnitude, target, pga_targets, pga_values):
        """
        get result.
        """
        inputs = [ms.tensor(waveforms, dtype=ms.float32), ms.tensor(metadata, dtype=ms.float32)]
        outputs = []
        if not self.no_event_token:
            outputs += [ms.tensor(magnitude, dtype=ms.float32)]

            if self.coords_target:
                target = np.expand_dims(target, axis=-1)
                outputs += [ms.tensor(target, dtype=ms.float32)]

        if self.pga_targets:
            inputs += [ms.tensor(pga_targets, dtype=ms.float32)]
            outputs += [ms.tensor(pga_values, dtype=ms.float32)]

        return inputs, outputs

    def on_epoch_end(self):
        """
        Resets the indexes for a new epoch, optionally with oversampling and shuffling.
        """
        self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def location_transformation(self, metadata, target=None):
        """
        Transforms the event coordinates and optionally the target coordinates.
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

        # Coordinates to kilometers (assuming a flat earth, which is okay close to equator)
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
