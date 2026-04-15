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
"generate dataset"
import os
import datetime
from typing import Union, Sequence, Tuple
import h5py
import pandas as pd
from einops import rearrange
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import nn, ops, Tensor
from mindspore.dataset.vision import RandomRotation, Rotate
from mindspore.dataset.transforms import Compose


SEVIR_DATA_TYPES = ["vis", "ir069", "ir107", "vil", "lght"]
LIGHTING_FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {
    "lght": (48, 48),
}
PREPROCESS_SCALE_01 = {
    "vis": 1,
    "ir069": 1,
    "ir107": 1,
    "vil": 1 / 255,
    "lght": 1,
}
PREPROCESS_OFFSET_01 = {
    "vis": 0,
    "ir069": 0,
    "ir107": 0,
    "vil": 0,
    "lght": 0,
}


def path_splitall(path):
    """
    Split a file path into all its components.

    Recursively splits the path into directory components and the final file name,
    handling both absolute and relative paths across different OS conventions.

    Args:
        path (str): Input file path to split

    Returns:
        List[str]: List of path components from root to leaf
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def change_layout(data, in_layout="NHWT", out_layout="NHWT"):
    """
    Convert data layout between different dimension orderings.

    Handles layout transformations using einops.rearrange, with special handling
    for 'C' (channel) dimensions which are treated as singleton dimensions.

    Args:
        data (Tensor/ndarray): Input data to transform
        in_layout (str): Current dimension order (e.g., "NHWT")
        out_layout (str): Target dimension order (e.g., "THWC")

    Returns:
        ndarray: Data in new layout with applied transformations
    """
    if isinstance(data, ms.Tensor):
        data = data.asnumpy()
    in_layout = " ".join(in_layout.replace("C", "1"))
    out_layout = " ".join(out_layout.replace("C", "1"))
    data = rearrange(data, f"{in_layout} -> {out_layout}")
    return data


class DatasetSEVIR:
    """
    SEVIR Dataset class for weather event sequence data.

    Provides data loading and augmentation capabilities for SEVIR (Severe Weather Events Dataset)
    with support for different temporal layouts and data preprocessing.

    Attributes:
        layout (str): Output data layout configuration
        sevir_dataloader (SEVIRDataLoader): Core data loading component
        aug_pipeline (AugmentationPipeline): Data augmentation operations
    """
    def __init__(
            self,
            seq_in: int = 25,
            raw_seq_in: int = 49,
            sample_mode: str = "sequent",
            stride: int = 12,
            layout: str = "THWC",
            ori_layout: str = "NHWT",
            split_mode: str = "uneven",
            sevir_catalog: Union[str, pd.DataFrame] = None,
            sevir_data_dir: str = None,
            start_date: datetime.datetime = None,
            end_date: datetime.datetime = None,
            datetime_filter=None,
            catalog_filter="default",
            shuffle: bool = False,
            shuffle_seed: int = 1,
            output_type=np.float32,
            preprocess: bool = True,
            rescale_method: str = "01",
            verbose: bool = False,
            aug_mode: str = "0",
    ):
        super().__init__()
        self.layout = layout.replace("C", "1")
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=[
                "vil",
            ],
            seq_in=seq_in,
            raw_seq_in=raw_seq_in,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=1,
            layout=ori_layout,
            num_shard=1,
            rank=0,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            verbose=verbose,
        )
        self.aug_mode = aug_mode
        self.aug_pipeline = AugmentationPipeline(
            self.aug_mode,
            self.layout,
        )

    def __getitem__(self, index):
        """
        Get processed data sample by index.

        Performs data extraction, augmentation, and layout conversion.

        Args:
            index (int): Sample index

        Returns:
            ndarray: Processed data in specified layout
        """
        data_dict = self.sevir_dataloader.extract_data(index=index)
        data = data_dict["vil"]
        if self.aug_pipeline is not None:
            data = self.aug_pipeline(data_dict)
        return data

    def __len__(self):
        """len"""
        return self.sevir_dataloader.__len__()


class SEVIRDataModule(nn.Cell):
    """
    DataModule for SEVIR dataset.

    Manages dataset splits (train/val/test), data loading, and augmentation
    for training diffusion models on weather event sequences.

    Attributes:
        sevir_dir (str): Root directory of SEVIR dataset
        batch_size (int): Data loader batch size
        num_workers (int): Number of data loader workers
        aug_mode (str): Data augmentation configuration
        layout (str): Data layout configuration
    """

    def __init__(
            self,
            seq_in: int = 25,
            sample_mode: str = "sequent",
            stride: int = 12,
            layout: str = "NTHWC",
            output_type=np.float32,
            preprocess: bool = True,
            rescale_method: str = "01",
            verbose: bool = False,
            aug_mode: str = "0",
            dataset_name: str = "sevir",
            sevir_dir: str = None,
            start_date: Tuple[int] = None,
            train_val_split_date: Tuple[int] = (2019, 3, 20),
            train_test_split_date: Tuple[int] = (2019, 6, 1),
            end_date: Tuple[int] = None,
            val_ratio: float = 0.1,
            batch_size: int = 1,
            num_workers: int = 1,
            raw_seq_len: int = 25,
            seed: int = 0,
    ):
        super().__init__()
        self.sevir_dir = sevir_dir
        self.aug_mode = aug_mode
        self.seq_in = seq_in
        self.sample_mode = sample_mode
        self.stride = stride
        self.layout = layout.replace("N", "")
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.aug_mode = aug_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset_name = dataset_name
        self.sevir_dir = sevir_dir
        self.catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
        self.raw_data_dir = os.path.join(sevir_dir, "data")
        self.raw_seq_in = raw_seq_len
        self.start_date = (
            datetime.datetime(*start_date) if start_date is not None else None
        )
        self.train_test_split_date = (
            datetime.datetime(*train_test_split_date)
            if train_test_split_date is not None
            else None
        )
        self.train_val_split_date = (
            datetime.datetime(*train_val_split_date)
            if train_val_split_date is not None
            else None
        )
        self.end_date = datetime.datetime(*end_date) if end_date is not None else None
        self.val_ratio = val_ratio

    def setup(self, stage=None) -> None:
        """
        Prepare dataset splits for different stages.

        Creates train/val/test splits based on date ranges and configuration.

        Args:
            stage (str): Current stage ("fit", "test", etc.)
        """
        if stage in (None, "fit"):
            print("train")
            self.sevir_train_ori = DatasetSEVIR(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_in=self.raw_seq_in,
                split_mode="uneven",
                shuffle=False,
                seq_in=self.seq_in,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.start_date,
                end_date=self.train_val_split_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode=self.aug_mode,
            )
            self.sevir_train = ds.GeneratorDataset(
                source=self.sevir_train_ori,
                column_names="vil",
                shuffle=False,
                num_parallel_workers=self.num_workers,
            )
            self.sevir_train = self.sevir_train.batch(batch_size=self.batch_size)

        if stage in (None, "fit"):
            print("val")
            self.sevir_val = DatasetSEVIR(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_in=self.raw_seq_in,
                split_mode="uneven",
                shuffle=False,
                seq_in=self.seq_in,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.train_val_split_date,
                end_date=self.train_test_split_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode=self.aug_mode,
            )
            self.sevir_val = ds.GeneratorDataset(
                source=self.sevir_val,
                column_names="vil",
                shuffle=False,
                num_parallel_workers=self.num_workers,
            )
            self.sevir_val = self.sevir_val.batch(batch_size=self.batch_size)

        if stage in (None, "test"):
            print("test")
            self.sevir_test = DatasetSEVIR(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_in=self.raw_seq_in,
                split_mode="uneven",
                shuffle=False,
                seq_in=self.seq_in,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.train_test_split_date,
                end_date=self.end_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode=self.aug_mode,
            )
            self.sevir_test = ds.GeneratorDataset(
                source=self.sevir_test,
                column_names="vil",
                shuffle=False,
                num_parallel_workers=self.num_workers,
            )
            self.sevir_test = self.sevir_test.batch(batch_size=self.batch_size)

    @property
    def num_train_samples(self):
        """Get number of training samples"""
        return len(self.sevir_train_ori)

    @property
    def num_val_samples(self):
        """Get number of validation samples"""
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        """Get number of test samples"""
        return len(self.sevir_test)


class SEVIRDataLoader:
    r"""
    DataLoader that loads SEVIR sequences, and spilts each event
    into segments according to specified sequence length.
    """

    def __init__(
            self,
            data_types: Sequence[str] = None,
            seq_in: int = 49,
            raw_seq_in: int = 49,
            sample_mode: str = "sequent",
            stride: int = 12,
            batch_size: int = 1,
            layout: str = "NHWT",
            num_shard: int = 1,
            rank: int = 0,
            split_mode: str = "uneven",
            sevir_catalog: Union[str, pd.DataFrame] = None,
            sevir_data_dir: str = None,
            start_date: datetime.datetime = None,
            end_date: datetime.datetime = None,
            datetime_filter=None,
            catalog_filter="default",
            shuffle: bool = False,
            shuffle_seed: int = 1,
            output_type=np.float32,
            preprocess: bool = True,
            rescale_method: str = "01",
            verbose: bool = False,
    ):
        super().__init__()

        # configs which should not be modified
        self.lght_frame_times = LIGHTING_FRAME_TIMES
        self.data_shape = SEVIR_DATA_SHAPE

        self.raw_seq_in = raw_seq_in
        assert (
            seq_in <= self.raw_seq_in
        ), f"seq_in must not be larger than raw_seq_in = {raw_seq_in}, got {seq_in}."
        self.seq_in = seq_in
        assert sample_mode in [
            "random",
            "sequent",
        ], f"Invalid sample_mode = {sample_mode}, must be 'random' or 'sequent'."
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        valid_layout = ("NHWT", "NTHW", "NTCHW", "NTHWC", "TNHW", "TNCHW")
        if layout not in valid_layout:
            raise ValueError(
                f"Invalid layout = {layout}! Must be one of {valid_layout}."
            )
        self.layout = layout
        self.num_shard = num_shard
        self.rank = rank
        valid_split_mode = ("ceil", "floor", "uneven")
        if split_mode not in valid_split_mode:
            raise ValueError(
                f"Invalid split_mode: {split_mode}! Must be one of {valid_split_mode}."
            )
        self.split_mode = split_mode
        self._samples = None
        self._hdf_files = {}
        self.data_types = data_types
        if isinstance(sevir_catalog, str):
            self.catalog = pd.read_csv(
                sevir_catalog, parse_dates=["time_utc"], low_memory=False
            )
        else:
            self.catalog = sevir_catalog
        self.sevir_data_dir = sevir_data_dir
        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        if self.catalog_filter is not None:
            if self.catalog_filter == "default":
                self.catalog_filter = lambda c: c.pct_missing == 0
            self.catalog = self.catalog[self.catalog_filter(self.catalog)]

        self._compute_samples()
        print(self._samples.head(n=10))
        print("len", len(self._samples))
        self._open_files(verbose=self.verbose)
        self.reset()

    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets self._samples
        """
        imgt = self.data_types
        imgts = set(imgt)
        filtcat = self.catalog[
            np.logical_or.reduce([self.catalog.img_type == i for i in imgt])
        ]
        filtcat = filtcat.groupby("id").filter(
            lambda x: imgts.issubset(set(x["img_type"]))
        )
        filtcat = filtcat.groupby("id").filter(lambda x: x.shape[0] == len(imgt))
        self._samples = filtcat.groupby("id").apply(
            lambda df: self._df_to_series(df, imgt)
        )
        if self.shuffle:
            self.shuffle_samples()

    def shuffle_samples(self):
        """Shuffle the dataset samples using a fixed random seed for reproducibility."""
        self._samples = self._samples.sample(frac=1, random_state=self.shuffle_seed)

    def _df_to_series(self, df, imgt):
        """Convert catalog DataFrame entries to structured format for multi-image types."""
        d = {}
        df = df.set_index("img_type")
        for i in imgt:
            s = df.loc[i]
            idx = s.file_index if i != "lght" else s.id
            d.update({f"{i}_filename": [s.file_name], f"{i}_index": [idx]})

        return pd.DataFrame(d)

    def _open_files(self, verbose=True):
        """
        Opens HDF files
        """
        imgt = self.data_types
        hdf_filenames = []
        for t in imgt:
            hdf_filenames += list(np.unique(self._samples[f"{t}_filename"].values))

        print("hdf_filenames", hdf_filenames)
        self._hdf_files = {}
        for f in hdf_filenames:
            print("Opening HDF5 file for reading", f)
            if verbose:
                print("Opening HDF5 file for reading", f)
            self._hdf_files[f] = h5py.File(self.sevir_data_dir + "/" + f, "r")
            print("f:", f)
            print("self._hdf_files[f]:", self._hdf_files[f])

    def close(self):
        """
        Closes all open file handles
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
            print("close: ", f)
        self._hdf_files = {}

    @property
    def num_seq_per_event(self):
        """num seq per event"""
        return 1 + (self.raw_seq_in - self.seq_in) // self.stride

    @property
    def total_num_seq(self):
        """
        The total number of sequences within each shard.
        Notice that it is not the product of `self.num_seq_per_event` and `self.total_num_event`.
        """
        return int(self.num_seq_per_event * self.num_event)

    @property
    def total_num_event(self):
        """
        The total number of events in the whole dataset, before split into different shards.
        """
        return int(self._samples.shape[0])

    @property
    def start_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx >= start_event_idx
        """
        return self.total_num_event // self.num_shard * self.rank

    @property
    def end_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx < end_event_idx

        """
        if self.split_mode == "ceil":
            last_start_event_idx = (
                self.total_num_event // self.num_shard * (self.num_shard - 1)
            )
            num_event = self.total_num_event - last_start_event_idx
            return self.start_event_idx + num_event
        if self.split_mode == "floor":
            return self.total_num_event // self.num_shard * (self.rank + 1)
        if self.rank == self.num_shard - 1:
            return self.total_num_event
        return self.total_num_event // self.num_shard * (self.rank + 1)

    @property
    def num_event(self):
        """
        The number of events split into each rank
        """
        return self.end_event_idx - self.start_event_idx

    def _read_data(self, row, data):
        """
        Iteratively read data into data dict. Finally data[imgt] gets shape (batch_size, height, width, raw_seq_in).

        Parameters
        ----------
        row
            A series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index.
        data
            Dict, data[imgt] is a data tensor with shape = (tmp_batch_size, height, width, raw_seq_in).

        Returns
        -------
        data
            Updated data. Updated shape = (tmp_batch_size + 1, height, width, raw_seq_in).
        """
        imgtyps = np.unique([x.split("_")[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f"{t}_filename"]
            idx = row[f"{t}_index"]
            t_slice = slice(0, None)
            if t == "lght":
                lght_data = self._hdf_files[fname][idx][:]
                data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[fname][t][idx : idx + 1, :, :, t_slice]
            data[t] = (
                np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i
            )

        return data

    def _lght_to_grid(self, data, t_slice=slice(0, None)):
        """
        Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        """

        out_size = (
            (*self.data_shape["lght"], len(self.lght_frame_times))
            if t_slice.stop is None
            else (*self.data_shape["lght"], 1)
        )
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        x, y = data[:, 3], data[:, 4]
        m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
        data = data[m, :]
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)
        t = data[:, 0]
        if t_slice.stop is not None:
            if t_slice.stop > 0:
                if t_slice.stop < len(self.lght_frame_times):
                    tm = np.logical_and(
                        t >= self.lght_frame_times[t_slice.stop - 1],
                        t < self.lght_frame_times[t_slice.stop],
                    )
                else:
                    tm = t >= self.lght_frame_times[-1]
            else:
                tm = np.logical_and(
                    t >= self.lght_frame_times[0], t < self.lght_frame_times[1]
                )

            data = data[tm, :]
            z = np.zeros(data.shape[0], dtype=np.int64)
        else:
            z = np.digitize(t, self.lght_frame_times) - 1
            z[z == -1] = 0

        x = data[:, 3].astype(np.int64)
        y = data[:, 4].astype(np.int64)

        k = np.ravel_multi_index(np.array([y, x, z]), out_size)
        n = np.bincount(k, minlength=np.prod(out_size))
        return np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]

    @property
    def sample_count(self):
        """
        Record how many times self.__next__() is called.
        """
        return self._sample_count

    @property
    def _curr_event_idx(self):
        return self.__curr_event_idx

    @property
    def _curr_seq_idx(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self.__curr_seq_idx

    def _set__curr_event_idx(self, val):
        self.__curr_event_idx = val

    def _set__curr_seq_idx(self, val):
        """
        Used only when self.sample_mode == 'sequent'
        """
        self.__curr_seq_idx = val

    def reset(self, shuffle: bool = None):
        """reset"""
        self._set__curr_event_idx(val=self.start_event_idx)
        self._set__curr_seq_idx(0)
        self._sample_count = 0
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            self.shuffle_samples()

    def __len__(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self.total_num_seq // self.batch_size

    def _load_event_batch(self, event_idx, event_batch_size):
        """
        Loads a selected batch of events (not batch of sequences) into memory.

        Parameters
        ----------
        idx
        event_batch_size
            event_batch[i] = all_type_i_available_events[idx:idx + event_batch_size]
        Returns
        -------
        event_batch
            list of event batches.
            event_batch[i] is the event batch of the i-th data type.
            Each event_batch[i] is a np.ndarray with shape = (event_batch_size, height, width, raw_seq_in)
        """
        event_idx_slice_end = event_idx + event_batch_size
        pad_size = 0
        if event_idx_slice_end > self.end_event_idx:
            pad_size = event_idx_slice_end - self.end_event_idx
            event_idx_slice_end = self.end_event_idx
        pd_batch = self._samples.iloc[event_idx:event_idx_slice_end]
        data = {}
        for _, row in pd_batch.iterrows():
            data = self._read_data(row, data)
        if pad_size > 0:
            event_batch = []
            for t in self.data_types:
                pad_shape = [
                    pad_size,
                ] + list(data[t].shape[1:])
                data_pad = np.concatenate(
                    (
                        data[t].astype(self.output_type),
                        np.zeros(pad_shape, dtype=self.output_type),
                    ),
                    axis=0,
                )
                event_batch.append(data_pad)
        else:
            event_batch = [data[t].astype(self.output_type) for t in self.data_types]
        return event_batch


    def extract_data(self, index):
        """
        Extracts a batch of data without any processing.

        Parameters
        ----------
        index
            The index of the batch to sample.

        Returns
        -------
        event_batch
            The extracted data from the event batch without any processing.
        """
        event_idx = (index * self.batch_size) // self.num_seq_per_event
        seq_idx = (index * self.batch_size) % self.num_seq_per_event
        num_sampled = 0
        sampled_idx_list = []
        while num_sampled < self.batch_size:
            sampled_idx_list.append({"event_idx": event_idx, "seq_idx": seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]["event_idx"]
        event_batch_size = sampled_idx_list[-1]["event_idx"] - start_event_idx + 1

        event_batch = self._load_event_batch(
            event_idx=start_event_idx, event_batch_size=event_batch_size
        )
        ret_dict = {}
        for sampled_idx in sampled_idx_list:
            batch_slice = [
                sampled_idx["event_idx"] - start_event_idx,
            ]
            seq_slice = slice(
                sampled_idx["seq_idx"] * self.stride,
                sampled_idx["seq_idx"] * self.stride + self.seq_in,
            )
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate(
                        (ret_dict[imgt], sampled_seq), axis=0
                    )
                else:
                    ret_dict.update({imgt: sampled_seq})

        return ret_dict


class AugmentationPipeline:
    """Data augmentation pipeline for multi-frame image processing.
    """
    def __init__(
            self,
            aug_mode="0",
            layout=None,
    ):
        self.layout = layout
        self.aug_mode = aug_mode

        if aug_mode == "0":
            self.aug = lambda x: x
        elif self.aug_mode == "1":
            self.aug = Compose(
                [
                    vision.RandomHorizontalFlip(),
                    vision.RandomVerticalFlip(),
                    RandomRotation(degrees=180),
                ]
            )
        elif aug_mode == "2":
            self.aug = Compose(
                [
                    vision.RandomHorizontalFlip(),
                    vision.RandomVerticalFlip(),
                    FixedAngleRotation(angles=[0, 90, 180, 270]),
                ]
            )
        else:
            raise NotImplementedError

    def rearrange_tensor(self, tensor, from_layout, to_layout):
        """Permute and reshape tensor dimensions according to layout specifications."""
        return tensor.permute(*tuple(range(len(from_layout)))).reshape(to_layout)

    def __call__(self, data_dict):
        """Apply augmentation pipeline to input data dictionary.

        Args:
            data_dict (dict): Input data containing "vil" key with tensor data

        Returns:
            ms.Tensor: Processed tensor with applied augmentations and layout conversion
        """
        data = data_dict["vil"].squeeze(0)
        if self.aug_mode != "0":
            data = rearrange(
                data,
                "H W T -> T H W",
            )
            data = self.aug(data)
            data = rearrange(data, f"{' '.join('THW')} -> {' '.join(self.layout)}")
        else:
            data = rearrange(
                data,
                f"{' '.join('HWT')} -> {' '.join(self.layout)}",
            )

        return data


class FixedAngleRotation:
    """Image augmentation for rotating images by fixed predefined angles.

    Args:
        angles (List[int]): List of allowed rotation angles (degrees)
    """
    def __init__(self, angles=None):
        self.angles = angles

    def __call__(self, img):
        """Apply random rotation from predefined angles.

        Args:
            img (PIL.Image or mindspore.Tensor): Input image to transform

        Returns:
            PIL.Image or mindspore.Tensor: Rotated image with same format as input
        """
        angle = np.random.choice(self.angles)
        return Rotate(angle)(img)


class SEVIRDataset:
    """Base dataset class for processing SEVIR data with configurable preprocessing.

    Args:
        data_types (Sequence[str], optional):
            List of data types to process (e.g., ["vil", "lght"]). Defaults to SEVIR_DATA_TYPES.
        layout (str, optional):
            Tensor layout specification containing dimensions:
                N - batch size
                H - height
                W - width
                T - time/sequence length
                C - channel
            Defaults to "NHWT".
        rescale_method (str, optional):
            Data rescaling strategy identifier (e.g., "01" for 0-1 normalization). Defaults to "01".
    """
    def __init__(
            self,
            data_types: Sequence[str] = None,
            layout: str = "NHWT",
            rescale_method: str = "01",
    ):
        super().__init__()
        if data_types is None:
            data_types = SEVIR_DATA_TYPES
        else:
            assert set(data_types).issubset(SEVIR_DATA_TYPES)

        self.layout = layout
        self.data_types = data_types
        self.rescale_method = rescale_method

    @staticmethod
    def preprocess_data_dict(data_dict, data_types=None, layout="NHWT"):
        """
        Parameterss
        ----------
        data_dict:  Dict[str, Union[np.ndarray, ms.Tensor]]
        data_types: Sequence[str]
            The data types that we want to rescale. This mainly excludes "mask" from preprocessing.
        layout: str
            consists of batch_size 'N', seq_in 'T', channel 'C', height 'H', width 'W'
        Returns
        -------
        data_dict:  Dict[str, Union[np.ndarray, ms.Tensor]]
            preprocessed data
        """
        scale_dict = PREPROCESS_SCALE_01
        offset_dict = PREPROCESS_OFFSET_01
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
                elif isinstance(data, ms.Tensor):
                    data = data.float()
                else:
                    raise TypeError
                data = change_layout(
                    data=scale_dict[key] * (data + offset_dict[key]),
                    in_layout="NHWT",
                    out_layout=layout,
                )
                data_dict[key] = data
        return data_dict

    @staticmethod
    def data_dict_to_tensor(data_dict, data_types=None):
        """
        Convert each element in data_dict to ms.Tensor (copy without grad).
        """
        ret_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, ms.Tensor):
                    ret_dict[key] = data
                elif isinstance(data, np.ndarray):
                    ret_dict[key] = Tensor.from_numpy(data)
                else:
                    raise ValueError(
                        f"Invalid data type: {type(data)}. Should be ms.Tensor or np.ndarray"
                    )
            else:
                ret_dict[key] = data
        return ret_dict

    def process_data(self, data_dict):
        """
        Processes the extracted data.

        Parameters
        ----------
        data_dict
            The dictionary containing the extracted data.

        Returns
        -------
        processed_dict
            The dictionary containing the processed data.
        """
        split_tensors = data_dict.split(1, axis=0)
        processed_tensors = [
            self.process_singledata(tensor) for tensor in split_tensors
        ]
        tensor_list = []
        for item in processed_tensors:
            numpy_array = item["vil"]
            tensor = Tensor(numpy_array)
            tensor_list.append(tensor)
        output_tensor = ops.Stack(axis=0)(tensor_list)
        return output_tensor

    def process_singledata(self, singledata):
        """process singledata"""
        squeezed_tensor = ops.squeeze(singledata, 0)
        singledata = {"vil": squeezed_tensor}
        processed_dict = self.data_dict_to_tensor(
            data_dict=singledata, data_types=self.data_types
        )
        processed_dict = self.preprocess_data_dict(
            data_dict=processed_dict,
            data_types=self.data_types,
            layout=self.layout,
        )
        return processed_dict
