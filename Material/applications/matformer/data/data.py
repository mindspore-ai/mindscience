# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Implementation based on the template of ALIGNN."""
import os
import stat
import math
import random
import json
import logging
import pickle as pk
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata
from pathos.multiprocessing import ProcessingPool
from data.graphs import StructureGraph, StructureDataset

# pylint: disable=E1130
# pylint: disable=W0102

tqdm.pandas()


def load_dataset(
        name: str = "dft_3d",
        target=None,
        limit: Optional[int] = None,
        classification_threshold: Optional[float] = None,
):
    """Load jarvis data."""
    d = jdata(name)
    data = []
    for i in d:
        if i[target] != "na" and not math.isnan(i[target]):
            if classification_threshold is not None:
                if i[target] <= classification_threshold:
                    i[target] = 0
                elif i[target] > classification_threshold:
                    i[target] = 1
                else:
                    raise ValueError(
                        "Check classification data type.",
                        i[target],
                        type(i[target]),
                    )
            data.append(i)
    d = data
    if limit is not None:
        d = d[:limit]
    d = pd.DataFrame(d)
    return d


def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def load_graphs(
        df: pd.DataFrame,
        cutoff: float = 8,
        max_neighbors: int = 12,
        use_canonize: bool = False,
        use_lattice: bool = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        res = StructureGraph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
        )
        return res

    def parallel_apply(df, func, worker=None):
        if not worker:
            with ProcessingPool() as pool:
                result = list(tqdm(pool.imap(func, df), total=len(df)))
            return result
        with ProcessingPool(worker) as pool:
            result = list(tqdm(pool.imap(func, df), total=len(df)))
        return result

    graphs = parallel_apply(df['atoms'], atoms_to_graph, 10)
    return graphs


def get_id_train_val_test(
        total_size=1000,
        split_seed=123,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        n_train=None,
        n_test=None,
        n_val=None,
        keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
            train_ratio is None
            and val_ratio is not None
            and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            logging.info("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test): -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_dataset(
        dataset=[],
        id_tag="jid",
        target="",
        atom_features="",
        use_canonize="",
        line_graph="",
        cutoff=8.0,
        max_neighbors=12,
        classification=False,
        use_lattice=False,
        mean_train=None,
        std_train=None,
):
    """Get Dataset."""
    df = pd.DataFrame(dataset)
    vals = df[target].values
    if target in ('shear modulus', 'bulk modulus'):
        val_list = [vals[i].item() for i in range(len(vals))]
        vals = val_list
    graphs = load_graphs(
        df,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_lattice=use_lattice,
    )
    if mean_train is None:
        mean_train = np.mean(vals)
        std_train = np.std(vals)
        data = StructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            mean_train=mean_train,
            std_train=std_train,
        )
    else:
        data = StructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
            classification=classification,
            mean_train=mean_train,
            std_train=std_train,
        )
    return data, mean_train, std_train


def dump_data(train_data, val_data, dataset_path):
    """"helper function to dump the data"""
    logging.info("start dumping the data.....")
    graphs_map = {"x": 0, "edge_index": 1, "edge_attr": 2}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    ################################################# dump train data
    x_train = []
    edge_index_train = []
    edge_attr_train = []
    label_train = []
    for d in train_data:
        x_train.append(np.array(d[0][graphs_map['x']]))
        edge_index_train.append(np.array(d[0][graphs_map['edge_index']]))
        edge_attr_train.append(np.array(d[0][graphs_map['edge_attr']]))
        label_train.append(np.array(d[2]))
    with os.fdopen(os.open(dataset_path['x_train_path'], flags, modes), 'wb') as fh:
        pk.dump(x_train, fh)
    with os.fdopen(os.open(dataset_path['edge_index_train_path'], flags, modes), 'wb') as fh:
        pk.dump(edge_index_train, fh)
    with os.fdopen(os.open(dataset_path['edge_attr_train_path'], flags, modes), 'wb') as fh:
        pk.dump(edge_attr_train, fh)
    with os.fdopen(os.open(dataset_path['label_train_path'], flags, modes), 'wb') as fh:
        pk.dump(label_train, fh)
    ################################################# dump validation data
    x_val = []
    edge_index_val = []
    edge_attr_val = []
    label_val = []
    for d in val_data:
        x_val.append(np.array(d[0][graphs_map['x']]))
        edge_index_val.append(np.array(d[0][graphs_map['edge_index']]))
        edge_attr_val.append(np.array(d[0][graphs_map['edge_attr']]))
        label_val.append(np.array(d[2]))

    with os.fdopen(os.open(dataset_path['x_val_path'], flags, modes), 'wb') as fh:
        pk.dump(x_val, fh)
    with os.fdopen(os.open(dataset_path['edge_index_val_path'], flags, modes), 'wb') as fh:
        pk.dump(edge_index_val, fh)
    with os.fdopen(os.open(dataset_path['edge_attr_val_path'], flags, modes), 'wb') as fh:
        pk.dump(edge_attr_val, fh)
    with os.fdopen(os.open(dataset_path['label_val_path'], flags, modes), 'wb') as fh:
        pk.dump(label_val, fh)


def get_train_val_loaders(
        dataset: str = "dft_3d",
        dataset_array=[],
        target: str = "formation_energy_peratom",
        atom_features: str = "cgcnn",
        n_train=None,
        n_val=None,
        n_test=None,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        line_graph: bool = True,
        split_seed: int = 100,
        id_tag: str = "jid",
        use_canonize: bool = False,
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        classification_threshold: Optional[float] = None,
        keep_data_order=False,
        use_lattice=False,
        dataset_path=None
):
    """Help function to set up JARVIS train and val dataloaders."""
    # data loading
    mean_train = None
    std_train = None
    train_dataset = dataset_path['x_train_path']
    val_dataset = dataset_path['x_val_path']
    if (os.path.exists(train_dataset) and os.path.exists(val_dataset)):
        logging.info("Loading from saved file...")
        return
    logging.info("No existing saved file...Generate from scratch")
    with open('jdft_3d-12-12-2022.json') as f:
        dataset_array = json.load(f)
    if not dataset_array:
        d = jdata(dataset)
    else:
        d = dataset_array
    dat = []
    all_targets = []
    for i in d:
        if isinstance(i[target], list):  # multioutput target
            all_targets.append(np.array(i[target]))
            dat.append(i)
        elif (i[target] is not None and i[target] != "na" and not math.isnan(i[target])):
            dat.append(i)
            all_targets.append(i[target])
    id_train, id_val, _ = get_id_train_val_test(
        total_size=len(dat),
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        keep_data_order=keep_data_order,
    )
    dataset_train = [dat[x] for x in id_train]
    dataset_val = [dat[x] for x in id_val]

    logging.info("get train data ................................")
    train_data, mean_train, std_train = get_dataset(
        dataset=dataset_train,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        use_canonize=use_canonize,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        use_lattice=use_lattice,
    )
    logging.info("get val data ................................")
    val_data, _, _ = get_dataset(
        dataset=dataset_val,
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        use_canonize=use_canonize,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        use_lattice=use_lattice,
        mean_train=mean_train,
        std_train=std_train,
    )
    dump_data(train_data, val_data, dataset_path)
