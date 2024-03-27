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
"""
dataset
"""
import os
import ssl
import sys
import errno
import pickle
import zipfile
import numpy as np
from tqdm import tqdm
import os.path as osp
from six.moves import urllib

import rdkit
from rdkit import Chem
from rdkit import RDLogger

from mindspore import Tensor, float16, float32, float64
from mindchemistry.e3 import radius_graph
from mindchemistry.e3.o3 import Irreps, SphericalHarmonics

from mindchemistry.graph.dataloader import DataLoaderBase
from mindchemistry.graph.graph import AggregateEdgeToNode


RDLogger.DisableLog('rdApp.*')

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = np.array([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
           'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']

thermo_targets = ['U', 'U0', 'H', 'G']


def download_url(url, folder, log=True):
    """Download raw data"""
    filename = url.rpartition('/')[2]
    filename = filename if filename[0] == '?' else filename.split('?')[0]
    path = osp.join(folder, filename)

    if osp.exists(path):
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path
    if log:
        print(f'Downloading {url}', file=sys.stderr)

    try:
        os.makedirs(osp.expanduser(osp.normpath(folder)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(folder):
            raise e

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with open(path, 'wb') as f:
        f.write(data.read())
    return path


def extract_zip(path, folder):
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def files_exist(files):
    return len(files) != 0 and all([osp.exists(f) for f in files])


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


class TargetGetter(object):
    """ Gets relevant target """
    def __init__(self, target):
        self.target = target
        self.target_idx = targets.index(target)

    def __call__(self, data):
        data = data[0, self.target_idx]
        return data


def get_cormorant_features(one_hot, charges, charge_power, charge_scale):
    """ Create input features as described in section 7.3 of https://arxiv.org/pdf/1906.04015.pdf """
    charge_np = (charges.reshape(charges.shape + (-1,)) / charge_scale).pow(
        np.arange(charge_power + 1., dtype=np.float32))
    charge_np = charge_np.reshape(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.reshape(one_hot.shape + (-1,)) * charge_np).reshape(charges.shape[:2] + (-1,))
    return atom_scalars


class QM9:
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper. """

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root, target, radius, partition, lmax_attr, feature_type="one_hot", dtype=float32):
        assert feature_type in ["one_hot", "cormorant"], "Please use valid features"
        assert target in targets
        assert partition in ["train", "valid", "test"]
        self._dtype = {
            float16: np.float16,
            float32: np.float32,
            float64: np.float64
        }[dtype]
        self.root = osp.abspath(osp.join(root, "qm9"))
        self.target = target
        self.radius = radius
        self.partition = partition
        self.feature_type = feature_type
        self.lmax_attr = lmax_attr
        self.transform = TargetGetter(self.target)

        self.process_dir = osp.join(self.root, 'processed')
        self.process_path = osp.join(self.process_dir, self.processed_file_names)
        self._download()
        self._pre_process()

        with open(self.process_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.x_lst, self.pos_lst, self.y_lst = data_dict['x'], data_dict['pos'], data_dict['y']

    def _download(self):
        if files_exist(self.raw_paths):
            return
        makedirs(self.raw_dir)
        self.download()

    def _pre_process(self):
        if files_exist([self.process_path]):
            return
        print('Processing...', file=sys.stderr)
        makedirs(self.process_dir)

        self.pre_process()
        print('Done!', file=sys.stderr)

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @ property
    def raw_file_names(self):
        try:
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            raise "Please install rdkit"

    @property
    def raw_paths(self):
        files = self.raw_file_names
        return [osp.join(self.raw_dir, f) for f in files]

    @ property
    def processed_file_names(self):
        return "_".join(
            [self.partition, "r_" + str(np.round(self.radius, 2)), self.feature_type, "l_" + str(self.lmax_attr)]
        ) + '.pt'

    def download(self):
        print("i'm downloading", self.raw_dir, self.raw_url)
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3195404'),
                  osp.join(self.raw_dir, 'uncharacterized.txt'))

    def pre_process(self):
        print("Processing", self.partition, "with radius=" + str(np.round(self.radius, 2)) +
              ",", "l_attr=" + str(self.lmax_attr), "and", self.feature_type, "features.")
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = np.array(target, dtype=self._dtype)
            target = np.concatenate([target[:, 3:], target[:, :3]], axis=-1)
            target = target * conversion.reshape(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        # Create splits identical to Cormorant
        Nmols = len(suppl) - len(skip)
        Ntrain = 100000
        Ntest = int(0.1*Nmols)
        Nvalid = Nmols - (Ntrain + Ntest)

        np.random.seed(0)
        data_perm = np.random.permutation(Nmols)
        train, valid, test = np.split(data_perm, [Ntrain, Ntrain+Nvalid])
        indices = {"train": train, "valid": valid, "test": test}

        j = 0
        x_lst = []
        y_lst = []
        pos_lst = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue
            if j not in indices[self.partition]:
                j += 1
                continue
            j += 1

            # Get pos
            N = mol.GetNumAtoms()
            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = np.array(pos, dtype=self._dtype)
            pos_lst.append(pos)

            type_idx = []
            atomic_number = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())

            # Get x
            z = np.array(atomic_number, dtype=np.int64)
            if self.feature_type == "one_hot":
                x = np.eye(len(types))[np.array(type_idx)]
            elif self.feature_type == "cormorant":
                one_hot = np.eye(len(types))[np.array(type_idx)]
                x = get_cormorant_features(one_hot, z, 2, z.max())
            x_lst.append(x.astype(self._dtype))

            # Get y
            y = np.expand_dims(target[i], axis=0)
            y = y if self.transform is None else self.transform(y)
            y = np.array([y], dtype=self._dtype)
            y_lst.append(y)
            
        # save processed data
        data_dict = {'x': x_lst, 'pos': pos_lst, 'y': y_lst}
        with open(self.process_path, 'wb') as f:
            pickle.dump(data_dict, f)

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self.y_lst)

    def __getitem__(self, idx):
        return self.x_lst[idx], self.pos_lst[idx], self.y_lst[idx]


class QM9_DataLoader(DataLoaderBase):
    def __init__(self,
                 batch_size,
                 x,
                 edge_index,
                 node_attr,
                 edge_attr,
                 edge_dist,
                 label,
                 shuffle_dataset,
                 dtype
    ):
        super().__init__(
            batch_size=batch_size,
            edge_index=edge_index,
            label=label,
            node_attr=node_attr,
            edge_attr=edge_attr,
            shuffle_dataset=shuffle_dataset,
            dynamic_batch_size=False
        )

        self.x = x
        self.edge_dist = edge_dist
        self.dtype = dtype


    def shuffle_action(self):
        """shuffle_action"""
        indices = self.shuffle_index()
        self.x = [self.x[i] for i in indices]
        self.edge_index = [self.edge_index[i] for i in indices]
        self.label = [self.label[i] for i in indices]
        self.node_attr = [self.node_attr[i] for i in indices]
        self.edge_attr = [self.edge_attr[i] for i in indices]
        self.edge_dist = [self.edge_dist[i] for i in indices]


    def __iter__(self):
        if self.shuffle_dataset:
            self.shuffle()
        else:
            self.restart()

        while self.index < self.max_start_sample:
            # pylint: disable=W0612
            edge_index_step, node_batch_step, node_mask, edge_mask, batch_size_mask, node_num, edge_num, batch_size \
                = self.gen_common_data(self.node_attr, self.edge_attr)

            ### can be customized to generate different attributes or labels according to specific dataset
            x_step = self.gen_node_attr(self.x, batch_size, node_num)
            node_attr_step = self.gen_node_attr(self.node_attr, batch_size, node_num)
            edge_attr_step = self.gen_edge_attr(self.edge_attr, batch_size, edge_num)
            label_step = self.gen_global_attr(self.label, batch_size)
            edge_dist_step = self.gen_edge_attr(self.edge_dist, batch_size, edge_num)

            self.add_step_index(batch_size)

            # cast data type
            x_step = x_step.astype(self.dtype)
            node_attr_step = node_attr_step.astype(self.dtype)
            edge_attr_step = edge_attr_step.astype(self.dtype)
            edge_dist_step = edge_dist_step.astype(self.dtype)
            label_step = label_step.astype(self.dtype)
            node_mask = node_mask.astype(self.dtype)
            edge_mask = edge_mask.astype(self.dtype)

            yield x_step, node_attr_step, edge_attr_step, edge_dist_step, label_step, edge_index_step, \
                node_batch_step, node_mask, edge_mask, node_num, batch_size


def get_attrs(sh, pos, edge_index, dim_size, dtype):
    rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
    edge_dist = (rel_pos ** 2).sum(-1, keepdims=True)
    edge_attr = sh(Tensor(rel_pos))
    edge_attr = edge_attr.astype(dtype)

    scatter = AggregateEdgeToNode("mean", dim=1)
    node_attr = scatter(edge_attr, Tensor(edge_index), dim_size=dim_size)
    node_attr = node_attr.astype(dtype)
    return edge_attr, node_attr, edge_dist


def calc_stats(lst):
    mean = np.mean(lst)
    mad = np.mean(np.abs(lst - mean))
    return [mean, mad]
    
    
def create_training_dataset(data_params, dtype):
    dataset_dir = data_params['dataset_dir']
    pre_process_file = data_params['pre_process_file']
    
    dataloaders = {}
    train_stats = None
    for phase in ['train', 'valid', 'test']:
        pre_process_file_path = osp.join(dataset_dir, phase + '_' + pre_process_file)
        x_lst, edge_index_lst, edge_attr_lst, node_attr_lst, edge_dist_lst, label_lst = [], [], [], [], [], []
        if osp.exists(pre_process_file_path):
            with open(pre_process_file_path, 'rb') as f:
                data_dict = pickle.load(f)
            x_lst, edge_index_lst, edge_attr_lst, node_attr_lst, edge_dist_lst, label_lst = (
                data_dict['x'],
                data_dict['edge_index'],
                data_dict['edge_attr'],
                data_dict['node_attr'],
                data_dict['edge_dist'],
                data_dict['label']
            )
        else:
            lmax_attr = data_params['lmax_attr']
            sh = SphericalHarmonics(Irreps.spherical_harmonics(lmax_attr), normalize=True, normalization='component')
            data_source = QM9(dataset_dir, data_params['target'], data_params['radius'], phase, lmax_attr, dtype=dtype)
            for item in tqdm(data_source):
                x, pos, label = item
                edge_index, _batch = radius_graph(pos, data_params['radius'])
                dim_size = int(edge_index[1].max()) + 1
                edge_attr, node_attr, edge_dist = get_attrs(sh, pos, edge_index, dim_size, dtype)
                x_lst.append(x)
                edge_index_lst.append(edge_index)
                edge_attr_lst.append(edge_attr.asnumpy())
                node_attr_lst.append(node_attr.asnumpy())
                edge_dist_lst.append(edge_dist.astype(pos.dtype))
                label_lst.append(label)
                
            data_dict = {
                "x": x_lst,
                "edge_index": edge_index_lst,
                "edge_attr": edge_attr_lst,
                "node_attr": node_attr_lst,
                "edge_dist": edge_dist_lst,
                "label": label_lst
            }
            with open(pre_process_file_path, 'wb') as f:
                pickle.dump(data_dict, f)
    
        if phase == "train":
            train_stats = calc_stats(label_lst)
            
        tmp_dataloader = QM9_DataLoader(
            batch_size=data_params['batch_size'], 
            x=x_lst, 
            edge_index=edge_index_lst, 
            node_attr=node_attr_lst, 
            edge_attr=edge_attr_lst, 
            edge_dist=edge_dist_lst,
            label=label_lst,
            shuffle_dataset=True if phase == 'train' else False,
            dtype=dtype
        )
        dataloaders[phase] = tmp_dataloader
    return dataloaders['train'], dataloaders['valid'], dataloaders['test'], train_stats
