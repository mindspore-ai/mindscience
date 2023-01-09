# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Predictor"""
import os
import csv
import glob
import gzip
import secrets
import stat

import json
import mindspore as ms
import numpy as np

from utils import load_predicted_pdb, seq2onehot, datapipe
from model.deepfri import DeepFRI


class Predictor:
    """
        Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """

    def __init__(self, model_prefix, config_path, gcn=True):
        self.model_prefix = model_prefix
        self.config_path = config_path
        self.gcn = gcn

        # load parameters
        with open(self.config_path + "_model_params.json") as json_file:
            metadata = json.load(json_file)
        self.go_names = np.asarray(metadata['gonames'])
        self.go_terms = np.asarray(metadata['goterms'])
        self.thresh = 0.1 * np.ones(len(self.go_terms))
        # load model
        net = DeepFRI(metadata['input_dim'], metadata['output_dim'], metadata['gc_dims'],
                      metadata['fc_dims'], metadata['dropout'], train=False, lstm_input_dim=512)
        param_dict = ms.load_checkpoint(self.model_prefix + '.ckpt')
        ms.load_param_into_net(net, param_dict)
        self.model = ms.Model(net)
        # init params
        self.chain2path = {}
        self.y_hat = np.zeros((1, 1))
        self.test_prot_list = []
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}

    @staticmethod
    def load_cmap(filename, cmap_thresh=10.0):
        """load contact map"""
        if filename.endswith('.pdb'):
            dis, seq = load_predicted_pdb(filename)
            adj = np.double(dis < cmap_thresh)
        elif filename.endswith('.npz'):
            cmap = np.load(filename)
            if 'C_alpha' not in cmap:
                raise ValueError("C_alpha not in *.npz dict.")
            dis = cmap['C_alpha']
            adj = np.double(dis < cmap_thresh)
            seq = str(cmap['seqres'])
        elif filename.endswith('.pdb.gz'):
            rnd_fn = "".join([secrets.token_hex(10), '.pdb'])
            with gzip.open(filename, 'rb') as f:
                fr = f.read().decode()
            flags = os.O_WRONLY | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(rnd_fn, flags, modes), 'w') as out:
                out.write(fr)
            dis, seq = load_predicted_pdb(rnd_fn)
            adj = np.double(dis < cmap_thresh)
            os.remove(rnd_fn)
        else:
            raise ValueError("File must be given in *.npz or *.pdb format.")

        one_hot = seq2onehot(seq)
        one_hot = one_hot.reshape(1, *one_hot.shape)
        adj = adj.reshape(1, *adj.shape)
        return adj, one_hot, seq

    def predict(self, test_prot, cmap_thresh=10.0, chain='query_prot'):
        """predict"""
        print("### Computing predictions on a single protein...")
        self.y_hat = np.zeros((1, len(self.go_terms)), dtype=float)
        self.test_prot_list = [chain]
        if self.gcn:
            adj, seq_1hot, seq = self.load_cmap(test_prot, cmap_thresh=cmap_thresh)
            adj = ms.Tensor(adj, ms.float32)
            seq_1hot = ms.Tensor(seq_1hot, ms.float32)
            y = self.model.predict_network(adj, seq_1hot).asnumpy()[:, :, 0].reshape(-1)
            self.y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[adj, seq_1hot], seq]
            go_idx = np.where(y >= self.thresh)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                try:
                    self.goidx2chains[idx].add(chain)
                    self.prot2goterms[chain].append((self.go_terms[idx], self.go_names[idx], float(y[idx])))
                except KeyError:
                    print("KeyError")

    def predict_from_pdb_dir(self, dir_name, cmap_thresh=10.0):
        """predict from pdb directory"""
        print("### Computing predictions from directory with PDB files...")
        pdb_fn_list = glob.glob(dir_name + '/*.pdb*')
        self.chain2path = {pdb_fn.split('/')[-1].split('.')[0]: pdb_fn for pdb_fn in pdb_fn_list}
        self.test_prot_list = list(self.chain2path.keys())
        self.y_hat = np.zeros((len(self.test_prot_list), len(self.go_terms)), dtype=float)

        for i, chain in enumerate(self.test_prot_list):
            try:
                adj, seq_1hot, seq = self.load_cmap(self.chain2path[chain], cmap_thresh=cmap_thresh)
            except KeyError:
                print("KeyError")
            adj = ms.Tensor(adj, ms.float32)
            seq_1hot = ms.Tensor(seq_1hot, ms.float32)
            y = self.model.predict_network(adj, seq_1hot).asnumpy()[:, :, 0].reshape(-1)
            self.y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[adj, seq_1hot], seq]
            go_idx = np.where(y >= self.thresh)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                try:
                    self.goidx2chains[idx].add(chain)
                    self.prot2goterms[chain].append((self.go_terms[idx], self.go_names[idx], float(y[idx])))
                except KeyError:
                    print("KeyError")

    def predict_from_npz_dir(self, dir_name, cmap_thresh=10.0):
        """predict from catalogue"""
        print("### Computing predictions from catalogue...")
        npz_fn_list = glob.glob(dir_name + '/*.npz*')
        self.chain2path = {npz_fn.split('/')[-1].split('.')[0]: npz_fn for npz_fn in npz_fn_list}
        self.test_prot_list = list(self.chain2path.keys())
        self.y_hat = np.zeros((len(self.test_prot_list), len(self.go_terms)), dtype=float)

        for i, chain in enumerate(self.test_prot_list):
            try:
                adj, seq_1hot, seq = self.load_cmap(self.chain2path[chain], cmap_thresh=cmap_thresh)
            except KeyError:
                print("KeyError")
            adj = ms.Tensor(adj, ms.float32)
            seq_1hot = ms.Tensor(seq_1hot, ms.float32)
            y = self.model.predict_network(adj, seq_1hot).asnumpy()[:, :, 0].reshape(-1)
            self.y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[adj, seq_1hot], seq]
            go_idx = np.where(y >= self.thresh)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                try:
                    self.goidx2chains[idx].add(chain)
                    self.prot2goterms[chain].append((self.go_terms[idx], self.go_names[idx], float(y[idx])))
                except KeyError:
                    print("KeyError")

    def save_predictions(self, output_fn):
        """save predictions"""
        print("### Saving predictions to *.json file...")
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(output_fn, flags, modes), 'w') as fw:
            out_data = {'pdb_chains': self.test_prot_list,
                        'y_hat': self.y_hat.tolist(),
                        'goterms': self.go_terms.tolist(),
                        'gonames': self.go_names.tolist()}
            json.dump(out_data, fw, indent=1)

    def export_csv(self, output_fn, verbose):
        """export csv"""
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(output_fn, flags, modes), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"')
            writer.writerow(['### Predictions made by DeepFRI.'])
            writer.writerow(['Protein', 'GO_term/EC_number', 'Score', 'GO_term/EC_number name'])
            if verbose:
                print('Protein', 'GO-term/EC-number', 'Score', 'GO-term/EC-number name')
            for prot in self.prot2goterms:
                try:
                    sorted_rows = sorted(self.prot2goterms[prot], key=lambda x: x[2], reverse=True)
                    for row in sorted_rows:
                        if verbose:
                            print(prot, row[0], '{:.5f}'.format(row[2]), row[1])
                        writer.writerow([prot, row[0], '{:.5f}'.format(row[2]), row[1]])
                except KeyError:
                    print("KeyError")
        csv_file.close()

    def compute_precision(self, path, ont='mf'):
        """compute precision"""
        print("### Computing precision...")

        dim = len(self.go_terms)
        dataset = datapipe(path, n_go_terms=dim, channels=26, cmap_thresh=10.0, ont=ont)

        len_t = 500
        threshold = [i * (1 / len_t) for i in range(len_t)]
        threshold = [i * np.ones(dim) for i in threshold]

        labels = np.zeros((dim))
        predictions = np.zeros((len_t, dim))
        positives = np.zeros((len_t, dim))

        step = 1
        for data in dataset.create_dict_iterator():
            adj, seq_1hot, label = data['cmap'], data['seq'], data['labels'].asnumpy()[:, :, 0]
            prediction = self.model.predict(adj, seq_1hot).asnumpy()[:, :, 0]
            for i in range(label.shape[0]):
                pred = prediction[i]
                target = label[i]
                y = np.where(target > 0)[0]
                labels[y] += 1
                for j in range(len_t):
                    x = np.where(pred >= threshold[j])[0]
                    for _, xk in enumerate(x):
                        predictions[j, xk] += 1
                        if xk in y:
                            positives[j, xk] += 1
            if step % 5 == 0:
                print('complete {} batch'.format(step))
            step += 1
        print('successfully complete confusion matrix')

        print('Threshold\tPrecision\tRecall\t\tFscore')
        for i in range(len_t):
            precision = []
            recall = []
            for j in range(dim):
                if predictions[i, j] != 0:
                    precision.append(positives[i, j] / predictions[i, j])
                if labels[j] != 0:
                    recall.append(positives[i, j] / labels[j])
            precision = np.asarray(precision)
            recall = np.asarray(recall)
            mean_pre = np.mean(precision)
            mean_recall = np.mean(recall)
            f_score = 2 * mean_pre * mean_recall / (mean_pre + mean_recall)
            print('{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(threshold[i][0], mean_pre, mean_recall, f_score))
