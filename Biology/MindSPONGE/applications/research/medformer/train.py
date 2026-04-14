#    Copyright 2025 Yuanhanyu Luo & Linchang Zhu

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
This module implements the training and evaluation pipeline for the GenePertFormer model.
It handles data loading, preprocessing, model definition, training loop, and result visualization.
"""
import argparse
import os
import json
import logging
from datetime import datetime

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

import mindspore as ms
from mindspore import nn, context
import mindspore.dataset as ds

# Grouped scipy imports
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import wandb

from rdkit import Chem
from rdkit.Chem import AllChem

from module.MedFormer import GenePertFormer


## Execution Mode
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# utils
def parse_args():
    """
    Parses command line arguments for the perturbation model.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MindSpore version of perturbation model")
    parser.add_argument("--split_key", default="drug_split_0", type=str)
    parser.add_argument("--ablation", default=None, type=str)
    return parser.parse_args()


def shuffle_adata(adata_obj):
    """
    Shuffles the AnnData object in place.

    Args:
        adata_obj (anndata.AnnData): The AnnData object to be shuffled.

    Returns:
        anndata.AnnData: The shuffled AnnData object.
    """
    if sparse.issparse(adata_obj.X):
        adata_obj.X = adata_obj.X.A
    perm = np.random.permutation(adata_obj.shape[0])
    return adata_obj[perm, :]


def train_valid_test_split(adata_obj, split_key):
    """
    Splits the AnnData object into training, validation, and test sets,
    including control samples in all sets.

    Args:
        adata_obj (anndata.AnnData): The AnnData object containing the data.
        split_key (str): The observation key used for splitting (e.g., "drug_split_0").

    Returns:
        Tuple[anndata.AnnData, anndata.AnnData, anndata.AnnData]:
            Train, validation, and test AnnData objects.
    """
    shuffled = shuffle_adata(adata_obj)
    adata_ctrl0 = adata_obj[adata_obj.obs["control"] == 0]
    train_idx = adata_ctrl0.obs[adata_ctrl0.obs[split_key] == "train"].index.tolist()
    valid_idx = adata_ctrl0.obs[adata_ctrl0.obs[split_key] == "valid"].index.tolist()
    test_idx = adata_ctrl0.obs[adata_ctrl0.obs[split_key] == "test"].index.tolist()
    ctrl_idx = adata_obj.obs[adata_obj.obs["control"] == 1].index.tolist()

    def subset(idx_list):
        return shuffled[idx_list + ctrl_idx]
    return subset(train_idx), subset(valid_idx), subset(test_idx)


# ------------ Dataset & DataLoader ------------
def drug_smiles_encode(drug_smiles_list: list, num_bits=1024):
    """
    Encodes a list of drug SMILES strings into Morgan fingerprints.

    Args:
        drug_smiles_list (list): A list of SMILES strings.
        num_bits (int): The number of bits for the Morgan fingerprint.

    Returns:
        numpy.ndarray: A NumPy array of drug fingerprints.

    Raises:
        ValueError: If an invalid SMILES string is encountered.
    """
    arr = np.zeros((len(drug_smiles_list), num_bits), dtype=np.float32)
    for i, smiles in enumerate(drug_smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES") # Changed to lazy formatting
        bits = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_bits).ToBitString()
        arr[i] = np.array(list(bits), dtype=np.float32)
    return arr


class GenePertAnnDatasetMS:
    """
    A MindSpore Dataset for GenePert data, wrapping an AnnData object.
    """
    def __init__(self, adata_obj, gene2id_path, control_key="condition", smiles_key="SMILES", cell_key="cell_id"):
        """
        Initializes the GenePertAnnDatasetMS.

        Args:
            adata_obj (anndata.AnnData): The AnnData object containing the data.
            gene2id_path (str): Path to the JSON file mapping gene names to IDs.
            control_key (str): Observation key for control samples.
            smiles_key (str): Observation key for drug SMILES strings.
            cell_key (str): Observation key for cell IDs.
        """
        self.adata = adata_obj
        self.control_key = control_key
        self.smiles_key = smiles_key
        self.cell_key = cell_key
        self.drug_dim = 1024

        with open(gene2id_path, "r", encoding='utf-8') as f:
            self.gene2id = json.load(f)
        self.idx = np.where(self.adata.obs[self.control_key] != "control")[0]
        self.gene_order = self.adata.var_names.tolist()
        cells = self.adata.obs[self.cell_key].unique().tolist()
        self.cell2id = {cid: i for i, cid in enumerate(sorted(cells))}

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.idx)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                gene_ids, treated gene expression, drug fingerprint, cell one-hot encoding, control gene expression.
        """
        si = self.idx[idx]
        sample = self.adata[si]
        treat = sample.X.toarray().flatten()
        ctrl_id = sample.obs["paired_control_index"].values[0]
        ctrl_i = self.adata.obs_names.get_loc(ctrl_id)
        ctrl = self.adata[ctrl_i].X.toarray().flatten()
        gene_ids = np.array([self.gene2id[g] for g in self.gene_order], dtype=np.int32)
        treat = treat.astype(np.float32).reshape(-1, 1)
        ctrl = ctrl.astype(np.float32)
        smiles = sample.obs[self.smiles_key].values[0]
        drug_fp = drug_smiles_encode([smiles], self.drug_dim)[0]
        cid = sample.obs[self.cell_key].values[0]
        cell_onehot = np.eye(len(self.cell2id), dtype=np.float32)[self.cell2id[cid]]
        return gene_ids, treat, drug_fp, cell_onehot, ctrl


def build_ms_dataset(adata_obj, gene2id_path, batch_size=64, shuffle_data=True):
    """
    Builds a MindSpore GeneratorDataset from an AnnData object.

    Args:
        adata_obj (anndata.AnnData): The AnnData object to build the dataset from.
        gene2id_path (str): Path to the JSON file mapping gene names to IDs.
        batch_size (int): Batch size for the dataset.
        shuffle_data (bool): Whether to shuffle the dataset.

    Returns:
        mindspore.dataset.engine.datasets.GeneratorDataset: The built MindSpore dataset.
    """
    ds_src = GenePertAnnDatasetMS(adata_obj, gene2id_path)
    ms_ds = ds.GeneratorDataset(ds_src,
                                ["gene_ids", "gene_expr", "drug_fp", "cell_feat", "control_expr"],
                                shuffle=shuffle_data)
    ms_ds = ms_ds.batch(batch_size, drop_remainder=True)
    return ms_ds


# ------------ training process ------------
args = parse_args()
original_adata = sc.read_h5ad("./Lincs_L1000.h5ad")
train_data, valid_data, test_data = train_valid_test_split(original_adata, args.split_key)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
save_dir = f"./MSmodel_{args.split_key}_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

train_ds = build_ms_dataset(train_data, "./data/gene2id.json", batch_size=64, shuffle_data=True)
valid_ds = build_ms_dataset(valid_data, "./data/gene2id.json", batch_size=64, shuffle_data=False)
test_ds = build_ms_dataset(test_data, "./data/gene2id.json", batch_size=64, shuffle_data=False)

#  define model
model = GenePertFormer(drug_dim=1024, cell_dim=82, hidden_dim=256, use_cell_expr=True, cell_input_dim=978)

# loss
loss_fn = nn.MSELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.0005)

net = nn.WithLossCell(model, loss_fn)
train_net = nn.TrainOneStepCell(net, optimizer)
train_net.set_train()

# wandb init
wandb.init(project="GenePertFormerMS", name=f"ms_{args.split_key}_{datetime.now().strftime('%Y%m%d_%H%M')}")

best_val = 1e9
patience, counter = 5, 0
train_losses, val_losses = [], []

for ep in range(100):
    total, count = 0.0, 0
    for b in train_ds.create_tuple_iterator():
        _, loss = train_net(*b)
        total += loss.asnumpy()
        count += 1
    avg_train = total / count
    train_losses.append(avg_train)

    # validation
    model.set_train(False)
    total, count = 0.0, 0
    for b in valid_ds.create_tuple_iterator():
        out = model(*b[:-1])
        v_loss = loss_fn(out[0], b[1])
        total += v_loss.asnumpy()
        count += 1
    avg_val = total / count
    val_losses.append(avg_val)
    model.set_train(True)

    wandb.log({"epoch": ep, "train_loss": avg_train, "val_loss": avg_val})
    if avg_val < best_val:
        best_val = avg_val
        counter = 0
        ms.save_checkpoint(model, os.path.join(save_dir, "best.ckpt"))
    else:
        counter += 1
        if counter >= patience:
            break

# === results ===
pred_list, true_list = [], []

for batch in test_ds.create_tuple_iterator():
    pred_batch, _, _ = model(*batch[:-1])
    true_batch = batch[1].asnumpy()
    pred_list.append(pred_batch.asnumpy())
    true_list.append(true_batch)

pred_array = np.vstack(pred_list)
true_array = np.vstack(true_list)

r2 = np.nanmean([r2_score(t, p) for t, p in zip(true_array, pred_array)])
pcc = np.nanmean([pearsonr(t, p)[0] for t, p in zip(true_array, pred_array)])
logger.info("Test R²: %.4f, Pearson: %.4f", r2, pcc) # Changed to lazy formatting

np.savez(os.path.join(save_dir, f"{args.split_key}_test_result_ms.npz"),
         pred=pred_array,
         true=true_array,
         r2=r2, pcc=pcc)

wandb.log({"test_r2": r2, "test_pearson": pcc})

# === visualization ===
flat_true = true_array.flatten()
flat_pred = pred_array.flatten()
mask = ~np.isnan(flat_true) & ~np.isnan(flat_pred)
flat_true, flat_pred = flat_true[mask], flat_pred[mask]

sns.set_theme(style="ticks")
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(flat_true, flat_pred, alpha=0.4, s=3, color='steelblue')
ax.plot([flat_true.min(), flat_true.max()], [flat_true.min(), flat_true.max()],
        'r--', linewidth=1.2)
ax.set_xlabel("True Expression", fontsize=12)
ax.set_ylabel("Predicted Expression", fontsize=12)
ax.set_title(f"True vs Predicted (MS)\nR² = {r2:.3f}, PCC = {pcc:.3f}", fontsize=13)
sns.despine()
fig.tight_layout()

fig_path = os.path.join(save_dir, f"{args.split_key}_true_vs_pred_ms.png")
fig.savefig(fig_path, dpi=300)
plt.close(fig)
wandb.log({"true_vs_pred_scatter_ms": wandb.Image(fig_path)})
