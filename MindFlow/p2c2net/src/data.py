'''
Data processing and visualization utilities for P2C2Net.
'''


import os
import random
import matplotlib.pyplot as plt
from mindspore import Tensor
import numpy as np
import scipy.io
import pandas as pd

def generate_dataset(data, train_win, icnum, nolap=True):
    '''
    Generate train dataset
    '''
    gap = train_win if nolap else 8
    start = list(range(0, len(data[0]) - train_win + 1, gap))

    random.shuffle(start)
    train_set = []
    for i in start:
        for j in range(icnum):
            train_set.append(data[j][i:i + train_win])
    all_shuffle = []
    indices = list(range(len(train_set)))
    random.shuffle(indices)
    for i in indices:
        all_shuffle.append(train_set[i])
    all_shuffle = np.array(all_shuffle)
    return all_shuffle

def plot_loss(train_loss, save_dir):
    '''
    Plot training loss 
    '''
    i = list(range(1, len(train_loss) + 1))
    plt.plot(i, train_loss, color="red", label="train_loss")
    plt.title("Loss_iters_burgers", fontsize=24)
    plt.xlabel("iters", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.tick_params(axis="both", labelsize=14)
    plt.legend(fontsize=16)
    plt.savefig(save_dir + "/train_burgers_loss.png", dpi=600)
    plt.close()
    print("plot burgers loss over")

class MyDataset:
    '''
    Custom dataset class
    '''
    def __init__(self, data_features, eps=1.0e-5):
        self.len = len(data_features)
        self.features = data_features
        self.eps = eps

    def __getitem__(self, index):
        feature = self.features[index]
        x = feature[0:1, ...]
        seq = feature[1:, ...]
        return x, seq

    def __len__(self):
        return self.len

class UnitGaussianNormalizer():
    '''
    Encode data and decode data by mean and std
    '''
    def __init__(self, x, normalization_dim = None, eps=1.0e-5):
        super().__init__()

        if normalization_dim is None:
            normalization_dim = []

        self.mean = np.mean(x,  axis=normalization_dim, keepdims=True)
        self.std  = np.std(x,  axis=normalization_dim, keepdims=True)
        print("mean, std", self.mean, self.std)
        self.eps  = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps
        mean = self.mean
        x = x * std + mean
        return x

def get_data(exp_dir, num=5, down=4):
    '''
    Load data
    '''
    all_data = []
    for i in range(1, num + 1):
        data_dir = os.path.join(exp_dir, 'data', 'Burgers_2101x2x104x104_[RK4,R=200,dt=0_001,#seed' + str(i) + '].mat')
        uv = scipy.io.loadmat(data_dir)['uv']
        cur_uv = np.ascontiguousarray(uv[100:2101, ...], dtype=np.float32)
        cur_truth = cur_uv[:, :, ::down, ::down]
        all_data.append(cur_truth)
    all_data = np.array(all_data)   # [5, 2000, 2, 26, 26]

    return all_data


def infer(model, init, infersteps, compute_dtype):
    model.steps = infersteps
    init = Tensor(init, dtype=compute_dtype)
    outputs_uv = model(init)
    return outputs_uv

def evl_error(test_data, model, inferstep,
              delta_t,  resolution, err_save_dir,
              compute_dtype, normalizer=None,
              ploterrorprop=True, plotsnapshots=True
):
    '''
    Evaluate model prediction errors, generate plots and save error metrics.
    
    Args:
        test_data: Test data
        model: Trained model
        inferstep: Number of inference steps
        delta_t: Time step size
        resolution: Spatial resolution (height, width)
        err_save_dir: Directory to save error results and plots
        compute_dtype (mindspore.dtype): Computation data type
        normalizer (UnitGaussianNormalizer, optional): Data normalizer for denormalizing predictions
        ploterrorprop (bool, optional): Whether to plot error propagation, defaults to True
        plotsnapshots (bool, optional): Whether to plot data snapshots at final inference step, defaults to Truep
    
    Returns:
        rmse (float): Root Mean Square Error
        mae (float): Mean Absolute Error
        mnad (float): Mean Normalized Absolute Deviation
        hct (float): High Correlation Time
    '''
    pred = []
    ground_truth = []
    for i, data in enumerate(test_data):
        init = data[0:1]
        truth = data[1:inferstep]
        output = infer(model, init, inferstep, compute_dtype)
        output = output.asnumpy()
        if normalizer:
            output = normalizer.decode(output)
            truth = normalizer.decode(truth)
            output = output.squeeze()
            truth = truth.squeeze()

        if plotsnapshots:
            print("snapshots plotting")
            plot_snapshots(output[-1, :, :, :],
                           truth[-1, :, :, :],
                           resolution,
                           os.path.join(err_save_dir, f"test_set_{i+1}_burgers-snap")
            )

        if ploterrorprop:
            print("error propagation plotting")
            uv_truth = truth.transpose(1, 0, 2, 3)
            uv_net = output.transpose(1, 0, 2, 3)
            plot_err_prop(uv_truth,
                          uv_net,
                          delta_t,
                          os.path.join(err_save_dir, f"test_set_{i+1}_burgers-err-propa.png")
            )

        pred.append(np.expand_dims(output.transpose(1, 0, 2, 3), axis=0))
        ground_truth.append(np.expand_dims(truth.transpose(1, 0, 2, 3), axis=0))

    pred = np.concatenate(pred, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    rmse = np.sqrt(np.mean(np.sum((ground_truth - pred)**2, axis=1)))
    mae = np.mean(np.abs(pred - ground_truth))
    truth_norms = np.linalg.norm(ground_truth, axis=1)
    mnad = np.mean(np.linalg.norm(ground_truth - pred, axis=1) / (np.max(truth_norms) - np.min(truth_norms)))
    hct = 0
    for i in range(ground_truth.shape[0]):
        pcc = np.corrcoef(ground_truth[i].flatten(), pred[i].flatten())[0, 1]
        if not np.isnan(pcc) and pcc > 0.8:
            hct += delta_t

    print("rmse, mas, mnad, hct", rmse, mae, mnad, hct)
    error_data = {
    'Metric': ['RMSE', 'MAE', 'MNAD', 'HCT'],
    'Value': [rmse, mae, mnad, hct]
    }
    df = pd.DataFrame(error_data)
    df.to_csv(os.path.join(err_save_dir, 'evaluation_metrics.csv'), index=False)
    return rmse, mae, mnad, hct

def plot_snapshots(uv_truth, uv_net, resolution, fig_save_dir):
    '''
    Plot snapshots of ground truth and predicted data.
    '''
    z1 = uv_truth[0, :, :]
    z2 = uv_net[0, :, :]
    z3 = uv_truth[1, :, :]
    z4 = uv_net[1, :, :]
    xx = np.linspace(0, 1, resolution)
    yy = np.linspace(0, 1, resolution)
    x, y = np.meshgrid(xx, yy)

    h1 = np.max([np.max(z1), np.max(z2)])
    l1 = np.min([np.min(z1), np.min(z2)])
    level1 = np.linspace(l1, h1, 10, endpoint=True)
    h2 = np.max([np.max(z3), np.max(z4)])
    l2 = np.min([np.min(z3), np.min(z4)])
    level2 = np.linspace(l2, h2, 10, endpoint=True)

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    ax1, ax2, ax3, ax4 = axs.flatten()
    size = 24
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.25, hspace=0.4)

    ax1.contourf(x, y, z1, level1, cmap='coolwarm')
    ax1.set_title(r'$Ref.$', fontsize=size+8)
    ax1.text(-0.2, 0.5, r"$u$",
            ha='center',
            va='center',
            fontsize=size+12,
            transform=ax1.transAxes)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')

    ax2.contourf(x, y, z2, level1, cmap='coolwarm')
    ax2.set_title(r'$P^2C^2Net$', fontsize=size+8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('equal')

    ax3.contourf(x, y, z3, level2, cmap='coolwarm')
    ax3.set_title(r'$Ref.$', fontsize=size+8)
    ax3.text(-0.25, 0.5, r"$v$",
            ha='center',
            va='center',
            fontsize=size+12,
            transform=ax3.transAxes)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect('equal')

    ax4.contourf(x, y, z4, level2, cmap='coolwarm')
    ax4.set_title(r'$P^2C^2Net$', fontsize=size+8)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_aspect('equal')

    plt.savefig(fig_save_dir, dpi=600)
    plt.clf()
    plt.close()

def plot_err_prop(uv_truth, uv_net, dt, fig_save_dir):
    '''
    Plot error propagation.
    '''
    eps = 1e-4
    mse = np.mean((uv_truth - uv_net) ** 2, axis=(0, 2, 3))
    accum = np.array([[i * dt, eps + np.sqrt(mse[:i + 1].mean())] for i in range(0, uv_net.shape[1], 1)])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0.13, 0.12, 0.8, 0.80])
    ax.plot(accum[:, 0], accum[:, 1], alpha=0.8, linewidth=2, color='black', label=r'$P^2C^2Net$')

    ax.set_xlim([0, 1.4])
    ax.set_ylim([eps, 1e0])
    ax.set_xticks([0.0, 0.7, 1.4])
    ax.set_yticks([eps, 1e-1])
    ax.set_yscale('log')
    ax.set_ylabel(r'a-RMSE', fontsize=14)
    ax.set_xlabel('t(s)', fontsize=14, labelpad=-0.0)
    ax.tick_params(labelsize=14, direction='in')
    ax.set_title('Error propagation', fontsize=16)
    plt.legend()

    plt.savefig(fig_save_dir, dpi=600)
    plt.close()


def cal_reffe(output, truth):
    nume = np.linalg.norm(output - truth)
    deno = np.linalg.norm(truth)
    epsino = nume / deno
    return epsino

def ensure_directories(*directory_paths):
    """
    Create directories if they don't exist.
    """
    for directory_path in directory_paths:
        os.makedirs(directory_path, exist_ok=True)
