"""model test"""
import math
import os

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor, mint, load_checkpoint, load_param_into_net
from mindchemistry.graph.loss import L2LossMask
import numpy as np


from models.cspnet import CSPNet
from models.flow import CSPFlow
from data.dataset import fullconnect_dataset
from data.crysloader import Crysloader as DataLoader


ms.set_seed(1234)
np.random.seed(1234)

class SinusoidalTimeEmbeddings(nn.Cell):
    """time embedding"""
    def __init__(self, dim):
        super(SinusoidalTimeEmbeddings, self).__init__()
        self.dim = dim

    def construct(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = ops.Exp()(mnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = ops.Concat(axis=-1)(
            (ops.Sin()(embeddings), ops.Cos()(embeddings)))
        return embeddings

def test_cspnet():
    """test cspnet.py"""
    ms.set_seed(1234)
    time_embedding = SinusoidalTimeEmbeddings(256)
    cspnet = CSPNet(num_layers=6, hidden_dim=512, num_freqs=128)
    atom_types = Tensor([61, 12, 52, 52, 46, 46], dtype=ms.int32)
    frac_coords = Tensor(
        [[5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
         [6.66666687e-01, 3.33333343e-01, 7.50000000e-01],
         [3.33333343e-01, 6.66666687e-01, 2.50000000e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
         [0.00000000e+00, 0.00000000e+00, 5.00000000e-01]], dtype=ms.float32)
    lengths = Tensor(
        [[3.86215806e+00, 3.86215806e+00, 3.86215806e+00],
         [4.21191406e+00, 4.21191454e+00, 5.75016499e+00]], dtype=ms.float32)
    lattice_polar = Tensor(
        [[0.00000000e+00, 0.00000000e+00, 3.97458431e-1, 5.55111512e-16, 0.00000000e+00, 1.35122609e+00],
         [-2.74653047e-01, 1.58676151e-16, 6.82046943e-17, -5.38849108e-08, -1.27743945e-01, 1.49374068e+00]],
        dtype=ms.float32)
    edge_index = Tensor(
        [[0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
         [0, 1, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5]], dtype=ms.int32)
    node2graph = Tensor([0, 0, 1, 1, 1, 1], dtype=ms.int32)
    node_mask = Tensor([1, 1, 1, 1, 1, 1], dtype=ms.int32)
    edge_mask = Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=ms.int32,)
    tar_lat_polar = Tensor(
        [[-0.5366, 0.5920, 0.2546, 0.4013, -0.0032, 0.6611],
         [-0.5696, 0.6870, 0.2512, 0.4647, 0.0228, 0.5979]]
        )
    tar_coord = Tensor([[-0.7573, 0.2272, -0.4823],
                        [-0.7647, 0.2261, -0.4763],
                        [-0.7841, 0.2948, -0.3861],
                        [-0.7872, 0.2915, -0.3810],
                        [-0.7789, 0.2759, -0.4070],
                        [-0.7785, 0.2757, -0.4070]])

    np.random.seed(1234)
    times = np.random.rand(lengths.shape[0])
    times = ms.tensor(times, dtype=ms.float32)
    t = time_embedding(times)
    lattices_out, coords_out = cspnet(t, atom_types, frac_coords, lattice_polar, node2graph,\
                edge_index, node_mask, edge_mask)
    assert mint.isclose(lattices_out, tar_lat_polar, rtol=1e-4, atol=1e-4).all(), \
        f"For `cspnet`, the output should be {tar_lat_polar}, but got {lattices_out}."
    assert mint.isclose(coords_out, tar_coord, rtol=1e-4, atol=1e-4).all(), \
        f"For `cspnet`, the output should be {tar_coord}, but got {coords_out}."

def test_flow():
    """test flow.py"""
    ms.set_seed(1234)
    cspnet = CSPNet(num_layers=6, hidden_dim=512, num_freqs=128)
    cspflow = CSPFlow(cspnet)
    atom_types = Tensor([61, 12, 52, 52, 46, 46], dtype=ms.int32)
    frac_coords = Tensor(
        [[5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
         [6.66666687e-01, 3.33333343e-01, 7.50000000e-01],
         [3.33333343e-01, 6.66666687e-01, 2.50000000e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
         [0.00000000e+00, 0.00000000e+00, 5.00000000e-01]], dtype=ms.float32)
    lengths = Tensor(
        [[3.86215806e+00, 3.86215806e+00, 3.86215806e+00],
         [4.21191406e+00, 4.21191454e+00, 5.75016499e+00]], dtype=ms.float32)
    angles = Tensor(
        [[9.00000000e+01, 9.00000000e+01, 9.00000000e+01],
         [9.00000000e+01, 9.00000000e+01, 1.20000000e+02]], dtype=ms.float32)
    lattice_polar = Tensor(
        [[0.00000000e+00, 0.00000000e+00, 3.97458431e-1, 5.55111512e-16, 0.00000000e+00, 1.35122609e+00],
         [-2.74653047e-01, 1.58676151e-16, 6.82046943e-17, -5.38849108e-08, -1.27743945e-01, 1.49374068e+00]], \
            dtype=ms.float32)
    num_atoms = Tensor([2, 4], dtype=ms.int32)
    edge_index = Tensor(
        [[0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
         [0, 1, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5]], dtype=ms.int32)
    node2graph = Tensor([0, 0, 1, 1, 1, 1], dtype=ms.int32)
    node_mask = Tensor([1, 1, 1, 1, 1, 1], dtype=ms.int32)
    edge_mask = Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=ms.int32,)
    batch_size_mask = Tensor([1, 1], dtype=ms.int32)

    pred_l, tar_l, pred_f, tar_f = cspflow(atom_types, atom_types, lengths,
                                           angles, lattice_polar, num_atoms, frac_coords, node2graph,
                                           edge_index, node_mask, edge_mask, batch_size_mask)
    out_pred_l = Tensor([[-0.54417396, 0.6183988, 0.25345746, 0.41497535, -0.00219233, 0.6622897],
                         [-0.5647707, 0.68243337, 0.25912297, 0.45234668, 0.01847154, 0.6095263]])
    out_tar_l = Tensor([[-0.02254689, 0.04679973, 0.3856261, -0.08269336, -0.08592724, 0.45530552],
                        [-0.53301036, -0.21067567, -0.05119152, 0.04148455, -0.0907657, 0.3682214]])
    out_pred_f = Tensor([[-0.7662705, 0.24618103, -0.4741043],
                         [-0.77218896, 0.2367004, -0.4617761],
                         [-0.7825796, 0.28697833, -0.38660413],
                         [-0.7888657, 0.2943602, -0.39356205],
                         [-0.7792929, 0.26879176, -0.42642403],
                         [-0.77509487, 0.2633396, -0.41789246]])
    out_tar_f = Tensor([[0.20181239, -0.07186192, -0.40746307],
                        [-0.4028666, 0.18524933, 0.14020872],
                        [-0.31370556, 0.08878523, -0.18229586],
                        [-0.15636778, -0.44619012, 0.13355094],
                        [-0.03352255, -0.15093482, -0.13720155],
                        [-0.2018686, 0.07621789, -0.4946221]])
    assert mint.isclose(pred_l, out_pred_l, rtol=1e-4, atol=1e-4).all(), \
        f"For `cspnet`, the output should be {pred_l}, but got {out_pred_l}."
    assert mint.isclose(pred_f, out_pred_f, rtol=1e-4, atol=1e-4).all(), \
        f"For `cspnet`, the output should be {pred_f}, but got {out_pred_f}."
    assert mint.isclose(tar_l, out_tar_l, rtol=1e-4, atol=1e-4).all(), \
        f"For `cspnet`, the output should be {tar_l}, but got {out_tar_l}."
    assert mint.isclose(tar_f, out_tar_f, rtol=1e-4, atol=1e-4).all(), \
        f"For `cspnet`, the output should be {tar_f}, but got {out_tar_f}."

def test_loss():
    """test loss"""
    ms.set_context(device_target="CPU")
    ckpt_dir = "./ckpt/mp_20"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ms.set_seed(1234)
    batch_size_max = 256

    cspnet = CSPNet(num_layers=6, hidden_dim=512, num_freqs=256)
    cspflow = CSPFlow(cspnet)
    mindspore_ckpt = load_checkpoint("./torch2ms_ckpt/ms_flow.ckpt")
    load_param_into_net(cspflow, mindspore_ckpt)

    loss_func_mse = L2LossMask(reduction='mean')
    def forward(atom_types_step, frac_coords_step, _, lengths_step, angles_step, lattice_polar_step, \
            num_atoms_step, edge_index_step, batch_node2graph, \
            node_mask_step, edge_mask_step, batch_mask, node_num_valid, batch_size_valid):
        pred_l, tar_l, pred_x, tar_x = cspflow(batch_size_valid, atom_types_step, lengths_step,
                                               angles_step, lattice_polar_step, num_atoms_step,
                                               frac_coords_step, batch_node2graph, edge_index_step,
                                               node_mask_step, edge_mask_step, batch_mask)
        mseloss_l = loss_func_mse(pred_l, tar_l, mask=batch_mask, num=batch_size_valid)
        mseloss_x = loss_func_mse(pred_x, tar_x, mask=node_mask_step, num=node_num_valid)
        mseloss = mseloss_l + 10 * mseloss_x

        return mseloss, mseloss_l, mseloss_x

    train_datatset = fullconnect_dataset(name="mp_20", path='./dataset/mp_20/train.csv',
                                         save_path='./dataset/mp_20/train.npy')
    train_loader = DataLoader(batch_size_max, *train_datatset, shuffle_dataset=False)

    for atom_types_batch, frac_coords_batch, property_batch, lengths_batch, \
        angles_batch, lattice_polar_batch, num_atoms_batch,\
        edge_index_batch, batch_node2graph_, node_mask_batch, edge_mask_batch, batch_mask_batch,\
            node_num_valid_, batch_size_valid_ in train_loader:

        result = forward(atom_types_batch, frac_coords_batch, property_batch,
                         lengths_batch, angles_batch, lattice_polar_batch,
                         num_atoms_batch, edge_index_batch, batch_node2graph_,
                         node_mask_batch, edge_mask_batch, batch_mask_batch, node_num_valid_,
                         batch_size_valid_)

        _, mseloss_l, mseloss_x = result
        break
    assert mseloss_l <= 0.7, "The denoising of lattice accuracy is not successful."
    assert mseloss_x <= 0.7, "The denoising of fractional coordinates accuracy is not successful."
