# copyright 2024 Huawei Technologies co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
# See the License for the specific language governing permissions and
# limitations under the license.
##==========================================================================
"""
    南部非洲SAMTEX大地电磁数据反演，MT-VAE
"""
import time
import os
import platform
import math
import logging
import functools
import multiprocessing
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.io import savemat, loadmat
from scipy.sparse import csr_matrix
import help as hf
import library.MT2DFWD2 as MT
import library.Jacobians as jacos
from src import model

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO)


def plot_combined_code(v_true_array, addr):
    """

    :param v_true_array:
    :param addr:
    :return:
    """
    xs = np.linspace(0, xn_pr2 - xn_pr1, xn_pr2 - xn_pr1 + 1, dtype="uint8")
    zs = np.linspace(0, latent_dim, latent_dim + 1, dtype="uint8")
    [xss, zss] = np.meshgrid(xs, zs)
    v_true_array1 = np.squeeze(v_true_array)
    fig1 = plt.figure(figsize=(5, 5))
    fig1.add_subplot(1, 1, 1)
    plt.pcolor(
        xss,
        zss,
        np.reshape(v_true_array1, (latent_dim, xn_pr2 - xn_pr1), order="f"),
        cmap=plt.get_cmap("jet"),
    )
    plt.colorbar()
    plt.clim(-2.5, 2.5)
    plt.tight_layout()
    plt.savefig(addr)
    plt.close()


UID = "inversion"

SELECT_AS_INVERSIONMODE = 0  # 0: no multiprocessing, 1: use multiprocessing
SEL_ACTIFUNC = 1  # what activation function is used. 1 means swish
max_iter = 13

## Adjustable parameters, can be fine tuned around given value↓
lamda = 0.1e-3  # 介于0.02E-3和0.2E-3之间
lambda_decay = 0.9
alphahor = 6e-5
betahor = 1e-5
alphaver = 3e-5
betaver = 1e-12
## ↑

## the range of pixels to be encoded↓
xn_pr1 = 15
xn_pr2 = 55
zn_pr1 = 32
zn_pr2 = 64
## ↑

kl_weight = 10e-3  # 1e-2
ssim_w = 40e-2  # 1e-2
latent_dim = 16
bar_1 = np.log10(1)
bar_2 = 4.5
initial_res = np.log10(100)
cost_threshold = 0.005  #
delta_cost_threshold = 0.0005
model_sel_dim = 32

freq_n = 20

fre_sample = np.array(
    [0, 2, 4, 6, 8, 10, 13, 16, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 64],
    dtype=int,
)
dw_mask = np.ones((36, freq_n * 2))
# first twenty are for apparent resistivity and last twenty are for impedance phase↑

# design the mask of measured data to exclude the bad points
# No 3
valid_num_max = 44
mask_line = fre_sample < valid_num_max
dw_mask[3, freq_n:] = mask_line

# No 11
valid_num_max = 59
mask_line = fre_sample < valid_num_max
dw_mask[11, :freq_n] = mask_line
dw_mask[11, freq_n:] = mask_line
dw_mask[13, freq_n:] = mask_line
dw_mask[15, freq_n:] = mask_line
dw_mask[18, :freq_n] = mask_line
dw_mask[26, freq_n:] = mask_line
dw_mask[30, :freq_n] = mask_line
dw_mask[15, freq_n:] = mask_line


valid_num_max = 41
mask_line = fre_sample < valid_num_max
dw_mask[18, freq_n:] = mask_line

valid_num_max = 37
mask_line = fre_sample < valid_num_max
dw_mask[29, freq_n:] = mask_line

valid_num_max = 57
mask_line = fre_sample < valid_num_max
dw_mask[31, freq_n:] = mask_line

valid_num_max = 53
mask_line = fre_sample < valid_num_max
dw_mask[33, freq_n:] = mask_line

valid_num_max = 61
mask_line = fre_sample < valid_num_max
dw_mask[34, freq_n:] = mask_line

dw_mask[19, freq_n:] = dw_mask[19, freq_n:] * 0

valid_num_max = 43
mask_line = fre_sample < valid_num_max
dw_mask[30, freq_n:] = mask_line

depth_max = 80

MYDATA = loadmat(os.path.join("dataset", "measured_data", "ETO_raw.mat"))
freq_total = MYDATA["FREQ"]
freq_total = np.squeeze(freq_total)
data_total = MYDATA["DATA"]
data_total_use = data_total[:, fre_sample, :]
offset_total = MYDATA["OFFSET"]
offset_total = np.squeeze(offset_total)

MYDATA1 = loadmat(os.path.join("dataset", "measured_data", "KIM_raw.mat"))

data_total1 = MYDATA1["DATA"]
data_total_use1 = data_total1[-10:, fre_sample, :]
offset_total1 = MYDATA1["OFFSET"]
offset_total1 = np.squeeze(offset_total1)
ETO_KIM_OFFSET = np.zeros(36)
ETO_KIM_OFFSET[:10] = offset_total1[-10:]
ETO_KIM_OFFSET[10:] = offset_total
ETO_KIM_OFFSET[0] = 0
ETO_KIM_OFFSET[10] = 35.5
line_coor = np.cumsum(ETO_KIM_OFFSET) * 1e3
data_total_useb = np.concatenate((data_total_use1, data_total_use), axis=0)

# Simulation Setup
fieldxstart = -line_coor[-1] / 20
fieldxend = line_coor[-1] + line_coor[-1] / 20
xnumbermt = 100
xedgelocationmt = np.linspace(fieldxstart, fieldxend, xnumbermt + 1)
xelementlocationmt = 0.5 * (xedgelocationmt[0:-1] + xedgelocationmt[1:])
znumbermt = 64
dh = np.zeros(znumbermt)
for i in range(znumbermt):
    dh[i] = 2 * math.pow(1.146, i)  # 80 km
zedgelocationmt = np.concatenate(([0], np.cumsum(dh)))
zelementlocationmt = 0.5 * (zedgelocationmt[0:-1] + zedgelocationmt[1:])
timestamp2 = time.time()

XNUMBERMT1 = 100
xedgelocationmt1 = np.linspace(fieldxstart, fieldxend, XNUMBERMT1 + 1)
interpxlocations = 0.5 * (xedgelocationmt1[0:-1] + xedgelocationmt1[1:])
ZNUMBERMT1 = 64
dh1 = np.zeros(ZNUMBERMT1)
for i in range(ZNUMBERMT1):
    dh1[i] = 2 * math.pow(1.146, i)  # 1 km
zedgelocationmt1 = np.concatenate(([0], np.cumsum(dh1)))
interpdepths = 0.5 * (zedgelocationmt1[0:-1] + zedgelocationmt1[1:])
rxnumbermt = len(line_coor)

freqnumbermt = len(fre_sample)
frequencymt = freq_total[fre_sample]
rxindexmt = np.zeros(rxnumbermt)
for ii in range(rxnumbermt):
    rxindexmt[ii] = np.argmin(abs(line_coor[ii] - xelementlocationmt))
rxindexmt = np.array(rxindexmt, dtype="uint8")
RXMT = [0, 10000]

sys = platform.system()
if sys == "Windows":
    logging.info("OS is Windows.")
    dir_name = os.path.join("results", "ckpt", "net")
elif sys == "Linux":
    logging.info("OS is Linux.")
    dir_name = os.path.join("results", "ckpt", "net")

vae = model.Model()
checkpoint_path = os.path.join(dir_name, "VAE.ckpt")
params = load_checkpoint(checkpoint_path)
load_param_into_net(vae, params)
decoder = vae.decoder
# 注意这个同时输出mean和var
meanmodel = vae.mean_model
logging.info("load pretrained VAE")

logging.info("Inversion with Gauss-Newton")
if not os.path.exists(os.path.join("results", "inversion", UID)):
    os.makedirs(os.path.join("results", "inversion", UID))
    logging.info("UID directory does not exist! create it.")

[VG, HG] = hf.ComputeGradient(XNUMBERMT1, ZNUMBERMT1)
[VGP, HGP] = hf.ComputeGradient(xn_pr2 - xn_pr1, zn_pr2 - zn_pr1)

logrho_f = np.zeros(len(rxindexmt) * 2 * len(frequencymt))
fielddata = np.zeros(2 * freqnumbermt * rxnumbermt)
data_app = data_total_useb[:, :, 0]
data_pha = data_total_useb[:, :, 1]
data_app = np.reshape(data_app, -1, order="f")
data_pha = np.reshape(data_pha, -1, order="f")

# Load corrected static shift data
teapp_a_st = loadmat(os.path.join("dataset", "measured_data", "ETOKIM_corrected.mat"))
data_app = teapp_a_st["TE_app_new"]
data_app = data_app[:, fre_sample]
data_app = np.reshape(data_app, -1, order="f")
#
fielddata[: freqnumbermt * rxnumbermt] = data_app
fielddata[freqnumbermt * rxnumbermt :] = data_pha

logrho_f[: freqnumbermt * rxnumbermt] = np.log10(fielddata[: rxnumbermt * freqnumbermt])
logrho_f[rxnumbermt * freqnumbermt :] = fielddata[rxnumbermt * freqnumbermt :]


[ia_temp, ja, value, ub, area, index1, Z] = MT.MT2SparseEquationSetUp_zhhy(
    interpxlocations, interpdepths
)

MT2DFWD2_PACKET = {
    "freq": frequencymt,
    "Rx": RXMT,
    "Field_grid_x": interpxlocations,
    "Field_grid_z": interpdepths,
    "X_number": XNUMBERMT1,
    "Z_number": ZNUMBERMT1,
    "Rx_index": rxindexmt,
}

if SELECT_AS_INVERSIONMODE:
    pool = multiprocessing.Pool(8)

tmstp4a = time.time()
MT2DFWD2_PACKET["ia_temp"] = ia_temp
MT2DFWD2_PACKET["ja"] = ja
MT2DFWD2_PACKET["value"] = value
MT2DFWD2_PACKET["Ub"] = ub
MT2DFWD2_PACKET["Area"] = area
MT2DFWD2_PACKET["index1"] = index1
MT2DFWD2_PACKET["Z"] = Z

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
[xss1, zss1] = np.meshgrid(interpxlocations[rxindexmt], np.log10(frequencymt))
plt.pcolor(
    xss1,
    zss1,
    np.reshape(
        logrho_f[: freqnumbermt * rxnumbermt], (freqnumbermt, rxnumbermt), order="c"
    ),
    cmap=plt.get_cmap("jet"),
)
cbar = plt.colorbar()
plt.clim(
    np.min(logrho_f[: freqnumbermt * rxnumbermt]),
    np.max(logrho_f[: freqnumbermt * rxnumbermt]),
)
plt.tight_layout()
plt.savefig(os.path.join("results", "inversion", UID, "data_true.png"))
plt.close()
#

v_array = np.zeros((1, latent_dim * (xn_pr2 - xn_pr1)))
rho_recon = ms.Tensor(
    initial_res * np.ones(zn_pr2 - zn_pr1, dtype=np.float32)
).asnumpy()
# %%
v, _ = meanmodel(ms.Tensor(rho_recon.reshape([1, 1, zn_pr2 - zn_pr1], order="F")))
v_np = v.numpy()

for jj in range(xn_pr2 - xn_pr1):
    v_array[:, jj * latent_dim : (jj + 1) * latent_dim] = v_np

plot_combined_code(v_array, os.path.join("results", "inversion", UID, "code_start.png"))

## Initial model is uniform half space
rho_recon_pred_ii = decoder(v)  # tensor
rho_recon_pred = np.zeros((ZNUMBERMT1, XNUMBERMT1)) + initial_res
for jj in range(XNUMBERMT1):
    if xn_pr2 > jj >= xn_pr1:
        rho_recon_pred[zn_pr1:zn_pr2, jj] = np.reshape(rho_recon_pred_ii, -1)
logging.info("initial rho %.6f", rho_recon_pred.max())
coor = {
    "zelementlocation": interpdepths,
    "xelementlocation": interpxlocations,
    "xx": rho_recon_pred,
    "colorbaraxis": [bar_1, bar_2],
    "surf_flag": 1,
    "address": os.path.join("results", "inversion", UID, "model_start.png"),
}
hf.Plot2DImage(
    coor,
    rangex=[fieldxstart / 1e3, fieldxend / 1e3],
    rangez=[-80, 0],
    iflog=1,
    use_cmap="jet_r",
)
cost = [1]
MT2DFWD2_PACKET["Field_rho"] = rho_recon_pred
if SELECT_AS_INVERSIONMODE:
    MT2DFWD2_BACK = pool.map(
        functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_PACKET), range(len(frequencymt))
    )
else:
    MT2DFWD2_BACK = []
    for ii in range(len(frequencymt)):
        bci = MT.MT2DFWD2_zhhy(MT2DFWD2_PACKET, ii)
        MT2DFWD2_BACK.append(bci)
newdata = np.zeros(2 * len(rxindexmt) * len(frequencymt))
eobsvector = np.zeros(len(rxindexmt) * len(frequencymt), dtype="complex")
hobsvector = np.zeros(len(rxindexmt) * len(frequencymt), dtype="complex")
efieldvectorf = np.zeros((xnumbermt * znumbermt, len(frequencymt)), dtype="complex")
for i in range(len(frequencymt)):
    newdata[i * len(rxindexmt) : (i + 1) * len(rxindexmt)] = MT2DFWD2_BACK[i]["data_f"][
        : len(rxindexmt)
    ]
    newdata[
        len(rxindexmt) * len(frequencymt)
        + i * len(rxindexmt) : len(rxindexmt) * len(frequencymt)
        + (i + 1) * len(rxindexmt)
    ] = MT2DFWD2_BACK[i]["data_f"][len(rxindexmt) :]
    eobsvector[i * len(rxindexmt) : (i + 1) * len(rxindexmt)] = MT2DFWD2_BACK[i][
        "Eobs_in"
    ]
    hobsvector[i * len(rxindexmt) : (i + 1) * len(rxindexmt)] = MT2DFWD2_BACK[i][
        "Hobs_in"
    ]
    efieldvectorf[:, i] = MT2DFWD2_BACK[i]["EFieldVector_in"]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
[xss1, zss1] = np.meshgrid(interpxlocations[rxindexmt], np.log10(frequencymt))
plt.pcolor(
    xss1,
    zss1,
    np.reshape(
        newdata[: rxnumbermt * freqnumbermt], (freqnumbermt, rxnumbermt), order="c"
    ),
    cmap=plt.get_cmap("jet"),
)
cbar = plt.colorbar()
plt.clim(np.min(logrho_f), np.max(logrho_f))
plt.tight_layout()
plt.savefig(os.path.join("results", "inversion", UID, "data_start.png"))
plt.close()

valua = np.ones(freqnumbermt * rxnumbermt)
valus = np.zeros(2 * freqnumbermt * rxnumbermt)

##Assign zero weight to bad points
valus[: freqnumbermt * rxnumbermt] = np.reshape(dw_mask[:, :freq_n], -1, order="f")
valus[freqnumbermt * rxnumbermt :] = np.reshape(dw_mask[:, freq_n:], -1, order="f")
#
isss = np.array(
    np.linspace(0, freqnumbermt * rxnumbermt * 2 - 1, freqnumbermt * rxnumbermt * 2),
    dtype="int32",
)
js = np.array(
    np.linspace(0, freqnumbermt * rxnumbermt * 2 - 1, freqnumbermt * rxnumbermt * 2),
    dtype="int32",
)
dataweight = csr_matrix(
    (valus, (isss, js)),
    shape=(freqnumbermt * rxnumbermt * 2, freqnumbermt * rxnumbermt * 2),
)
res = logrho_f - newdata
try:
    ck = (
        norm(np.dot(dataweight.toarray(), res)) ** 1
        / norm(np.dot(dataweight.toarray(), logrho_f)) ** 1
    )
except ZeroDivisionError as e:
    logging.error("Error %s", e)
    raise

cost[0] = ck

v_ref = np.zeros((1, latent_dim * (xn_pr2 - xn_pr1)))
isss = np.array(
    np.linspace(0, 2 * freqnumbermt * rxnumbermt - 1, 2 * freqnumbermt * rxnumbermt),
    dtype="int32",
)
js = np.array(
    np.linspace(0, 2 * freqnumbermt * rxnumbermt - 1, 2 * freqnumbermt * rxnumbermt),
    dtype="int32",
)
valub = np.zeros(2 * freqnumbermt * rxnumbermt)

for jj in range(max_iter * 2):
    [jacobianMTRho, jacobianMTPhi] = jacos.ComputeJacobianFunc_z(
        eobsvector,
        hobsvector,
        efieldvectorf,
        frequencymt,
        np.reshape(rho_recon_pred, (ZNUMBERMT1, XNUMBERMT1), order="f"),
        RXMT,
        interpxlocations,
        interpdepths,
        XNUMBERMT1,
        ZNUMBERMT1,
        rxindexmt,
        ia_temp,
        ja,
        value,
        area,
        index1,
    )
    J = np.concatenate((jacobianMTRho, jacobianMTPhi), axis=0)  # [Ndata, Nmodel]

    lamda = lamda * lambda_decay
    logging.info("==================================")
    if np.mod(jj, 2) == 0:
        logging.info("Iter = %d, renew pixel part", np.floor(jj / 2))
        try:
            gk = (
                -np.matrix(J).H.dot(res)
                / norm(np.dot(dataweight.toarray(), logrho_f)) ** 2
            )
            Hk = (
                np.matrix(J).H.dot(J)
                / norm(np.dot(dataweight.toarray(), logrho_f)) ** 2
            )  ##
        except ZeroDivisionError as e:
            logging.error("Error %s", e)
            raise

        rho_recon_pred_array = np.reshape(rho_recon_pred, -1, order="f")
        hG_m = HG * rho_recon_pred_array
        vG_m = VG * rho_recon_pred_array
        gk = (
            gk
            + alphahor * ck * np.dot(np.matrix(HG.todense()).H, hG_m)
            + alphaver * ck * np.dot(np.matrix(VG.todense()).H, vG_m)
        )
        Hk = (
            Hk
            + alphahor * ck * np.dot(np.matrix(HG.todense()).H, HG.todense())
            + alphaver * ck * np.dot(np.matrix(VG.todense()).H, VG.todense())
        )
        time1 = time.time()
        pk = -np.linalg.solve(Hk, gk.T)
        time2 = time.time()
        logging.info("Computing pk time consumption：%.4f s", time2 - time1)
        a = 1
        rho_recon_pred_array_1 = rho_recon_pred_array + a * np.transpose(np.array(pk))
        rho_recon_pred_1 = np.reshape(
            rho_recon_pred_array_1, (ZNUMBERMT1, XNUMBERMT1), order="f"
        )
    else:
        logging.info("Iter = %d, renew code part", np.floor(jj / 2))
        v_array = np.zeros((1, (xn_pr2 - xn_pr1) * latent_dim))
        for yy in range(xn_pr2 - xn_pr1):
            cur_m = rho_recon_pred[zn_pr1:zn_pr2, yy + xn_pr1]
            v, _ = meanmodel(
                ms.Tensor(
                    cur_m.reshape([1, 1, zn_pr2 - zn_pr1], order="F"), dtype=ms.float32
                )
            )
            v_np = v.numpy()
            v_array[:, yy * latent_dim : (yy + 1) * latent_dim] = v_np
        gk = np.zeros((xn_pr2 - xn_pr1) * latent_dim)
        Hk = np.eye((xn_pr2 - xn_pr1) * latent_dim) * ck * lamda
        J_slice = np.zeros(
            (2 * freqnumbermt * rxnumbermt, (zn_pr2 - zn_pr1) * (xn_pr2 - xn_pr1))
        )
        for uu in range(xn_pr2 - xn_pr1):
            J_slice[:, (zn_pr2 - zn_pr1) * uu : (zn_pr2 - zn_pr1) * (uu + 1)] = J[
                :,
                (xn_pr1 + uu) * ZNUMBERMT1
                + zn_pr1 : (xn_pr1 + uu) * ZNUMBERMT1
                + zn_pr2,
            ]  # [Ndata, Nmodel]
        JD = np.zeros(
            ((zn_pr2 - zn_pr1) * (xn_pr2 - xn_pr1), latent_dim * (xn_pr2 - xn_pr1))
        )  # part part2model to v

        for uu in range(xn_pr2 - xn_pr1):
            v = v_array[:, uu * latent_dim : (uu + 1) * latent_dim]
            v = ms.Tensor(v, ms.float32)  # (1,16)
            rho_recon_pred_ii = decoder(v)
            v_broadcast = ms.ops.BroadcastTo(
                ((1, rho_recon_pred_ii.shape[-1], v.shape[-1]))
            )(v)
            jacb = ms.ops.Squeeze()(ms.ops.grad(decoder)(v_broadcast))
            jacb1 = np.reshape(
                jacb, [zn_pr2 - zn_pr1, latent_dim], order="F"
            )  # [Nmodel, N_latent_z]
            JD[
                uu * (zn_pr2 - zn_pr1) : (uu + 1) * (zn_pr2 - zn_pr1),
                uu * latent_dim : (uu + 1) * latent_dim,
            ] = jacb1

        gk += lamda * ck * np.squeeze(np.array(v_array))  #
        J2_full = np.matmul(J_slice, JD)  # part d 2 part v (full)
        try:
            gk = (
                gk
                - np.matrix(J2_full).H.dot(res)
                / norm(np.dot(dataweight.toarray(), logrho_f)) ** 2
            )
            Hk += (
                np.matrix(J2_full).H.dot(J2_full)
                / norm(np.dot(dataweight.toarray(), logrho_f)) ** 2
            )  ##
        except ZeroDivisionError as e:
            logging.error("Error %s", e)
            raise

        rho_recon_pred_array_p = np.reshape(
            rho_recon_pred[zn_pr1:zn_pr2, xn_pr1:xn_pr2], -1, order="f"
        )
        hGp_JD = HGP * JD
        hGp_m = HGP * rho_recon_pred_array_p
        vGp_JD = VGP * JD
        vGp_m = VGP * rho_recon_pred_array_p
        gk = (
            gk
            + betahor * ck * np.dot(np.matrix(hGp_JD).H, hGp_m)
            + betaver * ck * np.dot(np.matrix(vGp_JD).H, vGp_m)
        )
        Hk = (
            Hk
            + betahor * ck * np.dot(np.matrix(hGp_JD).H, hGp_JD)
            + betaver * ck * np.dot(np.matrix(vGp_JD).H, vGp_JD)
        )

        time1 = time.time()
        pk = -np.linalg.solve(Hk, gk.T)
        time2 = time.time()
        logging.info("Computing pk time consumption：%.4f s", time2 - time1)
        a = 1
        v_array_1 = v_array + a * np.transpose(np.real(pk))

        for hh in range(xn_pr2 - xn_pr1):
            v1 = ms.Tensor(
                v_array_1[:, hh * latent_dim : (hh + 1) * latent_dim], dtype=ms.float32
            )
            rho_recon_pred_ii = decoder(v1).numpy()
            rho_recon_pred_1 = rho_recon_pred.copy()
            rho_recon_pred_1[zn_pr1:zn_pr2, hh + xn_pr1] = np.squeeze(rho_recon_pred_ii)
    MT2DFWD2_PACKET["Field_rho"] = rho_recon_pred_1
    if SELECT_AS_INVERSIONMODE:
        MT2DFWD2_BACK = pool.map(
            functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_PACKET),
            range(len(frequencymt)),
        )
    else:
        MT2DFWD2_BACK = []
        for ii in range(len(frequencymt)):
            bci = MT.MT2DFWD2_zhhy(MT2DFWD2_PACKET, ii)
            MT2DFWD2_BACK.append(bci)
    newdata = np.zeros(2 * len(rxindexmt) * len(frequencymt))
    efieldvectorf = np.zeros(
        (XNUMBERMT1 * ZNUMBERMT1, len(frequencymt)), dtype="complex"
    )
    eobsvector = np.zeros(len(rxindexmt) * len(frequencymt), dtype="complex")
    hobsvector = np.zeros(len(rxindexmt) * len(frequencymt), dtype="complex")
    for i in range(len(frequencymt)):
        newdata[i * len(rxindexmt) : (i + 1) * len(rxindexmt)] = MT2DFWD2_BACK[i][
            "data_f"
        ][: len(rxindexmt)]
        newdata[
            len(rxindexmt) * len(frequencymt)
            + i * len(rxindexmt) : len(rxindexmt) * len(frequencymt)
            + (i + 1) * len(rxindexmt)
        ] = MT2DFWD2_BACK[i]["data_f"][len(rxindexmt) :]
        eobsvector[i * len(rxindexmt) : (i + 1) * len(rxindexmt)] = MT2DFWD2_BACK[i][
            "Eobs_in"
        ]
        hobsvector[i * len(rxindexmt) : (i + 1) * len(rxindexmt)] = MT2DFWD2_BACK[i][
            "Hobs_in"
        ]
        efieldvectorf[:, i] = MT2DFWD2_BACK[i]["EFieldVector_in"]
    res = logrho_f - newdata
    try:
        ck = (
            norm(np.dot(dataweight.toarray(), res)) ** 1
            / norm(np.dot(dataweight.toarray(), logrho_f)) ** 1
        )
    except ZeroDivisionError as e:
        logging.error("Error %s", e)
        raise

    #
    if ck < cost[-1]:
        if np.mod(jj, 2) == 0:
            rho_recon_pred = rho_recon_pred_1
        else:
            v_array = v_array_1

    ls_num = 1
    while ck > cost[-1] and ls_num < 6:
        try:
            a1 = -0.5 * a ** 2 * (np.dot(gk, pk)) / (ck - cost[-1] - a * np.dot(gk, pk))
        except ZeroDivisionError as e:
            logging.error("Error %s", e)
            raise
        a1 = a1[0, 0]
        if a1 < 0.01 * a:
            a1 = 0.01 * a
        a = a1
        if np.mod(jj, 2) == 0:
            logging.info(
                "ite=%d, line search, renew pixel, a=%.6f", int(np.floor(jj / 2)), a
            )
            rho_recon_pred_propose_1 = np.reshape(
                rho_recon_pred, -1, order="f"
            ) + a * np.transpose(np.array(pk))
            rho_recon_pred_propose = np.reshape(
                rho_recon_pred_propose_1, (ZNUMBERMT1, XNUMBERMT1), order="f"
            )
        else:
            logging.info(
                "ite=%d, line search, renew code, a=%.6f", int(np.floor(jj / 2)), a
            )
            v1_propose = v_array + a * np.transpose(np.real(pk))
            rho_recon_pred_propose = rho_recon_pred.copy()
            logging.info("rho down before %.6f", rho_recon_pred_propose.max())
            for hh in range(xn_pr2 - xn_pr1):
                v1 = ms.Tensor(
                    v1_propose[:, hh * latent_dim : (hh + 1) * latent_dim],
                    dtype=ms.float32,
                )
                rho_recon_pred_ii = decoder(v1).numpy()
                rho_recon_pred_propose[zn_pr1:zn_pr2, hh + xn_pr1] = np.squeeze(
                    rho_recon_pred_ii
                )
        MT2DFWD2_PACKET["Field_rho"] = rho_recon_pred_propose
        if SELECT_AS_INVERSIONMODE:
            MT2DFWD2_BACK = pool.map(
                functools.partial(MT.MT2DFWD2_zhhy, MT2DFWD2_PACKET),
                range(len(frequencymt)),
            )
        else:
            MT2DFWD2_BACK = []
            for ii in range(len(frequencymt)):
                bci = MT.MT2DFWD2_zhhy(MT2DFWD2_PACKET, ii)
                MT2DFWD2_BACK.append(bci)
        newdata_propose = np.zeros(2 * len(rxindexmt) * len(frequencymt))
        efieldvectorf = np.zeros(
            (XNUMBERMT1 * ZNUMBERMT1, len(frequencymt)), dtype="complex"
        )
        eobsvector = np.zeros(len(rxindexmt) * len(frequencymt), dtype="complex")
        hobsvector = np.zeros(len(rxindexmt) * len(frequencymt), dtype="complex")
        for i in range(len(frequencymt)):
            newdata_propose[
                i * len(rxindexmt) : (i + 1) * len(rxindexmt)
            ] = MT2DFWD2_BACK[i]["data_f"][: len(rxindexmt)]
            newdata_propose[
                len(rxindexmt) * len(frequencymt)
                + i * len(rxindexmt) : len(rxindexmt) * len(frequencymt)
                + (i + 1) * len(rxindexmt)
            ] = MT2DFWD2_BACK[i]["data_f"][len(rxindexmt) :]
            eobsvector[i * len(rxindexmt) : (i + 1) * len(rxindexmt)] = MT2DFWD2_BACK[
                i
            ]["Eobs_in"]
            hobsvector[i * len(rxindexmt) : (i + 1) * len(rxindexmt)] = MT2DFWD2_BACK[
                i
            ]["Hobs_in"]
            efieldvectorf[:, i] = MT2DFWD2_BACK[i]["EFieldVector_in"]
        res_propose = logrho_f - newdata_propose
        try:
            ck_propose = (
                norm(np.dot(dataweight.toarray(), res_propose)) ** 1
                / norm(np.dot(dataweight.toarray(), logrho_f)) ** 1
            )
            ck = ck_propose
        except ZeroDivisionError as e:
            logging.error("Error %s", e)
            raise
        ls_num = ls_num + 1

        if ck_propose < cost[-1] or ls_num == 6:
            newdata = newdata_propose
            res = res_propose
            ck = ck_propose
            rho_recon_pred = rho_recon_pred_propose
            break

    logging.info("Iter # %d, C = %.6f", int(np.floor(jj / 2)), ck)
    cost.append(ck)
    logging.info("Relative data misfit = %s", cost)

    if np.mod(jj, 2):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        plt.pcolor(
            xss1,
            zss1,
            np.reshape(
                newdata[: freqnumbermt * rxnumbermt],
                (freqnumbermt, rxnumbermt),
                order="c",
            ),
            cmap=plt.get_cmap("jet"),
        )
        cbar = plt.colorbar()
        plt.clim(
            np.min(logrho_f[: freqnumbermt * rxnumbermt]),
            np.max(logrho_f[: freqnumbermt * rxnumbermt]),
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "results",
                "inversion",
                UID,
                "data_ite_" + str(int(np.floor(jj / 2))) + ".png",
            )
        )
        plt.close()
        plot_combined_code(
            np.array(v_array),
            os.path.join(
                "results",
                "inversion",
                UID,
                "code_ite_" + str(int(np.floor(jj / 2))) + ".png",
            ),
        )
        coor = {
            "zelementlocation": interpdepths,
            "xelementlocation": interpxlocations,
            "xx": rho_recon_pred,
            "colorbaraxis": [bar_1, bar_2],
            "surf_flag": 1,
            "address": os.path.join(
                "results",
                "inversion",
                UID,
                "model_ite_" + str(int(np.floor(jj / 2))) + ".png",
            ),
        }
        hf.Plot2DImage(
            coor,
            rangex=[fieldxstart / 1e3, fieldxend / 1e3],
            rangez=[-80, 0],
            iflog=1,
            use_cmap="jet_r",
        )
        savemat(
            os.path.join(
                "results",
                "inversion",
                UID,
                "No." + str(int(np.floor(jj / 2))) + " Resistivity Model.mat",
            ),
            {"model": rho_recon_pred},
        )
    if cost[-1] < cost_threshold and cost[-2] - cost[-1] < delta_cost_threshold:
        break

cost_array = np.array(cost, dtype="float32")
np.savetxt(os.path.join("results", "inversion", UID, "cost.txt"), cost_array, "%.5f")
savemat(
    os.path.join("results", "inversion", UID, "final_model.mat"),
    {"xs": interpxlocations, "zs": interpdepths, "value": rho_recon_pred},
)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
[xss1, zss1] = np.meshgrid(xedgelocationmt[rxindexmt], np.log10(frequencymt))
plt.pcolor(
    xss1,
    zss1,
    np.reshape(
        newdata[: freqnumbermt * rxnumbermt], (freqnumbermt, rxnumbermt), order="c"
    ),
    cmap=plt.get_cmap("jet"),
)
cbar = plt.colorbar()
plt.clim(
    min(logrho_f[: rxnumbermt * freqnumbermt]),
    max(logrho_f[: rxnumbermt * freqnumbermt]),
)
plt.tight_layout()
plt.savefig(os.path.join("results", "inversion", UID, "app_final.png"))
plt.close()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
[xss1, zss1] = np.meshgrid(xedgelocationmt[rxindexmt], np.log10(frequencymt))
plt.pcolor(
    xss1,
    zss1,
    np.reshape(
        newdata[freqnumbermt * rxnumbermt :], (freqnumbermt, rxnumbermt), order="c"
    ),
    cmap=plt.get_cmap("jet"),
)
cbar = plt.colorbar()
plt.clim(
    min(logrho_f[rxnumbermt * freqnumbermt :]),
    max(logrho_f[rxnumbermt * freqnumbermt :]),
)
plt.tight_layout()
plt.savefig(os.path.join("results", "inversion", UID, "pha_final.png"))
plt.close()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
[xss1, zss1] = np.meshgrid(xedgelocationmt[rxindexmt], np.log10(frequencymt))
plt.pcolor(
    xss1,
    zss1,
    np.reshape(
        newdata[: rxnumbermt * freqnumbermt] - logrho_f[: rxnumbermt * freqnumbermt],
        (freqnumbermt, rxnumbermt),
        order="c",
    ),
    cmap=plt.get_cmap("jet"),
)
cbar = plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join("results", "inversion", UID, "app_delta.png"))
plt.close()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
[xss1, zss1] = np.meshgrid(xedgelocationmt[rxindexmt], np.log10(frequencymt))
plt.pcolor(
    xss1,
    zss1,
    np.reshape(
        newdata[rxnumbermt * freqnumbermt :] - logrho_f[rxnumbermt * freqnumbermt :],
        (freqnumbermt, rxnumbermt),
        order="c",
    ),
    cmap=plt.get_cmap("jet"),
)
cbar = plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join("results", "inversion", UID, "pha_delta.png"))
plt.close()
logging.info("ALL DONE")
