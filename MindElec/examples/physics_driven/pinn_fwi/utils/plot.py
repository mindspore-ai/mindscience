# Copyright 2021 Huawei Technologies Co., Ltd
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
"""plot"""

import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

with open('src/default_config.yaml', 'r') as y:
    cfg = yaml.full_load(y)


def plot_ini_total_disp_spec_sumevents(xx, zz, u_ini1x, u_ini1z, t01):
    """plot initial total displacement field simulated by specfem2D"""

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'],
                 np.sqrt(u_ini1x ** 2 + u_ini1z ** 2).reshape(xx.shape), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Scaled I.C total disp. input specfem t=' + str(t01))
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Ini_total_disp_spec_sumEvents.png', dpi=400)
    plt.close(fig)


def plot_sec_wavefield_input_spec_sumevents(xx, zz, u_ini2x, u_ini2z, t02):
    """plot second total displacement field simulated by specfem2D"""

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'],
                 np.sqrt(u_ini2x ** 2 + u_ini2z ** 2).reshape(xx.shape), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Scaled sec I.C total disp. input specfem t=' + str(round(t02, 4)))
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('sec_wavefield_input_spec_sumEvents.png', dpi=400)
    plt.close(fig)


def plot_total_disp_spec_testdata_sumevents(xx, zz, u_specx, u_specz, t_la, t01):
    """plot test total displacement field simulated by specfem2D"""

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'],
                 np.sqrt(u_specx ** 2 + u_specz ** 2).reshape(xx.shape), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Test data: Total displacement specfem t=' +
              str(round((t_la - t01), 4)))
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('total_disp_spec_testData_sumEvents.png', dpi=400)
    plt.close(fig)


def plot_true_wavespeed(xx, zz, alpha_true0, x_s):
    """plot true acoustic wavespeed"""

    fig = plt.figure()
    plt.contourf(cfg['Lx'] * xx, cfg['Lz'] * zz,
                 alpha_true0.reshape((xx.shape)), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'True acoustic wavespeed ($\alpha$)')
    plt.colorbar()
    plt.axis('scaled')
    plt.plot(cfg['Lx'] * 0.99 * x_s[:, 0], cfg['Lz'] * x_s[:, 1], 'r*', markersize=5)
    plt.savefig('True_wavespeed.png', dpi=400)
    plt.close(fig)


def plot_ini_guess_wavespeed(xx, zz, alpha_plot):
    """plot initial guessed  wavespeed"""

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'],
                 alpha_plot.reshape((xx.shape)), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'Initial guess ($\alpha$)')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Ini_guess_wavespeed.png', dpi=400)
    plt.close(fig)


def plot_total_predicted_dispfield_and_diff(xx, zz, ux01, uz01, ux02, uz02, uxt, uzt, u_specx, u_specz, t01, t02, t_la):
    """plot test total displacement field obtained through pinn method
     and the gap between pinn's and specfem2D's result
    """

    u_pinn01 = ((ux01.reshape(xx.shape)) ** 2 + (uz01.reshape(xx.shape)) ** 2) ** 0.5
    u_pinn02 = ((ux02.reshape(xx.shape)) ** 2 + (uz02.reshape(xx.shape)) ** 2) ** 0.5
    u_pinnt = ((uxt.reshape(xx.shape)) ** 2 + (uzt.reshape(xx.shape)) ** 2) ** 0.5
    u_diff = np.sqrt(u_specx ** 2 + u_specz ** 2).reshape(xx.shape) - u_pinnt
    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'], u_pinn01, 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'PINNs $U(x,z,t=$' + str(0) + r'$)$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Total_Predicted_dispfield_t=' + str(0) + '.png', dpi=400)
    plt.close(fig)

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'], u_pinn02, 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'PINNs $U(x,z,t=$' + str(round(t02 - t01, 4)) + r'$)$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Total_Predicted_dispfield_t=' +
                str(round(t02 - t01, 4)) + '.png', dpi=400)
    plt.close(fig)

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'], u_pinnt, 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'PINNs $U(x,z,t=$' + str(round((t_la - t01), 4)) + r'$)$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('Total_Predicted_dispfield_t=' +
                str(round((t_la - t01), 4)) + '.png', dpi=400)
    plt.close(fig)

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'], u_diff, 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'Total disp. Specfem-PINNs ($t=$' +
              str(round((t_la - t01), 4)) + r'$)$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('pointwise_Error_spec_minus_PINNs_t=' +
                str(round((t_la - t01), 4)) + '.png', dpi=400)
    plt.close(fig)


def plot_inverted_alpha(xx, zz, alpha0, alpha_true0):
    """plot inverted  wavespeed through pinn method"""

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'],
                 alpha0.reshape(xx.shape), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r'Inverted $\alpha$')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('inverted_alpha.png', dpi=400)
    plt.close(fig)

    fig = plt.figure()
    plt.contourf(xx * cfg['Lx'], zz * cfg['Lz'], alpha_true0 -
                 (alpha0.reshape(xx.shape)), 100, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(r' $\alpha$ misfit (true-inverted)')
    plt.colorbar()
    plt.axis('scaled')
    plt.savefig('alpha_misfit.png', dpi=400)
    plt.close(fig)


def plot_misfit(loss_rec):
    """plot the training misfits"""

    fig = plt.figure()
    plt.plot(loss_rec[0:, 0], loss_rec[0:, 4], 'g', label='ini_disp2')
    plt.plot(loss_rec[0:, 0], loss_rec[0:, 6], 'black', label='B.C')
    plt.plot(loss_rec[0:, 0], loss_rec[0:, 1], '--y', label='Total')
    plt.plot(loss_rec[0:, 0], loss_rec[0:, 2], 'r', label='PDE')
    plt.plot(loss_rec[0:, 0], loss_rec[0:, 3], 'b', label='ini_disp1')
    plt.plot(loss_rec[0:, 0], loss_rec[0:, 5], 'c', label='Seism')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('misfit')
    plt.legend()
    plt.savefig('misfit.png', dpi=400)
    plt.close(fig)


def plot_seismogram(x_s, s_z, s_x, uz_seism_pred, ux_seism_pred, az, d_s):
    """plot the seismogram"""

    fig = plt.figure()
    plt.plot(x_s[600:750, 2], s_z[600:750], 'ok', mfc='none', label='Input')
    plt.plot(x_s[600:750, 2], uz_seism_pred[600:750], 'r', label='PINNs')
    plt.legend()
    plt.title(r' Vertical Seismogram z=' + str(round(az - d_s, 4)))
    plt.savefig('ZSeismograms_compare_z=' +
                str(round(az - d_s, 4)) + '.png', dpi=400)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(x_s[600:750, 2], s_x[600:750], 'ok', mfc='none', label='Input')
    plt.plot(x_s[600:750, 2], ux_seism_pred[600:750], 'r', label='PINNs')
    plt.legend()
    plt.title(r' Horizontal Seismogram z=' + str(round(az - d_s, 4)))
    plt.savefig('XSeismograms_compare_z=' +
                str(round(az - d_s, 4)) + '.png', dpi=400)
    plt.close(fig)


def plot_wave_pnential(xx, zz, ux01, uz01, u_ini1x, u_ini1z, ux02, uz02, u_ini2x, u_ini2z, uxt, uzt, u_specx, u_specz):
    """plot the Ground Truth  and PINN's Prediction"""

    l_x = cfg['Lx']
    l_z = cfg['Lz']

    u_pinn01 = ((ux01.reshape(xx.shape)) ** 2 + (uz01.reshape(xx.shape)) ** 2) ** 0.5
    u_diff01 = np.sqrt(u_ini1x ** 2 + u_ini1z ** 2).reshape(xx.shape) - u_pinn01

    mi = np.min(np.sqrt(u_ini1x ** 2 + u_ini1z ** 2))
    ma = np.max(np.sqrt(u_ini1x ** 2 + u_ini1z ** 2))
    print(mi, ma)
    norm1 = matplotlib.colors.Normalize(vmin=mi, vmax=ma)
    norm2 = matplotlib.colors.Normalize(vmin=0, vmax=0.15)

    # plots of inputs for sum of the events
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    ax = ax.flatten()
    im1 = ax[0].contourf(xx * l_x, zz * l_z,
                         np.sqrt(u_ini1x ** 2 + u_ini1z **
                                 2).reshape(xx.shape), 100,
                         norm=norm1, cmap='seismic')
    ax[0].set_title('Ground Truth')
    ax[1].contourf(xx * l_x, zz * l_z, u_pinn01, 100, norm=norm1, cmap='seismic')
    ax[1].set_title("PINN's Prediction")
    ax[2].contourf(xx * l_x, zz * l_z, u_diff01, 100, norm=norm2, cmap='binary')
    ax[2].set_title('Misfit')

    u_pinn02 = ((ux02.reshape(xx.shape)) ** 2 + (uz02.reshape(xx.shape)) ** 2) ** 0.5
    u_diff02 = np.sqrt(u_ini2x ** 2 + u_ini2z ** 2).reshape(xx.shape) - u_pinn02
    ax[3].contourf(xx * l_x, zz * l_z,
                   np.sqrt(u_ini2x ** 2 + u_ini2z ** 2).reshape(xx.shape), 100,
                   norm=norm1, cmap='seismic')
    ax[4].contourf(xx * l_x, zz * l_z, u_pinn02, 100, norm=norm1, cmap='seismic')
    ax[5].contourf(xx * l_x, zz * l_z, u_diff02, 100, norm=norm2, cmap='binary')

    u_pinnt = ((uxt.reshape(xx.shape)) ** 2 + (uzt.reshape(xx.shape)) ** 2) ** 0.5
    u_diff = np.sqrt(u_specx ** 2 + u_specz ** 2).reshape(xx.shape) - u_pinnt

    ax[6].contourf(xx * l_x, zz * l_z,
                   np.sqrt(u_specx ** 2 + u_specz ** 2).reshape(xx.shape), 100,
                   norm=norm1, cmap='seismic')
    ax[7].contourf(xx * l_x, zz * l_z, u_pinnt, 100, norm=norm1, cmap='seismic')

    im2 = ax[8].contourf(xx * l_x, zz * l_z, u_diff, 100,
                         norm=norm2, cmap='binary')

    fig.colorbar(im1, ax=[ax[0], ax[1], ax[3], ax[4], ax[6], ax[7]], orientation='horizontal',
                 ticks=FixedLocator([0, 0.5, 0.99]), fraction=0.02, pad=0.6)
    fig.colorbar(im2, ax=[ax[2], ax[5], ax[8]], orientation='horizontal', ticks=FixedLocator(
        [0, 0.06]), fraction=0.02, pad=0.6)

    fig.text(0.5, 0.15, 'x(km)', ha='center')
    fig.text(0.03, 0.5, 'z(km)', va='center', rotation='vertical')

    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)
    fig.subplots_adjust(top=0.75)
    fig.subplots_adjust(bottom=0.25)

    for i in range(9):
        ax[i].set_aspect(1)

    plt.savefig('wave_ponential.png', dpi=400)
    plt.close(fig)


def plot_alpha(xx, zz, x_s, alpha_true0, alpha0):
    """plot the True acoustic wavespeed  and Inverted wavespeed"""

    l_x = cfg['Lx']
    l_z = cfg['Lz']

    norm3 = matplotlib.colors.Normalize(vmin=2.5, vmax=3.5)
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
    ax = ax.flatten()
    im1 = ax[0].contourf(
        l_x * xx, l_z * zz, alpha_true0.reshape((xx.shape)), 100, norm=norm3, cmap='jet')
    ax[0].set_title(r'True acoustic wavespeed ($\alpha$)')
    ax[0].plot(l_x * 0.99 * x_s[:, 0], l_z * x_s[:, 1], 'r*', markersize=5)
    ax[1].contourf(xx * l_x, zz * l_z, alpha0.reshape(xx.shape),
                   100, norm=norm3, cmap='jet')
    ax[1].set_title(r'Inverted $\alpha$')
    im2 = ax[2].contourf(xx * l_x, zz * l_x, alpha_true0 -
                         (alpha0.reshape(xx.shape)), 100, cmap='jet')
    ax[2].set_title(r' $\alpha$ misfit (true-inverted)')

    for i in range(3):
        ax[i].set_aspect(1)

    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(hspace=0.3)

    fig.colorbar(im1, ax=[ax[0], ax[1]], ticks=FixedLocator(
        [2.6, 2.8, 3]), fraction=0.02, pad=0.1)
    fig.colorbar(im2, ax=[ax[2]], ticks=FixedLocator(
        [-0.2, -0.1, 0, 0.1, 0.2]), fraction=0.02, pad=0.1)
    plt.savefig('alpha.png', dpi=400)
    plt.close(fig)
