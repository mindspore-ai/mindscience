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
    Help functions of MT-VAE code
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def FspecialGaussian(p2, p3):
    """

    :param p2: patch side length
    :param p3: Gauss std
    :return:
    """
    siz = (p2 - 1) / 2
    std = p3

    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    try:
        arg = -(x * x + y * y) / (2 * std * std)
    except ZeroDivisionError as e:
        logging.error("Error %s", e)
        raise
    eps = 2.2204e-16
    h = np.exp(arg)
    h[h < eps * np.max(np.reshape(h, (-1, 1), order="f"))] = 0

    sumh = np.sum(np.reshape(h, (-1, 1), order="f"))
    if sumh != 0:
        h = h / sumh
    return h


def Plot2DImage(coor, rangex=None, rangez=None, iflog=0, use_cmap="jet"):
    """

    :param coor:
    :param rangex:
    :param rangez:
    :param iflog:
    :param use_cmap:
    :return:
    """
    zelementlocation = coor["zelementlocation"]
    xelementlocation = coor["xelementlocation"]
    xx = coor["xx"]
    colorbaraxis = coor["colorbaraxis"]
    surf_flag = coor["surf_flag"]
    address = coor["address"]
    if rangex is None:
        rangex = [0, 10]
    if rangez is None:
        rangez = [-2, 0]
    znumber = len(zelementlocation)
    xnumber = len(xelementlocation)
    [x, y] = np.meshgrid(xelementlocation / 1000, -zelementlocation / 1000)
    if iflog:
        a = np.reshape(xx, (znumber, xnumber), order="f")
    else:
        a = np.reshape(10 ** xx, (znumber, xnumber), order="f")
    if surf_flag == 1:
        plt.ion()
        fig = plt.figure(figsize=(8, 3))
        fig.add_subplot(1, 1, 1)
        plt.pcolor(x, y, a, cmap=plt.get_cmap(use_cmap))
        plt.xlim(rangex[0], rangex[1])
        plt.ylim(rangez[0], rangez[1])
        plt.xlabel("Distance (km)")
        plt.ylabel("Depth (km)")
        cbar = plt.colorbar()
        if iflog:
            plt.clim(colorbaraxis[0], colorbaraxis[1])
        else:
            plt.clim(10 ** colorbaraxis[0], 10 ** colorbaraxis[1])

        cbar.set_label("Logarithm of Resistivity")
        plt.tight_layout()
        plt.savefig(address)
        plt.ioff()
        plt.close()


def ComputeGradient(xnumber, znumber):
    """

    :param xnumber:
    :param znumber:
    :return:
    """
    gridnumber = xnumber * znumber
    ivertical = -1 * np.ones(2 * gridnumber)
    jvertical = -1 * np.ones(2 * gridnumber)
    valuevertical = -10 * np.ones(2 * gridnumber)
    ihorizontal = -1 * np.ones(2 * gridnumber)
    jhorizontal = -1 * np.ones(2 * gridnumber)
    valuehorizontal = -10 * np.ones(2 * gridnumber)
    indexvertical = 0
    indexhorizontal = 0
    for i in range(gridnumber):
        temp1 = i % znumber
        if temp1 != znumber - 1:
            ivertical[indexvertical] = i
            jvertical[indexvertical] = i
            valuevertical[indexvertical] = 1
            indexvertical = indexvertical + 1
            ivertical[indexvertical] = i
            jvertical[indexvertical] = i + 1
            valuevertical[indexvertical] = -1
            indexvertical = 1 + indexvertical

        if i < znumber * (xnumber - 1):
            ihorizontal[indexhorizontal] = i
            jhorizontal[indexhorizontal] = i
            valuehorizontal[indexhorizontal] = 1
            indexhorizontal = 1 + indexhorizontal
            ihorizontal[indexhorizontal] = i
            jhorizontal[indexhorizontal] = i + znumber
            valuehorizontal[indexhorizontal] = -1
            indexhorizontal = 1 + indexhorizontal

    ivertical = ivertical[ivertical != -1]
    jvertical = jvertical[jvertical != -1]
    valuevertical = valuevertical[valuevertical != -10]
    ihorizontal = ihorizontal[ihorizontal != -1]
    jhorizontal = jhorizontal[jhorizontal != -1]
    valuehorizontal = valuehorizontal[valuehorizontal != -10]
    verticalgradient = csr_matrix(
        (valuevertical, (ivertical, jvertical)), shape=(gridnumber, gridnumber)
    )
    horizontalgradient = csr_matrix(
        (valuehorizontal, (ihorizontal, jhorizontal)), shape=(gridnumber, gridnumber)
    )
    return verticalgradient, horizontalgradient
