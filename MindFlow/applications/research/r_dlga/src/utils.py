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
# ============================================================================
"""utils"""
import random
import time
import collections


import numpy as np
from sklearn.linear_model import Lasso

from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype

from mindflow.utils import print_log

from .grads import UNet, VNet, PNet, Grad
from .generalized_ga import BurgersGeneAlgorithm, CylinderFlowGeneAlgorithm, PeriodicHillGeneAlgorithm
from .divide import divide_with_error

DatasetResult = collections.namedtuple('DatasetResult', [
    'h_data_choose', 'h_data_validate', 'database_choose', 'database_validate'])


class TrainArgs:
    """parameters"""

    def __init__(self, database_choose, h_data_choose, lefts, coef_list, dict_name, terms_dict):
        """init"""
        self.database_choose = database_choose
        self.h_data_choose = h_data_choose
        self.lefts = lefts
        self.coef_list = coef_list
        self.dict_name = dict_name
        self.terms_dict = terms_dict


class LossArgs:
    """loss args"""

    def __init__(self, prediction, label, lefts, coef, choose_train, dict_name, term_dict):
        """init"""
        self.prediction = prediction
        self.label = label
        self.lefts = lefts
        self.coef = coef
        self.choose_train = choose_train
        self.dict_name = dict_name
        self.term_dict = term_dict


def random_data(data_info, random_seed=525):
    """produce random data for burgers"""
    total, choose, data_validate, x, t, un, x_num, t_num = data_info
    random.seed(random_seed)
    un_raw = un
    data = np.zeros(2)
    h_data = np.zeros((total, 1))
    database = np.zeros((total, 2))

    num = 0

    for j in range(x_num):
        for i in range(t_num):
            data[0] = x[j]
            data[1] = t[i]
            h_data[num] = un_raw[j, i]
            database[num] = data
            num += 1

    a = []
    b = []
    data_array = np.arange(0, x_num*t_num, 1)
    np.random.seed(random_seed)
    np.random.shuffle(data_array)
    for i in range(choose):
        a.append(data_array[i])
    for i in range(data_validate):
        b.append(data_array[choose+i])

    choose = int(choose)
    data_validate = int(data_validate)
    h_data_choose = np.zeros((choose, 1))
    database_choose = np.zeros((choose, 2))
    h_data_validate = np.zeros((data_validate, 1))
    database_validate = np.zeros((data_validate, 2))
    num = 0
    for i in a:
        i = int(i)
        h_data_choose[num] = h_data[i]
        database_choose[num] = database[i]
        num += 1
    num = 0
    for i in b:
        i = int(i)
        h_data_validate[num] = h_data[i]
        database_validate[num] = database[i]
        num += 1
    return DatasetResult(h_data_choose, h_data_validate, database_choose, database_validate)


def random_cylinder_flow_data(choose, data_validate, points, label, random_seed=525):
    """produce random data for cylinder flow"""
    random.seed(random_seed)

    x_num = points.shape[0]
    y_num = points.shape[1]
    t_num = points.shape[2]

    label_raw = label
    total = x_num * y_num * t_num

    h_data = np.zeros((total, 3))   # labels
    database = np.zeros((total, 3))  # coordinates

    num = 0

    # flatten labels and coordinates
    for k in range(x_num):
        for j in range(y_num):
            for i in range(t_num):
                h_data[num] = label_raw[k, j, i]
                database[num] = points[k, j, i]
                num += 1

    a = []
    b = []
    data_array = np.arange(0, total, 1)
    np.random.seed(random_seed)
    np.random.shuffle(data_array)
    for i in range(choose):
        a.append(data_array[i])
    for i in range(data_validate):
        b.append(data_array[choose+i])

    choose = int(choose)
    data_validate = int(data_validate)
    h_data_choose = np.zeros((choose, 3))
    database_choose = np.zeros((choose, 3))
    h_data_validate = np.zeros((data_validate, 3))
    database_validate = np.zeros((data_validate, 3))

    num = 0
    for i in a:
        i = int(i)
        h_data_choose[num] = h_data[i]
        database_choose[num] = database[i]
        num += 1
    num = 0
    for i in b:
        i = int(i)
        h_data_validate[num] = h_data[i]
        database_validate[num] = database[i]
        num += 1

    return DatasetResult(h_data_choose, h_data_validate, database_choose, database_validate)


def random_periodic_hill_data(choose_train, data_validate, points, label, random_seed=525):
    """produce random data for periodic hill"""
    np.random.seed(random_seed)

    x_num = points.shape[0]
    y_num = points.shape[1]

    label_raw = label
    total = x_num * y_num

    h_data = np.zeros((total, 3))   # labels
    database = np.zeros((total, 2))  # coordinates

    num = 0

    # flatten labels and coordinates
    for j in range(y_num):
        for i in range(x_num):
            h_data[num] = label_raw[i, j]
            database[num] = points[i, j]
            num += 1

    data_array = np.arange(0, total, 1)
    np.random.shuffle(data_array)

    a = data_array[:choose_train]
    b = data_array[choose_train:choose_train + data_validate]

    h_data_choose = h_data[a]
    database_choose = database[a]
    h_data_validate = h_data[b]
    database_validate = database[b]

    return DatasetResult(h_data_choose, h_data_validate, database_choose, database_validate)


def get_dict_name(case_name):
    """Get the name/number of the dictionary.
    Burgers result: ["(HHxx)xxx_n", "(H**2)x_n", "Hxx_n", "Hxxxx_n", "(HHx)xx_n", "Hxx_n"]
    Cylinder_flow result:
        Ut: ["Ux", "U*Vy", "P"]
        Vt: ["V", "Vx", "(V*Px)x", "Py"]
        Pt: ["Px", "P"]
    Periodic_hill result:
           U: ["Ux", "Uy*Vy", "Px"]
           V: ["V", "Vx", "Vy", "Py"]
           P: ["Vx", "Px"]"""
    if case_name == "burgers":
        dict_name = [[1, 2, 3, 4, 5, 6]]
    else:
        dict_name = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    return dict_name


def evaluate(model, inputs, label, config):
    """Evaluate the model respect to input data and label."""

    # get prediction
    label_shape = label.shape
    batch_size = config["dataset"]["validate_batch"]
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    time_beg = time.time()

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + batch_size, inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    print_log(f"    predict total time: {(time.time() - time_beg)*1000} ms")
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))

    label = label.reshape((-1, label_shape[1]))

    # calculate l2 error
    error = label - prediction

    norm_value = np.sqrt(np.sum(np.square(label[..., 0])))
    l2_error = divide_with_error(np.sqrt(
        np.sum(np.square(error[..., 0]))), norm_value)
    print_log(f"    l2_error: {l2_error}")
    print_log(
        "==================================================================================================")


def burgers_cal_grads(model, database, choose):
    """Calculate grads."""
    burgers_grads = collections.namedtuple("Grads", ["Hx", "Hx_n", "Ht", "Ht_n", "Hxx", "Hxx_n",
                                                     "Hxxx", "Hxxx_n", "Hxxxx", "Hxxxx_n", "Hxxxxx", "Hxxxxx_n"])

    database = Tensor(database, mstype.float32)
    grad1 = Grad(model)(database)
    grad2 = Grad(Grad(model))(database)
    grad3 = Grad(Grad(Grad(model)))(database)
    grad4 = Grad(Grad(Grad(Grad(model))))(database)
    grad5 = Grad(Grad(Grad(Grad(Grad(model)))))(database)

    hx = grad1[:, 0].reshape(choose, 1)
    hx_n = hx.asnumpy()
    ht = grad1[:, 1].reshape(choose, 1)
    ht_n = ht.asnumpy()
    hxx = grad2[:, 0].reshape(choose, 1)
    hxx_n = hxx.asnumpy()
    hxxx = grad3[:, 0].reshape(choose, 1)
    hxxx_n = hxxx.asnumpy()
    hxxxx = grad4[:, 0].reshape(choose, 1)
    hxxxx_n = hxxxx.asnumpy()
    hxxxxx = grad5[:, 0].reshape(choose, 1)
    hxxxxx_n = hxxxxx.asnumpy()

    grads = burgers_grads(hx, hx_n, ht, ht_n, hxx, hxx_n,
                          hxxx, hxxx_n, hxxxx, hxxxx_n, hxxxxx, hxxxxx_n)
    libraries = [ht_n]
    return grads, libraries


CylinderFlowGrads = collections.namedtuple("Grads", ["Ut", "Ut_n", "Ux", "Ux_n",
                                                     "Vt", "Vt_n", "Vx", "Vx_n", "Vy", "Vy_n",
                                                     "Pt", "Pt_n", "Px", "Px_n", "Py", "Py_n", "Pxx", "Pxx_n"])
NsTerms = collections.namedtuple(
    "Terms", ["term1", "term2", "term3",
              "term4", "term5", "term6", "term7",
              "term8", "term9"])


def cylinder_flow_cal_grads(model, database, choose):
    """get grads. for cylinder flow"""
    database = Tensor(database, mstype.float32)
    u = UNet(model)
    v = VNet(model)
    p = PNet(model)

    u_grad1 = Grad(u)(database)
    v_grad1 = Grad(v)(database)

    p_grad1 = Grad(p)(database)
    p_grad2 = Grad(Grad(p))(database)

    ut = u_grad1[:, 2].reshape(choose, 1)
    ut_n = ut.asnumpy()
    ux = u_grad1[:, 0].reshape(choose, 1)
    ux_n = ux.asnumpy()

    vt = v_grad1[:, 2].reshape(choose, 1)
    vt_n = vt.asnumpy()
    vx = v_grad1[:, 0].reshape(choose, 1)
    vx_n = vx.asnumpy()
    vy = v_grad1[:, 1].reshape(choose, 1)
    vy_n = vy.asnumpy()

    pt = p_grad1[:, 2].reshape(choose, 1)
    pt_n = pt.asnumpy()
    px = p_grad1[:, 0].reshape(choose, 1)
    px_n = px.asnumpy()
    py = p_grad1[:, 1].reshape(choose, 1)
    py_n = py.asnumpy()
    pxx = p_grad2[:, 0].reshape(choose, 1)
    pxx_n = pxx.asnumpy()

    grads = CylinderFlowGrads(ut, ut_n, ux, ux_n, vt, vt_n, vx, vx_n, vy, vy_n,
                              pt, pt_n, px, px_n, py, py_n, pxx, pxx_n)
    libraries = [ut_n, vt_n, px_n]
    return grads, libraries


PeriodicHillGrads = collections.namedtuple("Grads", ["Ux", "Ux_n", "Uy", "Uy_n",
                                                     "Vx", "Vx_n", "Vy", "Vy_n",
                                                     "Px", "Px_n", "Py", "Py_n"])


def periodic_hill_cal_grads(model, database, choose):
    """cal grads for periodic hill"""
    database = Tensor(database, mstype.float32)
    u = UNet(model)
    v = VNet(model)
    p = PNet(model)

    u_grad1 = Grad(u)(database)
    v_grad1 = Grad(v)(database)

    p_grad1 = Grad(p)(database)

    ux = u_grad1[:, 0].reshape(choose, 1)
    ux_n = ux.asnumpy()
    uy = u_grad1[:, 1].reshape(choose, 1)
    uy_n = uy.asnumpy()

    vx = v_grad1[:, 0].reshape(choose, 1)
    vx_n = vx.asnumpy()
    vy = v_grad1[:, 1].reshape(choose, 1)
    vy_n = vy.asnumpy()

    px = p_grad1[:, 0].reshape(choose, 1)
    px_n = px.asnumpy()
    py = p_grad1[:, 1].reshape(choose, 1)
    py_n = py.asnumpy()

    grads = PeriodicHillGrads(
        ux, ux_n, uy, uy_n, vx, vx_n, vy, vy_n, px, px_n, py, py_n)
    libraries = [u(database).reshape(choose, 1), v(
        database).reshape(choose, 1), p(database).reshape(choose, 1)]
    return grads, libraries


def cal_grads(case_name, model, database, choose):
    """Calculate grads."""
    if case_name == "burgers":
        grads, libraries = burgers_cal_grads(model, database, choose)
    elif case_name == "cylinder_flow":
        grads, libraries = cylinder_flow_cal_grads(model, database, choose)
    else:
        grads, libraries = periodic_hill_cal_grads(model, database, choose)
    return grads, libraries


def burgers_cal_terms(prediction, grads):
    """Calculate burgers terms."""
    burgers_terms = collections.namedtuple(
        "Terms", ["term1", "term2", "term3", "term4", "term5", "term6"])
    term1 = 4 * grads.Hxx * grads.Hxxx + \
        3 * grads.Hx * grads.Hxxxx + \
        prediction * grads.Hxxxxx
    term2 = 2 * prediction * grads.Hx
    term3 = grads.Hxx
    term4 = grads.Hxxxx
    term5 = 3 * grads.Hx * grads.Hxx + prediction * grads.Hxxx
    term6 = grads.Hxx
    terms = burgers_terms(term1, term2, term3, term4, term5, term6)
    return terms


def cylinder_flow_cal_terms(prediction, grads):
    """cal terms for cylinder flow"""
    u_value = prediction[:, 0].reshape(grads.Ux.shape)
    v_value = prediction[:, 1].reshape(grads.Vx.shape)
    p_value = prediction[:, 2].reshape(grads.Px.shape)
    term1 = grads.Ux
    term2 = u_value * grads.Vy
    term3 = p_value
    term4 = v_value
    term5 = grads.Vx
    term6 = grads.Vx * grads.Px + v_value * grads.Pxx
    term7 = grads.Py
    term8 = grads.Px
    term9 = p_value
    terms = NsTerms(term1, term2, term3,
                    term4, term5, term6, term7, term8, term9)
    return terms


def periodic_hill_cal_terms(prediction, grads):
    """cal term for periodic hill"""
    v_value = prediction[:, 1].reshape(grads.Vx.shape)
    term1 = grads.Ux
    term2 = grads.Uy * grads.Vy
    term3 = grads.Px
    term4 = v_value
    term5 = grads.Vx
    term6 = grads.Vy
    term7 = grads.Py
    term8 = grads.Vx
    term9 = grads.Px
    terms = NsTerms(term1, term2, term3,
                    term4, term5, term6, term7, term8, term9)
    return terms


def cal_terms(case_name, prediction, grads):
    """calculate terms."""
    if case_name == "burgers":
        terms = burgers_cal_terms(prediction, grads)
    elif case_name == "cylinder_flow":
        terms = cylinder_flow_cal_terms(prediction, grads)
    else:
        terms = periodic_hill_cal_terms(prediction, grads)
    return terms


def get_dicts(terms):
    """Get the dictionary of terms in numpy."""
    dict_count = 1
    term_dict = {}
    dict_n = {}
    for term in terms._fields:
        term_value = getattr(terms, term)
        term_dict[dict_count] = term_value
        dict_n[dict_count] = term_value.asnumpy()
        dict_count += 1
    return term_dict, dict_n


def update_lib(case_name, dict_name, libraries, dict_n):
    """Update the libraries."""
    if case_name == "burgers":
        for key in dict_name[0]:
            libraries[0] = np.hstack((libraries[0], dict_n[key]))
        libraries[0] = np.delete(libraries[0], [0], axis=1)
    else:
        libraries[0] = np.hstack((libraries[0], dict_n[1]))
        libraries[0] = np.hstack((libraries[0], dict_n[2]))
        libraries[0] = np.hstack((libraries[0], dict_n[3]))
        libraries[0] = np.delete(libraries[0], [0], axis=1)
        libraries[1] = np.hstack((libraries[1], dict_n[4]))
        libraries[1] = np.hstack((libraries[1], dict_n[5]))
        libraries[1] = np.hstack((libraries[1], dict_n[6]))
        libraries[1] = np.hstack((libraries[1], dict_n[7]))
        libraries[1] = np.delete(libraries[1], [0], axis=1)
        libraries[2] = np.hstack((libraries[2], dict_n[8]))
        libraries[2] = np.hstack((libraries[2], dict_n[9]))
        libraries[2] = np.delete(libraries[2], [0], axis=1)
    return libraries


def get_lefts(case_name, grads, prediction, if_tensor):
    """get lefts"""
    if case_name == "burgers":
        if if_tensor:
            lefts = [Tensor(grads.Ht_n, dtype=mstype.float32)]
        else:
            lefts = [grads.Ht_n]
    elif case_name == "cylinder_flow":
        if if_tensor:
            lefts = [Tensor(grads.Ut_n, dtype=mstype.float32), Tensor(
                grads.Vt_n, dtype=mstype.float32), Tensor(grads.Pt_n, dtype=mstype.float32)]
        else:
            lefts = [grads.Ut_n, grads.Vt_n, grads.Pt_n]
    else:
        shape_value = grads.Ux.shape
        u_value = prediction[:, 0].reshape(shape_value)
        v_value = prediction[:, 1].reshape(shape_value)
        p_value = prediction[:, 2].reshape(shape_value)
        if if_tensor:
            lefts = [Tensor(u_value, dtype=mstype.float32), Tensor(
                v_value, dtype=mstype.float32), Tensor(p_value, dtype=mstype.float32)]
        else:
            lefts = [u_value, v_value, p_value]
    return lefts


def calculate_coef(lefts, libraries, epoch, config):
    """calculate coef"""
    coef_list = []
    lst_list = []
    for i, library in enumerate(libraries):
        library = libraries[i]
        if epoch <= 1000:
            lr = Lasso(alpha=config["pinn"]["L1_norm"], max_iter=2000)
            lr.fit(library, lefts[i])
            lst = lr.coef_
        else:
            lst = np.linalg.lstsq(library, lefts[i], rcond=None)[0]
        coef = Tensor(lst.astype(np.float32))
        coef_list.append(coef)
        lst_list.append(Tensor(lst.astype(np.float32)))
    return coef_list, lst_list


def pinn_loss_func(f1, lefts, coef, dict_name, term_dict):
    """calculate loss"""
    f1 = ops.reshape(f1, (-1,))
    for index, h_t in enumerate(lefts):
        res = h_t
        flag = 0
        zeros = ops.zeros_like(h_t)
        for key in dict_name[index]:
            res = res - term_dict[key] * coef[index][flag]
            flag += 1
        f2 = nn.MSELoss(reduction='mean')(res, zeros)
        f1 = ops.concat((f1, ops.reshape(f2, (-1,))), axis=0)
    mean_loss_value = ops.ReduceMean()(f1)
    return mean_loss_value


def gene_algorithm(case_name, config):
    """gene algorithm for three cases"""
    if case_name == "burgers":
        burgers_ga = BurgersGeneAlgorithm(config=config, case_name=case_name)
        total_genes, total_gene_left = burgers_ga.get_genes()
        child_select, left_select = burgers_ga.get_child(
            total_genes, total_gene_left)
        burgers_ga.evolve(child_select, left_select)
    elif case_name == "cylinder_flow":
        cylinder_flow_ga = CylinderFlowGeneAlgorithm(config=config, case_name=case_name)
        total_genes, total_gene_left = cylinder_flow_ga.get_genes()
        child_select, left_select = cylinder_flow_ga.get_child(
            total_genes, total_gene_left)
        cylinder_flow_ga.evolve(child_select, left_select)

        cylinder_flow_ga.change_left_v()
        total_genes, total_gene_left = cylinder_flow_ga.get_genes()
        child_select, left_select = cylinder_flow_ga.get_child(
            total_genes, total_gene_left)
        cylinder_flow_ga.evolve(child_select, left_select)

        cylinder_flow_ga.change_left_p()
        total_genes, total_gene_left = cylinder_flow_ga.get_genes()
        child_select, left_select = cylinder_flow_ga.get_child(
            total_genes, total_gene_left)
        cylinder_flow_ga.evolve(child_select, left_select)
    else:
        periodic_hill_ga = PeriodicHillGeneAlgorithm(config=config, case_name=case_name)
        total_genes, total_gene_left = periodic_hill_ga.get_genes()
        child_select, left_select = periodic_hill_ga.get_child(
            total_genes, total_gene_left)
        periodic_hill_ga.evolve(child_select, left_select)

        periodic_hill_ga.change_left_v()
        total_genes, total_gene_left = periodic_hill_ga.get_genes()
        child_select, left_select = periodic_hill_ga.get_child(
            total_genes, total_gene_left)
        periodic_hill_ga.evolve(child_select, left_select)

        periodic_hill_ga.change_left_p()
        total_genes, total_gene_left = periodic_hill_ga.get_genes()
        child_select, left_select = periodic_hill_ga.get_child(
            total_genes, total_gene_left)
        periodic_hill_ga.evolve(child_select, left_select)
