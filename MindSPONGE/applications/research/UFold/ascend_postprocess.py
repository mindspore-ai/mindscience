# Copyright 2022 Huawei Technologies Co., Ltd
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
"""postprocess for 310 inference"""
import os
import math
import argparse
import collections
import numpy as np
import mindspore as ms


def constraint_matrix_batch(x):
    """
    this function is referred from e2efold utility function, located at
    https://github.com/ml4bio/e2efold/tree/master/e2efold/common/utils.py
    """
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    au = np.matmul(base_a.reshape(batch, length, 1), base_u.reshape(batch, 1, length))
    au_ua = au + np.transpose(au, [0, 2, 1])
    cg = np.matmul(base_c.reshape(batch, length, 1), base_g.reshape(batch, 1, length))
    cg_gc = cg + np.transpose(cg, [0, 2, 1])
    ug = np.matmul(base_u.reshape(batch, length, 1), base_g.reshape(batch, 1, length))
    ug_gu = ug + np.transpose(ug, [0, 2, 1])
    return au_ua + cg_gc + ug_gu


def contact_a(a_hat, m):
    """contact a_hat"""
    a = a_hat * a_hat
    a = (a + np.transpose(a, [0, 2, 1])) / 2
    a = a * m
    return a


def soft_sign(x):
    """softsigh function"""
    k = 1
    return 1.0/(1.0+np.exp(-2*k*x))


def relu(x):
    # relu函数
    return np.maximum(0, x)


def sigmoid(x):
    # sigmoid函数
    return 1 / (1 + np.exp(-x))


def evaluate_exact_new(pred_a, true_a, eps=1e-11):
    """get pred, recall and f1_score"""
    tp_map = np.sign(pred_a * true_a)
    tp = tp_map.sum()
    pred_p = np.sign(pred_a).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps)/(tp + fn + eps)
    precision = (tp + eps)/(tp + fp + eps)
    f1_score_ms = (2 * tp + eps)/(2 * tp + fp + fn + eps)
    return precision, recall, f1_score_ms


def postprocess(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=math.log(9.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    m = constraint_matrix_batch(x)
    u = u.reshape(1, 80, 80)
    u = soft_sign(u - s) * u

    a_hat = sigmoid(u) * soft_sign(u - s)

    lmbd = relu(np.sum(contact_a(a_hat, m), axis=-1) - 1)
    # gradient descent
    for _ in range(num_itr):
        grad_a = lmbd * soft_sign(np.sum(contact_a(a_hat, m), axis=-1) - 1)
        grad_a = np.expand_dims(grad_a, axis=-1)
        grad_a = np.broadcast_to(grad_a, u.shape) - u / 2

        grad = a_hat * m * (grad_a + np.transpose(grad_a, [0, 2, 1]))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = relu(np.abs(a_hat) - rho * lr_min)

        lmbd_grad = relu(np.sum(contact_a(a_hat, m), axis=-1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

    a = a_hat * a_hat
    a = (a + np.transpose(a, [0, 2, 1])) / 2
    a = a * m
    return a



if __name__ == '__main__':
    ms.set_seed(1)
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    parser = argparse.ArgumentParser(description="postprocess")
    parser.add_argument("--result_dir", type=str, default="ascend310_infer/result_Files", help="result files path.")
    parser.add_argument("--ori", type=str, default="ascend310_infer/ori", help="result ori path.")
    parser.add_argument("--contact", type=str, default="ascend310_infer/contacts", help="result ori path.")
    args = parser.parse_args()

    print('postprocess start!')

    result_no_train = list()
    seq_lens_list = list()
    pre_file = []
    pre_ori = []
    pre_con = []
    score = []
    score_ori = []
    score_con = []
    contact_batch = []
    map_no_train = []

    rst_path = args.result_dir
    ori_path = args.ori
    con_path = args.contact
    filenames = os.listdir(rst_path)
    fileorinames = os.listdir(ori_path)
    fileconnames = os.listdir(con_path)

    for filename in filenames:
        pre_file.append(filename)
    for idx, filename in enumerate(pre_file):
        pred = np.fromfile(os.path.join(rst_path, filename), np.float32)
        score.append(pred)

    for filename in fileorinames:
        pre_ori.append(filename)
    for idx, filename in enumerate(pre_ori):
        pred_ori = np.fromfile(os.path.join(ori_path, filename), np.float32)
        score_ori.append(pred_ori)

    for filename in fileconnames:
        pre_con.append(filename)
    for idx, filename in enumerate(pre_con):
        pred_con = np.fromfile(os.path.join(con_path, filename), np.float32)
        score_con.append(pred_con)


    for _batch, pred in enumerate(score):
        pred_contacts = pred
        contact_batch = score_con[_batch]
        contact_batch = contact_batch.reshape(1, 80, 80)
        seq_ori = score_ori[_batch]
        seq_ori = seq_ori.reshape(1, 80, 4)
        u_no_train = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)


        map_no_train = (u_no_train > 0.5).astype(np.float32)
        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train[i],
                                                                    contact_batch[i]), range(contact_batch.shape[0])))
        result_no_train += result_no_train_tmp

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result_no_train)

    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
