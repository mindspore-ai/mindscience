# ============================================================================
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
"""uncover coef"""
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from mindspore import Tensor, context
import mindspore.common.dtype as mstype

from src.derivative import PhysicsLossGenerator


class STRidgeTrainer:
    """trainer"""

    def __init__(self, r0, ut, normalize=2, split_ratio=0.8):

        self.r0 = r0
        self.ut = ut
        self.normalize = normalize
        self.split_ratio = split_ratio

        # split the training and testing data
        np.random.seed(0)
        n, d = self.r0.shape

        r = np.zeros((n, d), dtype=np.float32)
        if normalize != 0:
            mreg = np.zeros((d, 1))
            for i in range(d):
                mreg[i] = 1.0 / (np.linalg.norm(self.r0[:, i], normalize))
                r[:, i] = mreg[i] * self.r0[:, i]
            normalize_inner = 0
        else:
            r = r0
            mreg = np.ones((d, 1)) * d   # why multiply by d
            normalize_inner = 2

        self.mreg = mreg
        self.r = r        # R0 - raw, R - normalized
        self.normalize_inner = normalize_inner

        train_idx = []
        test_idx = []
        for i in range(n):
            if np.random.rand() < split_ratio:
                train_idx.append(i)
            else:
                test_idx.append(i)

        self.train_r = r[train_idx, :]
        self.test_r = r[test_idx, :]
        self.train_y = ut[train_idx, :]
        self.test_y = ut[test_idx, :]

    def train(self, maxit=200, str_iters=10, lam=0.0001, d_tol=10.0, l0_penalty=None, kappa=1.0, must_have=5):
        """
        This function trains a predictor using STRidge.

        It runs over different values of tolerance and trains predictors on a training set, then evaluates them
        using a loss function on a holdout set.

        Please note published article has typo.  Loss function used here for model selection evaluates fidelity
        using 2-norm, not squared 2-norm.
        """

        # Set up the initial tolerance and l0 penalty
        tol = d_tol

        # Get the standard least squares estimator
        w_best = np.linalg.lstsq(self.train_r, self.train_y)[0]
        err_f = np.mean((self.test_y - self.test_r.dot(w_best))**2)
        err_w = np.count_nonzero(w_best)

        if l0_penalty is None:
            l0_penalty = kappa*err_f

        err_best = err_f + l0_penalty*err_w

        tol_best = 0

        # Now increase tolerance until test performance decreases
        for it in range(maxit):

            # Get a set of coefficients and error
            w = self.stridge(self.train_r, self.train_y, lam, str_iters, tol,
                             self.mreg, normalize=self.normalize_inner, must_have=must_have)

            err_f = np.mean((self.test_y - self.test_r.dot(w))**2)
            err_w = np.count_nonzero(w)
            err_must_have = 50.0 if abs(w[must_have]) < 1e-10 else 0.0
            err = err_f + l0_penalty*err_w  # + err_must_have

            print('__'*20)
            print('Number of iter: ', it)
            print('Tolerence: %.7f ' % (tol))
            print('Regression error: %.10f, Penalty: %.2f , Must-have penalty %.2f' %
                  (err_f, err_w, err_must_have))
            print('Weight w (normalized):')
            print(w.T)

            # Has the accuracy improved?
            if err <= err_best:
                err_best = err
                w_best = w
                tol_best = tol
                tol = tol + d_tol

            else:
                tol = max([0, tol - 2*d_tol])
                d_tol = 2*d_tol / (maxit - it)
                tol = tol + d_tol

        print('Train STRidge completed!')
        print("Optimal tolerance:", tol_best)

        return np.multiply(self.mreg, w_best)

    def stridge(self, x0, y, lam, maxit, tol, merg, normalize=0, must_have=5):
        """
        Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
        approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

        This assumes y is only one column
        """

        n, d = x0.shape
        x = np.zeros((n, d), dtype=np.float64)

        # if not normalized in the outer loop, do it here
        if normalize != 0:
            merg = np.zeros((d, 1))
            for i in range(d):
                merg[i] = 1.0/(np.linalg.norm(x0[:, i], normalize))
                x[:, i] = merg[i]*x0[:, i]
        else:
            x = x0

        # Get the standard ridge estimate
        if lam != 0:
            w = np.linalg.lstsq(x.T.dot(x) + lam*np.eye(d), x.T.dot(y))[0]
        else:
            w = np.linalg.lstsq(x, y)[0]

        num_relevant = d
        biginds = np.where(abs(w) > tol)[0]

        # Threshold and continue
        for j in range(maxit):

            # Figure out which items to cut out
            smallinds = np.where(abs(w) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]

            if must_have not in new_biginds:
                new_biginds.append(must_have)
                new_biginds.sort()

            # If nothing changes then stop
            if num_relevant == len(new_biginds):
                break
            else:
                num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if not new_biginds:  # len of new_biginds is 0
                if j == 0:
                    print(
                        'Warning: initial tolerance %.4f too large, all coefficients under threshold!' % tol)
                    # return all zero coefficient immediately
                    return w*0

                # break and return the w in the last iteration
                break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0
            if lam != 0:
                w[biginds] = np.linalg.lstsq(x[:, biginds].T.dot(
                    x[:, biginds]) + lam*np.eye(len(biginds)), x[:, biginds].T.dot(y))[0]
            else:
                w[biginds] = np.linalg.lstsq(x[:, biginds], y)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []:
            w[biginds] = np.linalg.lstsq(x[:, biginds], y)[0]

        if normalize != 0:
            return np.multiply(merg, w)

        return w


def gen_library():
    list_a = ['ones', 'u', 'v', 'u**2', 'u*v',
              'v**2', 'u**3', 'u**2*v', 'u*v**2', 'v**3']
    list_b = ['ones', 'u_x', 'u_y', 'v_x', 'v_y', 'lap_u', 'lap_v']
    library = []
    for a in list_a:
        for b in list_b:
            library.append(a + '*' + b)
    return library


def uncover(sym, data_path):
    """uncover"""
    # read data of u, v
    uv = sio.loadmat(data_path)['uv']

    data_hr = uv[50:150]
    x1 = np.zeros(data_hr.shape)
    for i in range(data_hr.shape[0]):
        x1[i, ...] = data_hr[i, ...]
    pred_hr = Tensor(x1, mstype.float32)
    loss_generator = PhysicsLossGenerator(
        dx=0.01, dy=0.01, dt=0.00025, nu=0.005)
    terms = loss_generator.get_phy_residual(pred_hr)

    # Prepare each single terms
    terms_dict = {}
    for key in terms:
        terms_dict[key] = terms[key].asnumpy().flatten()[:, None]

    # Prepare all possible combinations among terms
    lib = gen_library()

    # Prepare ground truth
    coef = np.zeros((len(lib), 1))
    for i, name in enumerate(lib):
        if name == 'ones*lap_' + sym:
            coef[i] = 0.005
        elif name in ('v*' + sym + '_y', 'u*' + sym + '_x'):
            coef[i] = -1

    n = terms_dict.get('v', None)
    if n is None:
        raise KeyError('v should be in the dict.')
    n = n.shape[0]
    idx = np.random.choice(n, int(n * 0.2), replace=False)
    # randomly subsample 10% of the measurements clip
    terms_table = {}
    for key in terms_dict:
        expression = key + '=terms_dict[\'' + str(key) + '\'][idx, :]'
        print('Execute expression:', expression)
        terms_table[key] = terms_dict[key][idx, :]

    # Check the residual of ground truth (if you are interested)
    lhs_columns = []
    list_a = []
    try:
        list_a.extend(
            (terms_table['ones'], terms_table['u'], terms_table['v']))
        list_a.extend((terms_table['u']**2, terms_table['u']*terms_table['v'],
                       terms_table['v']**2, terms_table['u']**3, terms_table['v']**3))
        list_a.extend((terms_table['u']**2*terms_table['v'],
                       terms_table['u']*terms_table['v']**2))
        list_b = [terms_table['ones'], terms_table['u_x'], terms_table['u_y'],
                  terms_table['v_x'], terms_table['v_y'], terms_table['lap_u'], terms_table['lap_v']]
    except KeyError as e:
        print(f"Error: {e}")

    for a in list_a:
        for b in list_b:
            lhs_columns.append(a*b)

    lhs = np.concatenate(lhs_columns, axis=1)
    rhs = terms_table.get(sym+'_t', None)

    if rhs is None:
        raise KeyError(f'{sym}_t not found.')

    trainer = STRidgeTrainer(r0=lhs, ut=rhs, normalize=2, split_ratio=0.8)

    w_best = trainer.train(maxit=100, str_iters=40, lam=0.01,
                           d_tol=20, kappa=1, must_have=5)  # 30% noise kappa=10

    # get the valuation metrics, L2 error and discovery accuracy
    err_l2 = np.linalg.norm(w_best - coef, 2)/np.linalg.norm(coef, 2)
    dis_pr = np.count_nonzero(w_best * coef, 0) / np.count_nonzero(w_best, 0)
    dis_rc = np.count_nonzero(w_best*coef, 0)/np.count_nonzero(coef, 0)

    # note this is not the final result
    print('Relative L2 error: %.3f, discovery recall: %.3f, precision: %.3f' %
          (err_l2, dis_rc, dis_pr))

    # visualize the result
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.subplots_adjust(bottom=0.2)
    ax.plot(lib, w_best, '--', linewidth=1.5, marker="*", label='Identified')
    ax.plot(lib, coef, '--', linewidth=1.5, marker="^", label='Truth')
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('coe_comparison_'+sym+'.png')

    identified_dict = {}
    for coef, term in zip(list(w_best[:, 0]), lib):
        if coef != 0:
            identified_dict[term] = coef

    print(identified_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="burgers_uncover_coef train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=2,
                        help="ID of the target device")
    parser.add_argument("--data_path", type=str,
                        default="./dataset/Burgers_2001x2x100x100_[dt=00025].mat")
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id,
                        max_call_depth=99999999)
    uncover('u', args.data_path)
    uncover('v', args.data_path)
