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
"""model trainers"""
import os
import time
import warnings
import stat
import joblib

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import mindspore as ms
import mindspore.dataset as ds


def train_cls(model, data, params):
    '''Train classification network'''
    # load params
    model_name = params['model_name']
    exp_name = params['exp_name']
    num_epoch = params['num_epoch']
    lr = params['lr']
    w_decay = params['weight_decay']
    folder_dir = params['folder_dir']
    # prepare data split
    latents, label_y = data
    kf = KFold(n_splits=params['num_fold'])

    # model training
    train_acc = []
    test_acc = []
    k = 1

    # prepare training
    optimizer = ms.nn.Adam(params=model.trainable_params(), learning_rate=lr,
                           weight_decay=w_decay)  # initialize optimizer
    def forward_fn(input_x, label):
        y_pred = model(input_x)
        loss = ms.ops.binary_cross_entropy(y_pred, label)
        return loss, y_pred

    grad_fn = ms.ops.value_and_grad(forward_fn, None, optimizer.parameters)

    @ms.jit()
    def train_step(step_x, step_y):
        ((step_loss, step_y_pred), grads) = grad_fn(step_x, step_y)
        step_loss = ms.ops.depend(step_loss, optimizer(grads))
        return step_loss, step_y_pred

    for train, test in kf.split(latents):
        # split train and test data
        x_train, x_test, y_train, y_test = latents[train], latents[test], label_y[train], label_y[test]
        # prepare train data
        train_data = ds.NumpySlicesDataset(data={'x': x_train, 'y': y_train}, shuffle=True)
        train_data = train_data.batch(batch_size=params['batch_size'])
        train_iterator = train_data.create_dict_iterator()
        # prepare save_dir for checkpoint
        if not os.path.isdir(folder_dir):
            os.mkdir(folder_dir)
            warnings.warn('current model file not exists, please check history model training record.')
        if params['save_log']:
            flags = os.O_RDWR  | os.O_CREAT
            modes = stat.S_IWUSR | stat.S_IRUSR
            train_record = os.open(folder_dir + '/' + model_name + '-' + exp_name + '.txt', flags, modes)

        # start model training
        for epoch in range(num_epoch):
            start_time = time.time()
            epoch_acc = []
            test_epoch_acc = []
            model.set_train(True)
            for _, data_ in enumerate(train_iterator):
                x = data_['x']
                y = data_['y']
                iter_y_pred = train_step(x, y)[1]
                # train accuracy
                iter_acc = ms.numpy.equal(
                    ms.numpy.where(iter_y_pred >= ms.Tensor(0.5), ms.Tensor(1.), ms.Tensor(0.)),
                    y, ms.float32).mean().asnumpy()
                epoch_acc.append(iter_acc)
            # test
            # prepare test data
            test_data = ds.NumpySlicesDataset(data={'x': x_test, 'y': y_test}, shuffle=False)
            test_data = test_data.batch(batch_size=len(y_test))
            test_iterator = test_data.create_dict_iterator()

            for _, data_ in enumerate(test_iterator):
                x = data_['x']
                y = data_['y']
                test_y_pred = model(x)
                # test accuracy
                test_iter_acc = ms.numpy.equal(
                    ms.numpy.where(test_y_pred >= ms.Tensor(0.5), ms.Tensor(1.), ms.Tensor(0.)),
                    y, ms.float32).mean().asnumpy()
                test_epoch_acc.append(test_iter_acc)
            # print training info
            record = '[{}/{}/{}] train_acc: {:.04f} || test_acc: {:.04f}, time: {:.3f} sec'.format(epoch,
                                                                                                   k,
                                                                                                   params['num_fold'],
                                                                                                   sum(epoch_acc) /
                                                                                                   len(epoch_acc),
                                                                                                   sum(test_epoch_acc) /
                                                                                                   len(test_epoch_acc),
                                                                                                   time.time() -
                                                                                                   start_time)
            print(record)
            if params['save_log']:
                # save loss record
                os.write(train_record, str.encode(record + '\n'))
        train_acc_ = sum(epoch_acc) / len(epoch_acc)
        test_acc_ = sum(test_epoch_acc) / len(test_epoch_acc)
        train_acc.append(train_acc_)
        test_acc.append(test_acc_)
        k += 1
    record = 'average acc: train_acc: {:.04f} || test_acc: {:.04f}'.format(sum(train_acc) / len(train_acc),
                                                                           sum(test_acc) / len(test_acc))
    print(record)
    # save model checkpoint
    save_model_file = str(model_name + ".ckpt")
    save_model_dir = os.path.join(folder_dir, save_model_file)
    ms.save_checkpoint(model, save_model_dir)

    # save training info
    if params['save_log']:
        os.write(train_record, str.encode(record + '\n'))
        # loss record saved
        train_record.close()

    # visualize classifier
    if params['visualize']:
        plt.figure()
        sns.set_style()
        plt.xlabel('number of folds')
        plt.ylabel('loss')
        x = range(1, params['num_fold'] + 1)
        sns.set_style("darkgrid")
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        plt.plot(x, train_acc)
        plt.plot(x, test_acc, linestyle=':', c='steelblue')
        plt.legend(["train_accuracy", "test_accuracy"])
        ax.xaxis.set_major_locator(x_major_locator)
        plt.savefig(folder_dir + '/binary_classifier.png', dpi=300)
    print('=' * 200 + '\n' + 'Training Complete! Model file saved at' + save_model_dir + '\n' + '==' * 200)


def imq_kernel(input_x, output_y, h_dim):
    '''Compute maximum mean discrepancy using inverse multiquadric kernel'''
    batch_size = input_x.shape[0]
    norms_x = input_x.pow(2).sum(axis=1, keepdims=True)
    prods_x = ms.ops.MatMul()(input_x, input_x.T)
    dists_x = norms_x + norms_x.T - 2 * prods_x
    norms_y = output_y.pow(2).sum(axis=1, keepdims=True)
    prods_y = ms.ops.MatMul()(output_y, output_y.T)
    dists_y = norms_y + norms_y.T - 2 * prods_y
    dot_prd = ms.ops.MatMul()(input_x, output_y.T)
    dists_c = norms_x + norms_y.T - 2 * dot_prd
    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        c = 2 * h_dim * 1.0 * scale
        res1 = c / (c + dists_x)
        res1 += c / (c + dists_y)
        res1 = (1 - ms.ops.eye(batch_size, batch_size, ms.float32)) * res1
        res1 = res1.sum() / (batch_size - 1)
        res2 = c / (c + dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2
    return stats


def get_latents(model, iterator):
    model.set_train(mode=False)
    latents = []
    for _, data in enumerate(iterator):
        x = data['x']
        z = model.encode(x)
        latents.append(z.asnumpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


def train_wae(model, data, params):
    ''' Train WAE generation network'''
    # load params
    model_name = params['model_name']
    exp_name = params['exp_name']
    num_epoch = params['num_epoch']
    batch_size = params['batch_size']
    sigma = params['sigma']
    mmd_lambda = params['MMD_lambda']
    folder_dir = params['folder_dir']
    lr = params['lr']
    w_decay = params['weight_decay']
    raw_x = data
    # prepare train data
    train_data = ds.NumpySlicesDataset(data={'x': raw_x[:]}, shuffle=True)
    train_data = train_data.batch(batch_size=batch_size)
    train_iterator = train_data.create_dict_iterator()

    # prepare save_dir for checkpoint
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
        warnings.warn('current model file not exists, please check history model training record.')
    if params['save_log']:
        flags = os.O_RDWR  | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        train_record = os.open(folder_dir + '/' + model_name + '-' + exp_name + '.txt', flags, modes)

    # prepare model training
    optimizer = ms.nn.Adam(params=model.trainable_params(), learning_rate=lr,
                           weight_decay=w_decay)

    def forward_fn(x):
        recon_x, z_tilde = model(x)
        z = sigma * ms.ops.StandardNormal()(z_tilde.shape)
        recon_loss = ms.ops.binary_cross_entropy(recon_x, x)
        mmd_loss = imq_kernel(z_tilde, z, h_dim=2)
        mmd_loss = mmd_loss / x.shape[0]
        return recon_loss, mmd_loss * mmd_lambda

    grad_fn = ms.ops.value_and_grad(forward_fn, None, optimizer.parameters)

    @ms.jit()
    def train_step(x):
        ((step_recon_loss, step_mmd_loss), grads) = grad_fn(x)
        step_loss = step_recon_loss + step_mmd_loss
        step_loss = ms.ops.depend(step_loss, optimizer(grads))
        return step_loss, step_recon_loss, step_mmd_loss / mmd_lambda

    # start model training
    loss_ = []
    for epoch in range(num_epoch):
        start_time = time.time()
        epoch_loss = []
        epoch_recon = []
        epoch_mmd = []
        model.set_train(True)
        for _, data_ in enumerate(train_iterator):
            data_x = data_['x']
            (iter_loss, iter_recon_loss, iter_mmd_loss) = train_step(data_x)
            epoch_loss.append(iter_loss.asnumpy())
            epoch_recon.append(iter_recon_loss.asnumpy())
            epoch_mmd.append(iter_mmd_loss.asnumpy())
        # loss record
        avg_loss = np.sum(epoch_loss) / len(epoch_loss)
        avg_recon = np.sum(epoch_recon) / len(epoch_recon)
        avg_mmd = np.sum(epoch_mmd) / len(epoch_mmd)
        loss_.append(avg_loss)

        # print training info
        record = '[{:03}/{:03}] Total_loss: {:.6f} Recon_loss: {:.6f}, MMD_loss:{:.6f}, time: {:.3f} sec'.format(
            epoch + 1,
            num_epoch,
            avg_loss,
            avg_recon,
            avg_mmd,
            time.time() - start_time)
        print(record)

        # save training info
        if params['save_log']:
            # save loss record
            os.write(train_record, str.encode(record + '\n'))

    # save model checkpoint
    save_model_file = str(model_name + ".ckpt")
    save_model_dir = os.path.join(folder_dir, save_model_file)
    ms.save_checkpoint(model, save_model_dir)
    # save training info
    if params['save_log']:
        # loss record saved
        train_record.close()

    # prepare test data
    sampler = ds.SequentialSampler()
    test_data = ds.NumpySlicesDataset(data={'x': raw_x[:]}, sampler=sampler)
    test_data = test_data.batch(batch_size=2)
    test_iterator = test_data.create_dict_iterator()
    # save generated latents for GM eval
    latents = get_latents(model, test_iterator)
    latents_ = pd.DataFrame(latents)
    latents_.to_csv(folder_dir + '/latents.csv', index=False)

    # visualize latent space
    if params['visualize']:
        sns.set_style('ticks')
        # assign different colors to alloy with and without Copper,
        low_cu = raw_x[:, 5] < 0.05
        low_cu_latent = latents[low_cu]
        high_cu = raw_x[:, 5] >= 0.05
        high_cu_latent = latents[high_cu]
        fig, axs = plt.subplots(figsize=(3, 3), dpi=200)
        axs.set_yticks(np.arange(-6, 8, step=2))
        axs.set_xticks(np.arange(-10, 5, step=2))
        axs.set_yticklabels(np.arange(-6, 8, step=2), fontsize=7)
        axs.set_xticklabels(np.arange(-10, 5, step=2), fontsize=7)
        for axis in ['top', 'bottom', 'left', 'right']:
            axs.spines[axis].set_linewidth(1.)
        axs.tick_params(axis='both', which='major', top=False, labeltop=False, direction='out', width=1., length=4)
        axs.tick_params(axis='both', which='major', right=False, labelright=False, direction='out', width=1., length=4)

        axs.scatter(low_cu_latent[:, 0], low_cu_latent[:, 1], c='steelblue', alpha=.55, s=8, linewidths=0,
                    label='Alloys w/o Cu')
        axs.scatter(high_cu_latent[:, 0], high_cu_latent[:, 1], c='firebrick', alpha=.65, s=14, linewidths=0,
                    marker='^', label='Alloys w/ Cu')
        handles, labels = axs.get_legend_handles_labels()
        handles = handles[::1]
        labels = labels[::1]
        legend_properties = {'size': 7.5}
        axs.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.015, 1.017), handletextpad=-0.3, frameon=False,
                   prop=legend_properties)
        fig.savefig(folder_dir + '/latents.tif', bbox_inches='tight', pad_inches=0.01)
    print('=' * 200 + '\n' + 'Training Complete! Model file saved at' + save_model_dir + '\n' + '==' * 200)
    return latents


def train_mlp(model, data, seed, params):
    ''' Train MLP ranking network'''
    # load params:
    w_decay = params['weight_decay']
    num_epoch = params['num_epoch']
    folder_dir = params['folder_dir']
    model_name = 'MLP_' + params['model_name']
    exp_name = params['exp_name']
    batch_size = int(params['batch_size'])
    lr = params['lr']
    search_params_no = int(params['no'])
    # prepare train data
    train_x, test_x, train_labels, test_labels = data
    train_data = ds.NumpySlicesDataset(data={'x': train_x, 'y': train_labels}, shuffle=True)
    train_data = train_data.batch(batch_size=batch_size)
    train_iterator = train_data.create_dict_iterator()
    # prepare save_dir for checkpoint
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
        warnings.warn('current model file not exists, please check history model training record.')
    if params['save_log']:
        flags = os.O_RDWR  | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        train_record = os.open(folder_dir + '/' + model_name + '-' + exp_name + '.txt', flags, modes)

    # prepare model training
    optimizer = ms.nn.Adam(params=model.trainable_params(), learning_rate=lr,
                           weight_decay=w_decay)

    def forward_fn(x, y):
        y_predict = model(x)
        forward_loss = (y_predict - y).square().mean()
        return forward_loss

    grad_fn = ms.ops.value_and_grad(forward_fn, None, optimizer.parameters)

    @ms.jit()
    def train_step(x, y):
        (step_loss, grads) = grad_fn(x, y)
        step_loss = ms.ops.depend(step_loss, optimizer(grads))
        return step_loss

    # start model training
    epoch_losses = []
    for epoch in range(num_epoch):
        start_time = time.time()
        iter_losses = []
        model.set_train(True)
        for _, data_ in enumerate(train_iterator):
            data_x = data_['x']
            data_y = data_['y']
            iter_loss = train_step(data_x, data_y)
            iter_losses.append(iter_loss.asnumpy())
        # train loss
        epoch_loss = np.mean(iter_losses)
        epoch_losses.append(epoch_loss)

        # eval
        # prepare test data
        test_data = ds.NumpySlicesDataset(data={'x': test_x, 'y': test_labels})
        test_data = test_data.batch(batch_size=len(test_labels))
        test_iterator = test_data.create_dict_iterator()
        model.set_train(False)
        for _, data_ in enumerate(test_iterator):
            data_x_ = data_['x']
            data_y_ = data_['y']
            y_predict_ = model(data_x_)
            # test loss
            test_loss = (y_predict_ - data_y_).square().mean()

        # print training info
        record = '[{:03}/{:03}] train loss: {:.6f} , test loss: {:.6f}, time: {:.3f} sec'.format(
            epoch + 1,
            num_epoch,
            epoch_loss,
            test_loss.asnumpy(),
            time.time() - start_time)
        print(record)
        # save training info
        if params['save_log']:
            os.write(train_record, str.encode(record + '\n'))

    # save model checkpoint
    save_model_file = str(model_name + "_{}_{}.ckpt".format(seed, search_params_no))
    save_model_dir = os.path.join(folder_dir, save_model_file)
    ms.save_checkpoint(model, save_model_dir)
    # save training info
    if params['save_log']:
        # loss record saved
        train_record.close()
    print('=' * 200 + '\n' + 'Training Complete! Model file saved at' + save_model_dir + '\n' + '==' * 200)


def train_tree(model, data, seed, params):
    ''' Train Tree ranking network'''
    # load params
    folder_dir = params['folder_dir']
    model_name = 'Tree_' + params['model_name']
    exp_name = params['exp_name']
    search_params_no = int(params['no'])

    # prepare train data
    train_features, test_features, train_labels, test_labels = data
    train_labels, test_labels = train_labels.reshape(-1), test_labels.reshape(-1)
    # prepare save_dir for checkpoint
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
        warnings.warn('current model file not exists, please check history model training record.')
    if params['save_log']:
        flags = os.O_RDWR  | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        train_record = os.open(folder_dir + '/' + model_name + '-' + exp_name + '.txt', flags, modes)

    # start model training
    model.fit(train_features, train_labels)

    # save model checkpoint
    save_model_file = str(model_name + "_{}_{}.pkl".format(seed, search_params_no))
    save_model_dir = os.path.join(folder_dir, save_model_file)
    joblib.dump(model, save_model_dir)

    # model testing
    preds = model.predict(test_features)
    test_loss = np.mean(np.square((preds - test_labels)))

    # print training info
    record = '[Gradient boosting decision tree test loss: {:.6f}'.format(test_loss)
    print(record)
    # save training info
    if params['save_log']:
        os.write(train_record, str.encode(record + '\n'))
        # loss record saved
        train_record.close()
    print('=' * 200 + '\n' + 'Training Complete! Model file saved at' + save_model_dir + '\n' + '==' * 200)
