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
# ==============================================================================
"""
utils
"""
import os
import numpy as np

from mindspore import ops


def check_file_path(path):
    """check file dir"""
    if not os.path.exists(path):
        os.makedirs(path)


def unpatchify(labels, img_size=(192, 384), patch_size=16, nchw=False):
    """unpatchify"""
    label_shape = labels.shape
    output_dim = label_shape[-1] // (patch_size * patch_size)
    labels = ops.Reshape()(labels, (label_shape[0],
                                    img_size[0] // patch_size,
                                    img_size[1] // patch_size,
                                    patch_size,
                                    patch_size,
                                    output_dim))

    labels = ops.Transpose()(labels, (0, 1, 3, 2, 4, 5))
    labels = ops.Reshape()(labels, (label_shape[0],
                                    img_size[0],
                                    img_size[1],
                                    output_dim))
    if nchw:
        labels = ops.Transpose()(labels, (0, 3, 1, 2))
    return labels


def get_ckpt_summ_dir(callback_params, model_name, method):
    """get ckpt and summary dir"""
    summary_dir = os.path.join(f"{callback_params['summary_dir']}/summary_{method}", model_name)
    ckpt_dir = os.path.join(summary_dir, "ckpt_dir")
    check_file_path(ckpt_dir)
    print(f'model_name: {model_name}')
    print(f'summary_dir: {summary_dir}')
    print(f'ckpt_dir: {ckpt_dir}')
    return ckpt_dir, summary_dir


def display_error(error_name, error, error_list):
    """display error"""
    print(f'mean {error_name} : {error}, max {error_name} : {max(error_list)},'
          f' average {error_name} : {np.mean(error_list)},'
          f' min {error_name} : {min(error_list)}, median {error_name} : {np.median(error_list)}'
          )


def calculate_eval_error(dataset, model, save_error=False, post_dir=None):
    """calculate evaluation error"""
    print("================================Start Evaluation================================")
    length = dataset.get_dataset_size()
    l1_error, l1_error_u, l1_error_v, l1_error_p, l1_error_cp = 0.0, 0.0, 0.0, 0.0, 0.0
    l1_error_list, l1_error_u_list, l1_error_v_list, l1_error_p_list, l1_error_cp_list, l1_avg_list = \
        [], [], [], [], [], []
    for data in dataset.create_dict_iterator(output_numpy=False):
        label, pred = get_label_and_pred(data, model)
        l1_max_step, l1_max_u_step, l1_max_v_step, l1_max_p_step, cp_max_step = calculate_max_error(label, pred)

        l1_avg = np.mean(np.mean(np.mean(np.abs(label - pred), axis=1), axis=1), axis=1).tolist()
        l1_error_list.extend(l1_max_step)
        l1_error_u_list.extend(l1_max_u_step)
        l1_error_v_list.extend(l1_max_v_step)
        l1_error_p_list.extend(l1_max_p_step)
        l1_error_cp_list.extend(cp_max_step)
        l1_avg_list.extend(l1_avg)

        l1_error_step, l1_error_u_step, l1_error_v_step, l1_error_p_step, cp_error_step = \
            calculate_mean_error(label, pred)
        l1_error += l1_error_step
        l1_error_u += l1_error_u_step
        l1_error_v += l1_error_v_step
        l1_error_p += l1_error_p_step
        l1_error_cp += cp_error_step
    l1_error /= length
    l1_error_u /= length
    l1_error_v /= length
    l1_error_p /= length
    l1_error_cp /= length
    display_error('l1_error', l1_error, l1_error_list)
    display_error('u_error', l1_error_u, l1_error_u_list)
    display_error('v_error', l1_error_v, l1_error_v_list)
    display_error('p_error', l1_error_p, l1_error_p_list)
    display_error('cp_error', l1_error_cp, l1_error_cp_list)
    if save_error:
        save_dir = os.path.join(post_dir, "ViT")
        check_file_path(save_dir)
        print(f"eval error save dir: {save_dir}")
        np.save(os.path.join(save_dir, 'l1_error_list'), l1_error_list)
        np.save(os.path.join(save_dir, 'l1_error_u_list'), l1_error_u_list)
        np.save(os.path.join(save_dir, 'l1_error_v_list'), l1_error_v_list)
        np.save(os.path.join(save_dir, 'l1_error_p_list'), l1_error_p_list)
        np.save(os.path.join(save_dir, 'l1_error_cp_list'), l1_error_cp_list)
        np.save(os.path.join(save_dir, 'l1_error_avg_list'), l1_avg_list)
    print("=================================End Evaluation=================================")


def calculate_mean_error(label, pred):
    """calculate mean l1 error"""
    l1_error = np.mean(np.abs(label - pred))
    l1_error_u = np.mean(np.abs(label[..., 0] - pred[..., 0]))
    l1_error_v = np.mean(np.abs(label[..., 1] - pred[..., 1]))
    l1_error_p = np.mean(np.abs(label[..., 2] - pred[..., 2]))
    cp_error = np.mean(np.abs(label[..., 2][0, 0, :] - pred[..., 2][0, 0, :]))
    return l1_error, l1_error_u, l1_error_v, l1_error_p, cp_error


def calculate_max_error(label, pred):
    """calculate max l1 error"""
    l1_error = np.max(np.max(np.abs(label - pred), axis=1), axis=1)
    l1_error_avg = np.mean(l1_error, axis=1).tolist()
    l1_error_u = l1_error[:, 0].tolist()
    l1_error_v = l1_error[:, 1].tolist()
    l1_error_p = l1_error[:, 2].tolist()
    cp_error = np.max(np.abs(label[..., 2][:, 0, :] - pred[..., 2][:, 0, :]), axis=1).tolist()
    return l1_error_avg, l1_error_u, l1_error_v, l1_error_p, cp_error


def save_label_and_pred(label, pred, save_img_dir):
    """save abel and pred"""
    save_dir = os.path.join(save_img_dir, 'label_pred')
    print(f'label_and_pred: {save_dir}')
    check_file_path(save_dir)
    label = unpatchify(label)
    pred = unpatchify(pred)
    np.save(os.path.join(save_dir, 'label_list'), label.asnumpy())
    np.save(os.path.join(save_dir, 'ViT'), pred.asnumpy())


def get_label_and_pred(data, model):
    """get abel and pred"""
    labels = data["labels"]
    pred = model(data['inputs'])
    pred = unpatchify(pred)
    label = unpatchify(labels)
    return label.asnumpy(), pred.asnumpy()
