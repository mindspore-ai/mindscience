# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train deepfri model"""
import os
import json
import warnings
import time
import argparse

import numpy as np
import mindspore as ms
import mindspore.nn as nn

from utils import load_go_annot, load_ec_annot, datapipe, MyWithLossCell, plot_losses
from model.deepfri import DeepFRI


def train_loop(net, train_network, eval_network, train_dataset, val_dataset, current_epoch, epochs,
               save_model_path="./output"):
    """
        Train Loop
        net: DeepFRI network. It is used to save weights or checkpoints.
        train_network: MyWithLossCell for training. Set train True.
        eval_network: MyWithLossCell for valuing. Set train False.
        val_dataset: Val Dataset.Inherit from ms.dataset.GeneratorDataset.
        current_epoch: Current epoch.
        epochs: Total epochs.
        save_model_pat: Trained models will be saved in the folder named checkpoints under this path.
    """
    save_model_path = os.path.join(save_model_path, "checkpoints")
    train_steps = train_dataset.get_dataset_size()
    val_steps = val_dataset.get_dataset_size()

    # train part
    train_init_t = time.time()
    step = 1
    train_loss = 0
    for d in train_dataset.create_dict_iterator():
        start_time = time.time()
        result = train_network(d["cmap"], d["seq"], d["labels"]).asnumpy()

        print(f"Epoch: [{current_epoch} / {epochs}], "
              f"step: [{step} / {train_steps}], "
              f"time:[{time.time() - start_time:.2f}s],"
              f"train_loss: {result}")
        train_loss += result
        step = step + 1
    mean_train_loss = train_loss / step
    print(f"epoch {current_epoch}: mean_train_loss:{mean_train_loss}")
    train_finish_t = time.time()
    print(f"train_time:{train_finish_t - train_init_t}")

    # val part
    val_init_t = time.time()
    step = 1
    val_loss = 0
    for d in val_dataset.create_dict_iterator():
        result = eval_network(d["cmap"], d["seq"], d["labels"]).asnumpy()
        print(f"Epoch: [{current_epoch} / {epochs}], "
              f"step: [{step} / {val_steps}], "
              f"val_loss: {result}")
        val_loss += result
        step = step + 1
    mean_val_loss = val_loss / step
    print(f"epoch {current_epoch}: mean_val_loss:{mean_val_loss}")
    val_finish_t = time.time()
    print(f"eval_time:{val_finish_t - val_init_t}")
    # save checkpoint
    print('Saving model (epoch = {}, mean_val_loss = {:.4f})'.format(current_epoch + 1, mean_train_loss))
    ms.save_checkpoint(net,
                       save_model_path +
                       f"/train_model_e{current_epoch + 1}_t{val_finish_t - val_init_t:4f}_l{mean_train_loss:.4f}.ckpt")

    return mean_train_loss, mean_val_loss


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Number of epochs to train.")
    parser.add_argument('-ont', '--ontology', type=str, default='mf', choices=['mf', 'bp', 'cc', 'ec'],
                        help="Ontology.")
    parser.add_argument('-device', '--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('-id', '--device_id', type=int, default=6,
                        help='device id where the data will be set in (default: 0)')
    parser.add_argument('-out', '--output_dir', type=str, default="./output",
                        help='trained models will be saved in the folder named checkpoints under this path')
    parser.add_argument('-pcp', '--pretrained_ckpt_path', type=str,
                        help='breakpoint training path')
    args = parser.parse_args()
    ms.set_context(device_target=args.device_target, device_id=args.device_id, mode=ms.GRAPH_MODE)
    print("Device target : {}\nDevice id : {}\nDevice target : {}".
          format(args.device_target, args.device_id, ms.GRAPH_MODE))
    args = parser.parse_args()

    MODEL_CONFIG = './config/model_config.json'
    with open(MODEL_CONFIG) as json_file:
        params = json.load(json_file)
    params = params['gcn']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"))
    else:
        if not os.path.exists(os.path.join(args.output_dir, "checkpoints")):
            os.makedirs(os.path.join(args.output_dir, "checkpoints"))
    print("Your checkpoints will be save in", os.path.join(args.output_dir, "checkpoints"))

    with open(params['configs'][args.ontology] + '_model_params.json') as json_file:
        metadata = json.load(json_file)

    if args.ontology == 'ec':
        prot2annot, go_terms, go_names, counts = load_ec_annot(metadata['annot_fn'])
    else:
        prot2annot, go_terms, go_names, counts = load_go_annot(metadata['annot_fn'])
    go_terms = go_terms[args.ontology]
    go_names = go_names[args.ontology]
    n_go_terms = metadata['output_dim']

    # computing weights for imbalanced go classes
    class_sizes = counts[args.ontology]
    mean_class_size = np.mean(class_sizes)

    train_tfrecord_fn = metadata['train_tfrecord_fn'] + '*'
    valid_tfrecord_fn = metadata['valid_tfrecord_fn'] + '*'
    pad_len = metadata['pad_len']

    batch_size = metadata['batch_size']
    print("Pad length is", pad_len)

    print("Preparing Dataset:Loading...")
    T1 = time.time()
    train_set = datapipe(train_tfrecord_fn, metadata['cmap_type'], n_go_terms, metadata["cmap_thresh"], channels=26,
                         ont=args.ontology, batch_size=batch_size, pad_len=pad_len)
    val_set = datapipe(valid_tfrecord_fn, metadata['cmap_type'], n_go_terms, metadata["cmap_thresh"], channels=26,
                       ont=args.ontology, batch_size=batch_size, pad_len=pad_len)
    print(f"Preparing Dataset:Finish and cost {time.time() - T1:.2f}s")

    print("Model initing:Loading...")
    T2 = time.time()
    network = DeepFRI(metadata['input_dim'], metadata['output_dim'], metadata['gc_dims'],
                      metadata['fc_dims'], metadata['dropout'], train=True, lstm_input_dim=512)
    if args.device_target == "Ascend":
        network.lm_dense.to_float(ms.float16)
        network.aa_dense.to_float(ms.float16)
        network.en_dense.to_float(ms.float16)
        network.func_predictor.output_layer.to_float(ms.float16)

    if args.pretrained_ckpt_path:
        param_dict = ms.load_checkpoint(args.pretrained_ckpt_path)
        unload_param = ms.load_param_into_net(network, param_dict)
        print(f"there are params loading failed in whole model\n{unload_param}")
    else:
        param_dict = ms.load_checkpoint(metadata['lm_model_name'])
        unload_param = ms.load_param_into_net(network.lstm, param_dict)
        print(f"there are params loading failed in lstm layer\n{unload_param}")
        print([v.shape for v in param_dict.values()])

    for param in network.get_parameters():
        param_name = param.name
        if "lstm" in param_name:
            param.requires_grad = False

    net_opt = nn.Adam(network.trainable_params(),
                      learning_rate=metadata['lr'], beta1=0.99, beta2=0.99, eps=1e-7, weight_decay=metadata['l2_reg'])

    print("### Training model: {} on {} GO terms".format(args.ontology, metadata['output_dim']))

    net_with_loss = MyWithLossCell(network)
    if args.device_target == "Ascend":
        train_net = nn.TrainOneStepCell(net_with_loss, net_opt, sens=1024.0)
    else:
        train_net = nn.TrainOneStepCell(net_with_loss, net_opt)

    train_net.set_train()

    eval_net = MyWithLossCell(network)
    eval_net.set_train(False)
    print(f"Model Init:Finish and cost {time.time() - T2:.2f}s")

    total_loss = dict()
    total_loss["train_loss"] = []
    total_loss["val_loss"] = []

    PATIENCE_EPOCHS = 5
    train_start_time = time.time()
    for epoch in range(args.epochs):
        print(ms.get_context("mode"))
        mean_train_losses, mean_val_losses = train_loop(network, train_net, eval_net,
                                                        train_set, val_set, epoch, args.epochs, args.output_dir)
        try:
            total_loss["train_loss"].append(mean_train_losses)
            total_loss["val_loss"].append(mean_val_losses)

            # early stopping
            if len(total_loss["val_loss"]) > PATIENCE_EPOCHS and \
                    total_loss["val_loss"][-1] >= total_loss["val_loss"][-PATIENCE_EPOCHS]:
                break

        except KeyError:
            print("KeyError")
    train_end_time = time.time()
    print(f"Training costs: {train_end_time - train_start_time:.2f}s")
    plot_losses(total_loss, save_path=args.output_dir)
