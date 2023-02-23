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
"""docking main script."""

import os
import pickle
import argparse
import numpy as np

import mindspore.context as context
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.nn import DistributedGradReducer
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean, _get_parallel_mode
from mindspore.context import ParallelMode

from get_train_data import create_dataset
from model import GCN

parser = argparse.ArgumentParser()

# Add argument
parser.add_argument('--dataset_path', default='../xxx.pkl',
                    help='dataset path davis or kiba')
parser.add_argument('--ckpt_path', type=str, default="../xxx.ckpt", help='loss_scale')
parser.add_argument('--mode', type=str, default="train", help='mode, only support train and inference mode')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--device', type=str, default="GPU", help='device, GPU or Ascend')
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument('--seed', type=int, default=0, help='global seed')
parser.add_argument('--epochs', type=int, default=1, help='epochs')
parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss_scale')
parser.add_argument('--save_path', type=str,
                    default="../ms_ckpt/", help='checkpoint save path')

args = parser.parse_args()

batch_size = args.batch_size
lr = args.lr
seed = args.seed
dataset_path = args.dataset_path
epochs = args.epochs
loss_scale = args.loss_scale
save_path = args.save_path
mode = args.mode
ckpt_path = args.ckpt_path
device = args.device
device_id = args.device_id

os.makedirs(save_path, exist_ok=True)
set_seed(seed)


context.set_context(mode=context.GRAPH_MODE, device_target=device, max_device_memory="29GB", device_id=device_id)


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 0.1

clip_grad = ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """clip grad"""
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array((-clip_value,)), dt),
                                     ops.cast(ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.Reciprocal()(scale)


grad_cast = C.MultitypeFuncGraph("grad_cast")


@grad_cast.register("Tensor")
def tensor_grad_cast(grad):
    return ops.Cast()(grad, ms.float32)


class TrainOneStepCell(nn.Cell):
    """自定义训练网络"""
    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True, use_global_norm=True):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = Tensor(sens)
        self.enable_clip_grad = enable_clip_grad
        self.hyper_map = ops.HyperMap()
        self.use_global_norm = use_global_norm

        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        """construct 函數"""
        if self.train_backward:
            losses = self.network(*inputs)
            loss, logits = losses
            sens1 = F.fill(loss.dtype, loss.shape, 1.0)
            sens2 = F.fill(logits.dtype, logits.shape, 0.0)
            grads = self.grad(self.network, self.weights)(*inputs, (sens1, sens2))
            grads = self.hyper_map(F.partial(grad_scale, self.sens), grads)
            grads = self.grad_reducer(grads)

            if self.enable_clip_grad:
                if self.use_global_norm:
                    grads = C.clip_by_global_norm(grads, GRADIENT_CLIP_VALUE)
                else:
                    grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
            loss = F.depend(loss, self.optimizer(grads))
        else:
            loss, logits = self.network(*inputs)
        return loss, logits


docking_net = GCN()

opt = nn.Adam(params=docking_net.trainable_params(), learning_rate=lr, eps=1e-5)
train_net = TrainOneStepCell(docking_net, opt, sens=loss_scale)

with open(dataset_path, "rb") as f:
    input_data = pickle.load(f)

if mode == "inference":
    param_dict = load_checkpoint(ckpt_path)
    keys = list(param_dict.keys())
    for key in keys:
        if "learning_rate" in key or "global_step" in key:
            param_dict.pop(key)
    load_param_into_net(docking_net, param_dict)
    train_net.set_train(False)
    train_net.add_flags_recursive(train_backward=False)
    train_index_all_final = list(range(len(input_data)))

else:
    train_net.set_train(True)
    train_net.add_flags_recursive(train_backward=True)
    index_all = list(range(len(input_data)))
    train_index_all_final = []
    np.random.seed(seed)
    for i in range(epochs):
        np.random.shuffle(index_all)
        train_index_all_final.extend(index_all)

train_dataset = create_dataset(batch_size, input_data, train_index_all_final, num_parallel_worker=4)
dataset_iter = train_dataset.create_dict_iterator(num_epochs=epochs, output_numpy=True)

for step, d in enumerate(dataset_iter):
    inputs_feats = np.array(d["x_feature"], np.float32), \
                   np.array(d["edge_feature"], np.int32), \
                   np.array(d["target_feature"], np.int32), \
                   np.array(d["batch_info"], np.int32), \
                   np.array(d["label"], np.float32)
    index_all = d["index_all"]
    inputs_feat = [Tensor(feat) for feat in inputs_feats]
    loss_out, logits_out = train_net(*inputs_feat)
    if mode == "train":
        print("loss is: ", loss_out, "step is: ", step)
        if step % 10 == 0:
            save_checkpoint(train_net, save_path + f"/ms_gcn_{step}.ckpt")
    else:
        print("predict is: ", logits_out, "step is: ", step)
