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
"""train process"""
import time
import argparse

from mindspore import nn, ops
import mindspore as ms

from mindflow.utils import load_yaml_config

from src.boltzmann import BoltzmannBGK, BoltzmannFSM
from src.utils import get_vdis, visual, get_potential, get_mu, get_kn_bzm
from src.cells import SplitNet
from src.dataset import Wave1DDataset

ms.set_seed(0)

parser = argparse.ArgumentParser(description="boltzmann train")
parser.add_argument(
    "--mode",
    type=str,
    default="GRAPH",
    choices=["GRAPH", "PYNATIVE"],
    help="Running in GRAPH_MODE OR PYNATIVE_MODE",
)
parser.add_argument(
    "--save_graphs",
    type=bool,
    default=False,
    choices=[True, False],
    help="Whether to save intermediate compilation graphs",
)
parser.add_argument("--save_graphs_path", type=str, default="./graphs")
parser.add_argument(
    "--device_target",
    type=str,
    default="GPU",
    choices=["GPU", "Ascend"],
    help="The target device to run, support 'Ascend', 'GPU'",
)
parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
parser.add_argument("--config_file_path", type=str, default="./WaveD1V3.yaml")


def train():
    """train process"""
    args = parser.parse_args()
    config = load_yaml_config(args.config_file_path)

    ms.set_context(
        mode=ms.context.GRAPH_MODE
        if args.mode.upper().startswith("GRAPH")
        else ms.context.PYNATIVE_MODE,
        save_graphs=args.save_graphs,
        save_graphs_path=args.save_graphs_path,
        device_target=args.device_target,
        device_id=args.device_id,
    )

    dataset = Wave1DDataset(config)
    vdis, _ = get_vdis(config["vmesh"])
    if config["collision"] == "BGK":
        model = SplitNet(2, config["model"]["layers"], config["model"]["neurons"], vdis)
        problem = BoltzmannBGK(model, config["kn"], config["vmesh"])
    elif config["collision"] == "FSM":
        model = SplitNet(2, config["model"]["layers"], config["model"]["neurons"], vdis)
        omega = config["omega"]
        alpha = get_potential(omega)
        mu_ref = get_mu(alpha, omega, config["kn"])
        kn_bzm = get_kn_bzm(alpha, mu_ref)
        problem = BoltzmannFSM(model, kn_bzm, config["vmesh"])
    elif config["collision"] == "LR":
        raise ValueError
    elif config["collision"] == "LA":
        raise ValueError
    else:
        raise ValueError

    cosine_decay_lr = nn.CosineDecayLR(
        config["optim"]["lr_scheduler"]["min_lr"],
        config["optim"]["lr_scheduler"]["max_lr"],
        config["optim"]["Adam_steps"],
    )
    optim = nn.Adam(params=problem.trainable_params(), learning_rate=cosine_decay_lr)

    grad_fn = ops.value_and_grad(problem, None, optim.parameters, has_aux=True)

    @ms.jit
    def train_step(*inputs):
        loss, grads = grad_fn(*inputs)
        optim(grads)
        return loss

    for i in range(1, config["optim"]["Adam_steps"] + 1):
        time_beg = time.time()
        ds = dataset()
        loss, _ = train_step(*ds)
        if i % 100 == 0:
            e_sum = loss.mean().asnumpy().item()
            print(
                f"epoch: {i} train loss: {e_sum:.3e} epoch time: {(time.time() - time_beg) * 1000 :.3f}ms"
            )
    ms.save_checkpoint(problem, f'./model_{config["collision"]}_kn{config["kn"]}.ckpt')
    visual(
        problem,
        config["visual_resolution"],
        f'Wave_{config["collision"]}_kn{config["kn"]}.png',
    )


if __name__ == "__main__":
    start_time = time.time()
    train()
    print("End-to-End total time: {} s".format(time.time() - start_time))
