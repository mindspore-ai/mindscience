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

"""deeponet plot"""
import os.path

import matplotlib.pyplot as plt


def save_loss_fig(args, ii, losst, lossv, lossv0):
    """save loss figures"""
    fig = plt.figure()
    plt.semilogy(ii, losst, "r", label="Training loss")
    plt.semilogy(ii, lossv, "b", label="Test loss")
    plt.semilogy(ii, lossv0, "b", label="Test loss0")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Training and test")
    plt.legend()
    plt.savefig(f"{args.figures_path}/Training_test0.png", dpi=300)
    plt.tight_layout()
    plt.close(fig)
    fig = plt.figure()
    plt.semilogy(ii, losst, "r", label="Training loss")
    plt.semilogy(ii, lossv, "b", label="Test loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Training and test")
    plt.legend()
    plt.savefig(f"{args.figures_path}/Training_test.png", dpi=300)
    plt.tight_layout()
    plt.close(fig)


def plot_prediction(y_pred, y_test, args):
    """plot prediction"""
    fig = plt.figure()
    plt.plot(y_pred.asnumpy(), y_test, "r.", y_test, y_test, "b:")
    plt.savefig(os.path.join(args.figures_path, "prediction.png"), dpi=300)
    plt.close(fig)
