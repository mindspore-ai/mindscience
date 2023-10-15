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

"""plot"""
import numpy as np
import matplotlib.pyplot as plt

def plot_your_figure(args, output, gamma_list):
    plt.hist(gamma_list, bins=100, color='blue', weights=np.zeros_like(gamma_list)+1/len(gamma_list), edgecolor="black")
    plt.title('The performance of output <'+ output +'>')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Probability')
    plt.gca().invert_xaxis()
    if args.save_fig:
        plt.savefig(args.figures_path +'/' + output + '.eps', bbox_inches='tight')
