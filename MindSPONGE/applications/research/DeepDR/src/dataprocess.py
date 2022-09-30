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
"""datapreprocess"""
import numpy as np
import scipy.sparse as sp
from randsurf import randsurf
from getppmimatrix import getppmimatrix

with open('/DeepDR/dataset/drugNets/drugsimWmnet.txt') as file_object:
    data = file_object.readlines()
    for i, line_str in enumerate(data):
        line_data = line_str.split('\t')
        line_data = list(map(float, line_data))
        data[i] = line_data
data = np.array(data)
KSTEP = 3
ALPHA = 0.98

# Step 1. Randomly Surf to Generate K steps Transition Matrix
MK = randsurf(data, KSTEP, ALPHA)

# Step 2. Get PPMI Matrix
PPMI = getppmimatrix(MK)

NET = sp.csc_matrix(PPMI)
np.save('/deepDR/dataset/PPMI/drug_net_x.npy', NET)
