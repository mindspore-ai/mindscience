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
"""Compute Jaccard Similarity Coefficient"""
import time
from scipy.spatial.distance import pdist, squareform
import numpy as np

NETS = ['drugProtein', 'drugsideEffect']

for i in range(1, len(NETS)):
    time_start = time.time()
    inputID = '../dataset/drugNets/' + NETS[i] + 'txt'
    M = open(inputID)
    Sim = 1 - pdist(M, 'jaccard')
    Sim = squareform(Sim)
    Sim = Sim + np.eye(np.size(M, 0))
    np.nan_to_num(Sim)
    outputID = '../dataset/drugNets/Sim_' + NETS[i] + 'txt'
    np.save(outputID, Sim, '/t')
    time_end = time.time()
    time_sum = time_end - time_start
