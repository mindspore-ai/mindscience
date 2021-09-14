# Copyright 2021 Huawei Technologies Co., Ltd
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
"""inputs process"""
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Electromagnetic Simulation')
parser.add_argument('--input_path', default='/input/path/')
parser.add_argument('--processed_input_path', default='./processed_input.npy')
parser.add_argument('--x_solution', type=int, default=50)
parser.add_argument('--y_solution', type=int, default=50)
parser.add_argument('--z_solution', type=int, default=8)
parser.add_argument('--t_solution', type=int, default=100)
parser.add_argument('--x_size', type=float, default=80*1e-2, help='x physics size')
parser.add_argument('--y_size', type=float, default=160*1e-2, help='y physics size')
parser.add_argument('--z_size', type=float, default=10*1e-2, help='z physics size')
parser.add_argument('--t_size', type=float, default=17*1e-9, help='t physics size')
parser.add_argument('--src_idx', type=tuple, default=(15, 15, 4), help='source cordinates')
parser.add_argument('--fmax', type=float, default=4e9, help='max frequency')
opt = parser.parse_args()

data = np.load(opt.input_path)
src_idx = opt.src_idx
fmax = opt.fmax
dt = opt.t_size / opt.t_solution
dx = opt.x_size / opt.x_solution
dy = opt.y_size / opt.y_solution
dz = opt.z_size / opt.z_solution
X, Y, Z, T = opt.x_solution, opt.y_solution, opt.z_solution, opt.t_solution

gridx = np.linspace(0, opt.x_size, X).astype(np.float32)
gridy = np.linspace(0, opt.y_size, Y).astype(np.float32)
gridz = np.linspace(0, opt.z_size, Z).astype(np.float32)
gridt = np.linspace(0, opt.t_size, T).astype(np.float32)
gridxx = gridx.reshape([X, 1, 1, 1]).repeat(Y, axis=1).repeat(Z, axis=2)
gridyy = gridy.reshape([1, Y, 1, 1]).repeat(X, axis=0).repeat(Z, axis=2)
gridzz = gridz.reshape([1, 1, Z, 1]).repeat(X, axis=0).repeat(Y, axis=1)

# smooth for source region
gridxyz = np.concatenate((gridxx, gridyy, gridzz), axis=-1)
src_matrix = np.zeros((X, Y, Z, 1), dtype=np.float32)
src_matrix = np.reshape(src_matrix, (X * Y * Z, 1))
gridxyz = np.reshape(gridxyz, (X * Y * Z, 3))
datax = np.linspace(0, 1.0, num=100 + 1, endpoint=True)
datay = np.linspace(0, 1.0, num=100 + 1, endpoint=True)
dataz = np.linspace(0, 1.0, num=100 + 1, endpoint=True)
std1, std2, std3 = np.std(datax) / 5, np.std(datay) / 5, np.std(dataz) / 5
mean1, mean2, mean3 = src_idx[0]*dx, src_idx[1]*dy, src_idx[2]*dz
src_matrix[:, 0] = np.exp(-(((gridxyz[:, 0] - mean1) / std1) ** 2 + ((gridxyz[:, 1] - mean2) / std2) ** 2 +
                            ((gridxyz[:, 2] - mean3) / std3) ** 2))

# binding coordinates
gridx = gridxx.reshape([1, X, Y, Z, 1]).repeat(T, axis=0)
gridy = gridyy.reshape([1, X, Y, Z, 1]).repeat(T, axis=0)
gridz = gridzz.reshape([1, X, Y, Z, 1]).repeat(T, axis=0)
gridt = gridt.reshape([T, 1, 1, 1, 1]).repeat(X, axis=1).repeat(Y, axis=2).repeat(Z, axis=3)
grid_src = np.zeros((T, X, Y, Z, 1), dtype=np.float32)
tao = np.sqrt(2.3) / (np.pi*fmax)
T0 = 3.65 * tao
for t in range(T):
    grid_src[t, src_idx[0], src_idx[1], src_idx[2], 0] = np.exp(-(t*dt-T0)**2/tao**2)

grid = np.concatenate((gridx, gridy, gridz, gridt, grid_src), axis=-1)
processed_input = np.ones((data.shape[0], T, X, Y, Z, data.shape[-1]+5), dtype=np.float32)
for idx in range(data.shape[0]):
    inputs = np.expand_dims(data[idx], 0).repeat(T, axis=0)
    print(inputs.shape)
    print(grid.shape)
    inputs = np.concatenate((grid, inputs), axis=-1)
    for i in range(T):
        src_val = inputs[i, src_idx[0], src_idx[1], src_idx[2], 4]
        src_matrix_tmp = src_matrix * src_val
        matrix_src = np.reshape(src_matrix_tmp, (X, Y, Z))
        inputs[i, :, :, :, 4] = matrix_src
    processed_input[idx] = inputs

scale_t = 1e9
scale_src = np.max([np.abs(np.max(processed_input[..., 4])), np.abs(np.min(processed_input[..., 4]))])
processed_input[..., 3] = processed_input[..., 3] * scale_t
processed_input[..., 4] = np.log10(processed_input[..., 4] / scale_src + 1.0)

# fusion time axis
processed_input = np.reshape(processed_input, (processed_input.shape[0] * T, X, Y, Z, processed_input.shape[-1]))

# save data
path = opt.processed_input_path
np.save(path, processed_input)
print("save finished!")
