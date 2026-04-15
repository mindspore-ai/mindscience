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
"""datasets"""
from mindspore.ops import deepcopy
from mindspore import Tensor
import numpy as np
import mindspore as ms
import mindspore.ops as ops

def get_iso_permuted_dataset(picklefile, amberFlag=0, **atm_iso):
    import math
    import pickle
    import numpy as np
    import mindspore as ms
    from mindspore import Tensor

    dataset = []

    for key, value in atm_iso.items():
        if key == 'h_iso':
            h_data = np.loadtxt(value, skiprows=2, usecols=1)
        elif key == 'c_iso':
            c_data = np.loadtxt(value, skiprows=2, usecols=1)
        elif key == 'n_iso':
            n_data = np.loadtxt(value, skiprows=2, usecols=1)
        elif key == 'o_iso':
            o_data = np.loadtxt(value, skiprows=2, usecols=1)
        elif key == 'p_iso':
            p_data = np.loadtxt(value, skiprows=2, usecols=1)
        else:
            raise ValueError("Isolated atom type not found. Use kwargs \"h_iso\", \"c_iso\", etc.")

    with open(picklefile, "rb") as f:
        molecules = pickle.load(f)

    for idx, molecule in enumerate(molecules):
        pos    = molecule['pos']
        z      = np.expand_dims(molecule['type'], axis=1)
        x      = molecule['onehot']
        c      = molecule['coefficients']
        n      = molecule['norms']
        exp    = molecule['exponents']
        full_c = c.copy()
        iso_c  = np.zeros_like(c)

        if amberFlag==1:
            amber_chg = molecule['amber_chg']

        # Subtract the isolated atoms
        for atom, iso, typ in zip(c, iso_c, z):
            typ_value = typ.item()
            if typ_value == 1.0:
                atom[:h_data.shape[0]] -= h_data
                iso[:h_data.shape[0]] += h_data
            elif typ_value == 6.0:
                atom[:c_data.shape[0]] -= c_data
                iso[:c_data.shape[0]] += c_data
            elif typ_value == 7.0:
                atom[:n_data.shape[0]] -= n_data
                iso[:n_data.shape[0]] += n_data
            elif typ_value == 8.0:
                atom[:o_data.shape[0]] -= o_data
                iso[:o_data.shape[0]] += o_data
            elif typ_value == 15.0:
                atom[:p_data.shape[0]] -= p_data
                iso[:p_data.shape[0]] += p_data
            else:
                raise ValueError("Isolated atom type not supported!")

        pop = np.where(n != 0, c * 2 * math.sqrt(2) / n, n)

        # Permute positions, yzx -> xyz
        p_pos = pos.copy()
        p_pos[:, 0] = pos[:, 1]
        p_pos[:, 1] = pos[:, 2]
        p_pos[:, 2] = pos[:, 0]

        # Create dataset dictionary with Tensor
        if amberFlag==1:
            data_dict = {
                'pos':       Tensor(p_pos, dtype=ms.float32),
                'pos_orig':  Tensor(pos, dtype=ms.float32),
                'z':         Tensor(z, dtype=ms.float32),
                'x':         Tensor(x, dtype=ms.float32),
                'y':         Tensor(pop, dtype=ms.float32),
                'c':         Tensor(c, dtype=ms.float32),
                'full_c':    Tensor(full_c, dtype=ms.float32),
                'iso_c':     Tensor(iso_c, dtype=ms.float32),
                'exp':       Tensor(exp, dtype=ms.float32),
                'norm':      Tensor(n, dtype=ms.float32),
                'amber_chg': Tensor(amber_chg, dtype=ms.float32)
            }
        else:
            data_dict = {
                'pos':      Tensor(p_pos, dtype=ms.float32),
                'pos_orig': Tensor(pos, dtype=ms.float32),
                'z':        Tensor(z, dtype=ms.float32),
                'x':        Tensor(x, dtype=ms.float32),
                'y':        Tensor(pop, dtype=ms.float32),
                'c':        Tensor(c, dtype=ms.float32),
                'full_c':   Tensor(full_c, dtype=ms.float32),
                'iso_c':    Tensor(iso_c, dtype=ms.float32),
                'exp':      Tensor(exp, dtype=ms.float32),
            'norm':         Tensor(n, dtype=ms.float32)
            }

        dataset.append(data_dict)

    print(f"Loaded {len(dataset)} molecules from {picklefile}")

    return dataset


def get_iso_dataset(picklefile, **atm_iso):
    import math
    import pickle
    import numpy as np
    import mindspore as ms
    import mindspore.numpy as mnp
    from mindspore import Tensor

    dataset = []

    for key, value in atm_iso.items():
        if key == 'h_iso':
            h_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        elif key == 'c_iso':
            c_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        elif key == 'n_iso':
            n_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        elif key == 'o_iso':
            o_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        elif key == 'p_iso':
            p_data = Tensor(np.loadtxt(value, skiprows=2, usecols=1), dtype=ms.float32)
        else:
            raise ValueError("Isolated atom type not found. Use kwargs \"h_iso\", \"c_iso\", etc.")

    with open(picklefile, "rb") as f:
        molecules = pickle.load(f)
    
    cnt = 0
    for molecule in molecules:
        pos = Tensor(molecule['pos'], dtype=ms.float32)
        # z is atomic number- may want to make 1,0
        z = Tensor(np.expand_dims(molecule['type'], axis=1), dtype=ms.float32)
        x = Tensor(molecule['onehot'], dtype=ms.float32)
        c = Tensor(molecule['coefficients'], dtype=ms.float32)
        n = Tensor(molecule['norms'], dtype=ms.float32)
        exp = Tensor(molecule['exponents'], dtype=ms.float32)
        energy = Tensor(molecule['energy'], dtype=ms.float32)
        # this is a gradient, not forces
        # convert from hartree/bohr to kcal/mol/ang
        bohr2ang = 0.529177
        hartree2kcal = 627.5094740631
        forces = molecule['forces']*hartree2kcal/bohr2ang

        full_c = deepcopy(c)

        # Subtract the isolated atoms
        for atom, typ in zip(c, z):
            typ_value = typ.asnumpy().item()
            if typ_value == 1.0:
                atom[:h_data.shape[0]] -= h_data
            elif typ_value == 6.0:
                atom[:c_data.shape[0]] -= c_data
            elif typ_value == 7.0:
                atom[:n_data.shape[0]] -= n_data
            elif typ_value == 8.0:
                atom[:o_data.shape[0]] -= o_data
            elif typ_value == 15.0:
                atom[:p_data.shape[0]] -= p_data
            else:
                raise ValueError("Isolated atom type not supported!")

        pop = mnp.where(n != 0, c * 2 * math.sqrt(2) / n, n)

        # Create dataset dictionary
        data_dict = {
            'pos': pos,
            'pos_orig': pos,
            'z': z,
            'x': x,
            'y': pop,
            'c': c,
            'full_c': full_c,
            'exp': exp,
            'norm': n,
            'energy': energy,
            'forces': forces
        }

        dataset.append(data_dict)
        cnt += 1
        
    print(f"Loaded {cnt} molecules from {picklefile}")
    print(f"Loaded {len(dataset)} molecules from {picklefile}")

    return dataset
