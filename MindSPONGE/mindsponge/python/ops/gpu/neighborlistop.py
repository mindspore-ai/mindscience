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
"""neighborlistop"""

import os
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp

put_atom_into_bucket_add = CustomRegOp() \
    .input(0, "x0") \
    .input(1, "x1") \
    .input(2, "x2") \
    .output(0, "y0") \
    .output(1, "y1") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, \
                  DataType.I32_Default, DataType.I32_Default) \
    .target("GPU") \
    .get_op_info()

find_atom_neighbors_add = CustomRegOp() \
    .input(0, "x0") \
    .input(1, "x1") \
    .input(2, "x2") \
    .input(3, "x3") \
    .input(4, "x4") \
    .input(5, "x5") \
    .input(6, "x6") \
    .input(7, "x7") \
    .input(8, "x8") \
    .output(0, "y0") \
    .output(1, "y1") \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, \
                  DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .target("GPU") \
    .get_op_info()

delete_excluded_atoms_add = CustomRegOp() \
    .input(0, "x0") \
    .input(1, "x1") \
    .input(2, "x2") \
    .input(3, "x3") \
    .input(4, "x4") \
    .output(0, "y0") \
    .output(1, "y1") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, \
                  DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, \
                  DataType.I32_Default) \
    .target("GPU") \
    .get_op_info()

class NeighborListOP():
    """NeighborListOP"""
    def __init__(self):
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../libs/libneighborlist.so"))
        self.put_atom_path = lib_path + ":PutAtomIntoBucket"
        self.find_atom_path = lib_path + ":FindAtomNeighbors"
        self.delete_atom_path = lib_path + ":DeleteExcludedAtoms"

    def register(self, atom_numbers, grid_numbers, max_atom_in_grid_numbers, max_neighbor_numbers):
        """Register the neighbor list operator."""
        put_atom_into_bucket_op = ops.Custom(self.put_atom_path, \
            out_shape=(([grid_numbers, max_atom_in_grid_numbers], [grid_numbers,])), \
            out_dtype=(mstype.int32, mstype.int32), func_type="aot", reg_info=put_atom_into_bucket_add)
        find_atom_neighbors_op = ops.Custom(self.find_atom_path, \
            out_shape=(([atom_numbers,], [atom_numbers, max_neighbor_numbers])), \
            out_dtype=(mstype.int32, mstype.int32), func_type="aot", reg_info=find_atom_neighbors_add)
        delete_excluded_atoms_op = ops.Custom(self.delete_atom_path, \
            out_shape=(([atom_numbers,], [atom_numbers, max_neighbor_numbers])), \
            out_dtype=(mstype.int32, mstype.int32), func_type="aot", reg_info=delete_excluded_atoms_add)
        return  put_atom_into_bucket_op, find_atom_neighbors_op, delete_excluded_atoms_op
