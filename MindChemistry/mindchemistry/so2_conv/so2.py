# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
so2 file
"""
import mindspore as ms
from mindspore import ops, nn
from mindchemistry.e3.o3 import Irreps


class Silu(nn.Cell):
    """
    silu activation class
    """

    def __init__(self):
        super(Silu, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        """
        silu activation class construct process
        """
        return ops.mul(x, self.sigmoid(x))


class SO2MConvolution(nn.Cell):
    """
    SO2 Convolution subnetwork
    """

    def __init__(self, in_channels, out_channels):
        super(SO2MConvolution, self).__init__()
        self.fc = nn.Dense(in_channels // 2, out_channels,
                           has_bias=False).to_float(ms.float16)
        self.out_channels = out_channels

    def construct(self, x_m):
        """
        SO2 Convolution sub network construct process
        """
        x_m = self.fc(x_m).astype(ms.float32)
        x_i = ops.narrow(x_m, 2, 0, self.out_channels // 2)
        x_r = ops.narrow(x_m, 2, self.out_channels // 2,
                         self.out_channels // 2)

        x_m_r = ops.narrow(x_r, 1, 1, 1) - ops.narrow(
            x_i, 1, 0, 1)  # x_r[:, 1] - x_i[:, 0]
        x_m_i = ops.narrow(x_i, 1, 1, 1) + ops.narrow(
            x_r, 1, 0, 1)  # x_i[:, 1] + x_r[:, 0]

        x_out = ops.cat((x_m_i, x_m_r), axis=1)
        return x_out


class SO2Convolution(nn.Cell):
    """
    SO2 Convolution network
    """

    def __init__(self, irreps_in, irreps_out):
        super().__init__()

        self.irreps_in1 = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)

        self.max_order_in = -1
        for mulir in self.irreps_in1:
            self.max_order_in = max(self.max_order_in, mulir.ir.l)
        self.max_order_out = -1
        for mulir in self.irreps_out:
            self.max_order_out = max(self.max_order_out, mulir.ir.l)

        self.m_shape_dict_in, self.irreps_in1_length = self.get_m_info(
            self.irreps_in1, self.max_order_in)
        self.m_shape_dict_out, self.irreps_out_length = self.get_m_info(
            self.irreps_out, self.max_order_out)

        self.fc_m0 = nn.Dense(self.m_shape_dict_in.get(0, None),
                              self.m_shape_dict_out.get(0, None)).to_float(ms.float16)

        self.global_max_order = min(self.max_order_in + 1,
                                    self.max_order_out + 1)

        self.so2_m_conv = nn.CellList([])
        for i in range(self.global_max_order):
            if i == 0:
                continue
            so2_m_convolution = SO2MConvolution(self.m_shape_dict_in.get(i, None),
                                                self.m_shape_dict_out.get(i, None))
            self.so2_m_conv.append(so2_m_convolution)

        self.max_m_in = 2 * self.max_order_in + 1

        self.irreps_out_data = []
        for mulir in self.irreps_out:
            key = mulir.ir.l
            value = mulir.mul
            self.irreps_out_data.append((key, value))

    def get_m_info(self, irreps, max_order):
        """
        helper function to get m_info
        """
        m_shape_dict = {}
        m_mul_ir_l_dict = {}
        for i in range(max_order + 1):
            m_shape_dict[i] = 0

        for mulir in irreps:
            mul = mulir.mul
            ir_l = mulir.ir.l
            if ir_l not in m_mul_ir_l_dict:
                m_mul_ir_l_dict[ir_l] = mul
            else:
                m_mul_ir_l_dict[ir_l] = m_mul_ir_l_dict[ir_l] + mul
            for j in range(mulir.ir.l + 1):
                if j == 0:
                    m_shape_dict[j] = m_shape_dict[j] + mul
                else:
                    m_shape_dict[j] = m_shape_dict[j] + 2 * mul

        return m_shape_dict, len(irreps)

    def get_m_list_merge(self, x):
        """
        helper function to get m_list_merge
        """
        m_list = []
        for _ in range(self.max_m_in):
            m_list.append([])

        index_shifting = int((self.max_m_in - 1) / 2)
        for tmp in x:
            m_length = tmp.shape[-1]
            m_shift = int((m_length - 1) / 2)
            for j in range(m_length):
                m_list[j - m_shift + index_shifting].append(tmp[:, :, j])

        m_list_merge = []
        for i in range(index_shifting + 1):
            if i == 0:
                m_list_merge.append(ops.cat(m_list[index_shifting - i], -1))
            else:
                m_list_merge.append(
                    ops.cat((ops.cat(m_list[index_shifting - i], -1),
                             ops.cat(m_list[index_shifting + i], -1)), -1))
        return m_list_merge

    def construct(self, x, x_edge):
        """
        SO2 Convolution network construct process
        """
        ##################### _m_primary #########################
        num_edges = ops.shape(x_edge)[0]
        m_list_merge = self.get_m_list_merge(x)
        # ##################### finish _m_primary #########################
        # radial function
        out = []

        ### Compute m=0 coefficients separately since they only have real values
        x_0 = m_list_merge[0]

        x_0 = self.fc_m0(x_0).astype(ms.float32)
        out.append(x_0)

        #### Compute the values for the m > 0 coefficients
        for m in range(self.global_max_order):
            if m == 0:
                continue
            x_m = m_list_merge[m]
            x_m = x_m.reshape(num_edges, 2, -1)
            x_m = self.so2_m_conv[m - 1](x_m)
            out.append(x_m)

        ###################### start fill 0  ######################
        if self.max_order_out + 1 > len(m_list_merge):
            for m in range(len(m_list_merge), self.max_order_out + 1):
                extra_zero = ops.zeros(
                    (num_edges, 2, int(self.m_shape_dict_out.get(m, None) / 2)))
                out.append(extra_zero)
        ###################### finish fill 0  ######################

        ###################### start _l_primary #########################
        l_primary_list_0 = []
        l_primary_list_left = []
        l_primary_list_right = []

        for _ in range(self.irreps_out_length):
            l_primary_list_0.append([])
            l_primary_list_left.append([])
            l_primary_list_right.append([])

        m_0 = out[0]
        offset = 0
        index = 0

        for key_val in self.irreps_out_data:
            key = key_val[0]
            value = key_val[1]
            if key >= 0:
                l_primary_list_0[index].append(
                    ops.unsqueeze(m_0[:, offset:offset + value], -1))
                offset = offset + value
            index = index + 1

        for m in range(1, len(out)):
            right = out[m][:, 1]
            offset = 0
            index = 0

            for key_val in self.irreps_out_data:
                key = key_val[0]
                value = key_val[1]
                if key >= m:
                    l_primary_list_right[index].append(
                        ops.unsqueeze(right[:, offset:offset + value], -1))
                    offset = offset + value
                index = index + 1

        for m in range(len(out) - 1, 0, -1):
            left = out[m][:, 0]
            offset = 0
            index = 0

            for key_val in self.irreps_out_data:
                key = key_val[0]
                value = key_val[1]
                if key >= m:
                    l_primary_list_left[index].append(
                        ops.unsqueeze(left[:, offset:offset + value], -1))
                    offset = offset + value
                index = index + 1

        l_primary_list = []
        for i in range(self.irreps_out_length):
            if i == 0:
                tmp = ops.cat(l_primary_list_0[i], -1)
                l_primary_list.append(tmp)
            else:
                tmp = ops.cat(
                    (ops.cat((ops.cat(l_primary_list_left[i],
                                      -1), ops.cat(l_primary_list_0[i], -1)),
                             -1), ops.cat(l_primary_list_right[i], -1)), -1)
                l_primary_list.append(tmp)

        ##################### finish _l_primary #########################
        return tuple(l_primary_list)
