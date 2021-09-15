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
"""generate cloud data"""
import argparse
import numpy as np
import mindspore_data.data as md
from mindspore_data.data.scientific_compute import SamplingMode
from mindspore_data.data.scientific_compute import BBoxType
from mindspore_data.data.scientific_compute import StdPhysicalQuantity


def generate_cloud_data():
    """generate cloud data"""
    config = md.PointCloudSamplingConfig(SamplingMode.DIMENSIONS, BBoxType.STATIC, opt.sample_nums, opt.bbox_args)

    std_physical_info = {
        StdPhysicalQuantity.MU: 1.0,
        StdPhysicalQuantity.EPSILON: 1.0,
        StdPhysicalQuantity.SIGMA: 0.,
        StdPhysicalQuantity.TAND: 0.,
    }

    material_config = md.MaterialConfig(opt.json_path, opt.material_dir, std_physical_info, None, False)

    pointcloud = md.PointCloud(opt.stp_path, config, material_config)
    pointcloud.topology_solving()
    inner_point_dict = pointcloud.tensor_build()

    res = inner_point_dict[:, :, :, 4:]
    np.save(opt.save_path, res, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stp_path', type=str, default='stp path',
                        help='the path of folder containing STP files')
    parser.add_argument('--json_path', type=str, default='json path',
                        help='the path of json file')
    parser.add_argument('--material_dir', type=str, default='material dir',
                        help='the path of folder containing material information')
    parser.add_argument('--save_path', type=str, default="/data6/htc/tmp.npy",
                        help='save path of generated point cloud data')
    parser.add_argument('--sample_nums', type=tuple, default=(500, 2000, 80), help='sampling number on each axis')
    parser.add_argument('--bbox_args', type=tuple, default=(-40., -80., -5., 40., 80., 5.),
                        help='the location of bbox, (x_min, y_min, z_min, x_max, y_max, z_max)')

    opt = parser.parse_args()
    generate_cloud_data()
