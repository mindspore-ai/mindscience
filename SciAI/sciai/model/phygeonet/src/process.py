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
"""process for phygeonet"""
import os
import yaml

import Ofpp
import mindspore.dataset as ds

from sciai.utils import parse_arg
from .dataset import VaryGeoDataset
from .foam_ops import convert_of_mesh_to_image_structured_mesh
from .py_mesh import HcubeMesh


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def get_data(args):
    """get the training data"""
    h = 0.01
    ofbc_coord = Ofpp.parse_boundary_field(f'{args.load_data_path}/TemplateCase/30/C')
    oflowc = ofbc_coord['low']['value']
    ofupc = ofbc_coord['up']['value']
    ofleftc = ofbc_coord['left']['value']
    ofrightc = ofbc_coord['right']['value']
    left_x, left_y = ofleftc[:, 0], ofleftc[:, 1]
    low_x, low_y = oflowc[:, 0], oflowc[:, 1]
    right_x, right_y = ofrightc[:, 0], ofrightc[:, 1]
    up_x, up_y = ofupc[:, 0], ofupc[:, 1]
    nx, ny = len(low_x), len(left_x)
    my_mesh = HcubeMesh(left_x, left_y, right_x, right_y, low_x, low_y, up_x, up_y, h, args.save_fig, args.save_fig,
                        save_dir=f'{args.figures_path}/mesh.pdf', tol_mesh=1e-10, tol_joint=1)
    nvar_input, nvar_output = 2, 1
    mesh_list = []
    mesh_list.append(my_mesh)
    train_set = VaryGeoDataset(mesh_list)
    dataset = ds.GeneratorDataset(source=train_set, column_names=["JJInv", "coord", "xi", "eta", "j", "jinv",
                                                                  "dxdxi", "dydxi", "dxdeta", "dydeta"])
    dataset = dataset.batch(batch_size=args.batch_size)
    of_pic = convert_of_mesh_to_image_structured_mesh(nx, ny, f'{args.load_data_path}/TemplateCase/30/C',
                                                      [f'{args.load_data_path}/TemplateCase/30/T'], args.load_data_path)
    ofv_sb = of_pic[:, :, 2]
    return dataset, h, nvar_input, nvar_output, nx, ny, ofv_sb
