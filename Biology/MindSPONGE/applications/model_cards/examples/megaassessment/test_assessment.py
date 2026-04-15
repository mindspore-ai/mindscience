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
"""megaassessment test script"""
import os
import time
import stat
import pickle
import argparse
import numpy as np
from mindspore import context
from mindsponge import PipeLine
from mindsponge.common.protein import from_pdb_string

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--pkl_path', type=str, default="./data/T1070-D2.pkl", help='pkl path')
parser.add_argument('--pdb_path', type=str, default="./data/T1070-D2_decoy1.pdb", help='pdb_path')
parser.add_argument('--ckpt_path', type=str, default="./MEGA_Assessment.ckpt", help='pdb_path')
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument('--device_target', type=str, default="Ascend", help='device target')
args = parser.parse_args()
if args.device_target == "Ascend":
    context.set_context(device_target="Ascend", ascend_config={"precision_mode": "must_keep_origin_dtype"})
elif args.device_target == "GPU":
    context.set_context(device_target="GPU")

pipe = PipeLine(name="MEGAAssessment")
pipe.set_device_id(args.device_id)
pipe.initialize("predict_256")
pipe.model.from_pretrained(args.ckpt_path)

# load raw feature
f = open(args.pkl_path, "rb")
raw_feature = pickle.load(f)
f.close()

# load decoy pdb
with open(args.pdb_path, 'r') as f:
    decoy_prot_pdb = from_pdb_string(f.read())
    f.close()
raw_feature['decoy_aatype'] = decoy_prot_pdb.aatype
raw_feature['decoy_atom_positions'] = decoy_prot_pdb.atom_positions
raw_feature['decoy_atom_mask'] = decoy_prot_pdb.atom_mask

t0 = time.time()
_ = pipe.predict(raw_feature)
t1 = time.time()
res = pipe.predict(raw_feature)
t2 = time.time()
res = pipe.predict(raw_feature)
t3 = time.time()

total_time, execute_time0, execute_time1 = t1 - t0, t2 - t1, t3 - t2
compile_time = total_time - execute_time0
res = {
    "compile_time": compile_time,
    "execute_time": min(execute_time0, execute_time1),
    "correctness": float(np.mean(res))
}
os_flags = os.O_RDWR | os.O_CREAT
os_modes = stat.S_IRWXU
res_path = f'./MEGAAssessment_result.log'
with os.fdopen(os.open(res_path, os_flags, os_modes), 'w') as fout:
    fout.write(res)
print(res)
