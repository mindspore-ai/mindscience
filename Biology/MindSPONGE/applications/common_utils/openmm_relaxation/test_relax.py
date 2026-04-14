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
"""test script for openmm relaxation"""

# pylint: disable=C0413

import sys
sys.path.append("../../common_utils/")  # 将common_utils工具路径添加到环境变量中
from openmm_relaxation.run_relax import run_relax

unrelaxed_pdb_file_path = "../../model_cards/examples/MEGA-Protein/pdb/T1082-D1.pdb"
relaxed_pdb_file_path = "../../model_cards/examples/MEGA-Protein/pdb/T1082-D1_relaxed.pdb"

run_relax(unrelaxed_pdb_file_path, relaxed_pdb_file_path)
