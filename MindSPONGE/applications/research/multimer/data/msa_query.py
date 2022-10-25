# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
MSA query tools.
"""

import os


class MmseqQuery:
    """Runs the alignment tools"""

    def __init__(self,
                 database_envdb_dir,
                 mmseqs_binary,
                 uniref30_path,
                 result_path,
                 msa_search_sh=os.path.join(os.path.dirname(__file__),
                                            "msa_search.sh")):
        """Search the a3m info for a given FASTA file."""

        self.database_envdb_dir = database_envdb_dir
        self.mmseqs_binary = mmseqs_binary
        self.uniref30_path = uniref30_path
        self.result_path = result_path
        self.msa_search_sh = msa_search_sh

    @staticmethod
    def get_a3mlines(a3m_paths):
        """combine a3m files together"""
        a3m_lines = {}
        for a3m_file in a3m_paths:
            update_m, m = True, None
            with open(a3m_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_m = True
                if line.startswith(">") and update_m:
                    try:
                        m = int(line.strip()[-1])
                    except ValueError:
                        m = str(line.strip()[-1])
                    update_m = False
                    if m not in a3m_lines:
                        a3m_lines[m] = []
                a3m_lines.get(m).append(line)
        a3m_lines = ["".join(a3m_lines.get(key)) for key in a3m_lines]
        return a3m_lines[0]

    def msa_query(self, fasta_path, a3m_root_path):
        """main entry for msa_query"""
        result_path = os.path.join(a3m_root_path, fasta_path.split(".fasta")[0])
        os.makedirs(result_path, exist_ok=True)
        if self.database_envdb_dir:
            command = f"sh {self.msa_search_sh} {self.mmseqs_binary} " + fasta_path + " " + result_path + " " + \
                      self.uniref30_path + " " + "\"\"" + " " + self.database_envdb_dir + " \"1\" \"0\" \"1\""
        else:
            command = f"sh {self.msa_search_sh} {self.mmseqs_binary} " + fasta_path + " " + result_path + " " + \
                      self.uniref30_path + " " + "\"\"" + " \"\"" + " \"0\" \"0\" \"1\""
        os.system(command)
        a3m_file_path = os.listdir(result_path)
        a3m_file_path = [os.path.join(result_path, x) for x in a3m_file_path if x.endswith("a3m")]
        return a3m_file_path

    def aligned_a3m_files(self, input_fasta_path, result_path):
        """Runs alignment tools on the input sequence and creates features."""

        a3m_file_paths = self.msa_query(fasta_path=input_fasta_path, a3m_root_path=result_path)
        a3m_lines = self.get_a3mlines(a3m_paths=a3m_file_paths)

        return a3m_lines
