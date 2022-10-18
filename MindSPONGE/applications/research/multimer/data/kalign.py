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
Kalign tools.
"""

import os
import stat
import subprocess

from absl import logging
from data.utils import tmpdir_manager, timing


class Kalign:
    """Python wrapper of the Kalign binary."""

    def __init__(self, *, binary_path: str):
        """Initializes the Python Kalign wrapper.
        Cited from "https://github.com/deepmind/alphafold"
        Args:
          binary_path: The path to the Kalign binary.
        """
        self.binary_path = binary_path

    @staticmethod
    def to_a3m(sequences):
        """Converts sequences to an a3m file."""
        names = ['sequence %d' % i for i in range(1, len(sequences) + 1)]
        a3m = []
        for sequence, name in zip(sequences, names):
            a3m.append(u'>' + name + u'\n')
            a3m.append(sequence + u'\n')
        return ''.join(a3m)


    def align(self, sequences):
        """Aligns the sequences and returns the alignment in A3M string.

        Args:
          sequences: A list of query sequence strings. The sequences have to be at
            least 6 residues long (Kalign requires this). Note that the order in
            which you give the sequences might alter the output slightly as
            different alignment tree might get constructed.

        Returns:
          A string with the alignment in a3m format.

        Raises:
          RuntimeError: If Kalign fails.
          ValueError: If any of the sequences is less than 6 residues long.
        """
        logging.info('Aligning %d sequences', len(sequences))

        for s in sequences:
            if len(s) < 6:
                raise ValueError('Kalign requires all sequences to be at least 6 '
                                 'residues long. Got %s (%d residues).' % (s, len(s)))

        with tmpdir_manager(base_dir='/tmp') as query_tmp_dir:
            input_fasta_path = os.path.join(query_tmp_dir, 'input.fasta')
            output_a3m_path = os.path.join(query_tmp_dir, 'output.a3m')

            with os.fdopen(os.open(input_fasta_path, os.O_RDWR|os.O_CREAT, stat.S_IRWXU), 'w') as f:
                f.write(self.to_a3m(sequences))

            cmd = [self.binary_path, '-i', input_fasta_path, '-o', output_a3m_path, '-format', 'fasta',]

            logging.info('Launching subprocess "%s"', ' '.join(cmd))
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with timing('Kalign query'):
                stdout, stderr = process.communicate()
                retcode = process.wait()
                logging.info('Kalign stdout:\n%s\n\nstderr:\n%s\n', stdout.decode('utf-8'), stderr.decode('utf-8'))

            if retcode:
                raise RuntimeError(
                    'Kalign failed\nstdout:\n%s\n\nstderr:\n%s\n' % (stdout.decode('utf-8'), stderr.decode('utf-8')))

            with open(output_a3m_path) as f:
                a3m = f.read()

            return a3m
