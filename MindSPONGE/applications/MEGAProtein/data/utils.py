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
utils module used for tmpdir generation.
"""
import time
import contextlib
import tempfile
import shutil

from absl import logging


@contextlib.contextmanager
def tmpdir_manager(base_dir: str):
    """Context manager that deletes a temporary directory on exit.
    for example:
        with tmpdir_manager(base_dir='/tmp') as tmp_dir:
            test_file = os.path.join(tmp_dir, 'input.fasta')
            with open(test_file, "w") as f:
               f.write("this is a test. \n")
            print("exit")
    this would create a tmp data directory and when finished the main process of writing "this is a test. \n" into
    the tmp file,(after print "exit"), the system would destroy the previous tmp dir
    """
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info('Started %s', msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info('Finished %s in %.3f seconds', msg, toc - tic)
