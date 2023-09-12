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
"""utilities for sciai tests"""
import io
import itertools
import sys
import threading
import unittest

import mindspore as ms

from sciai.utils import set_seed


def stub_stdout():
    stderr = sys.stderr
    stdout = sys.stdout

    sys.stderr = io.StringIO()
    sys.stdout = io.StringIO()
    return stderr, stdout


def clear_stub(stderr, stdout):
    sys.stderr = stderr
    sys.stdout = stdout


class MindsporeTestBase(unittest.TestCase):
    """utilities for sciai tests"""
    CARD_SET = False
    LOCK = threading.Lock()

    @classmethod
    def setUpClass(cls) -> None:
        if not MindsporeTestBase.CARD_SET:
            with MindsporeTestBase.LOCK:
                if not MindsporeTestBase.CARD_SET:
                    MindsporeTestBase.CARD_SET = True
                    set_seed(1234)

    def setUp(self) -> None:
        self.stub_stdout()

    def stub_stdout(self):
        """
        Feature: ALL TO ALL
        Description: pass
        Expectation: pass
        """
        stderr = sys.stderr
        stdout = sys.stdout

        def cleanup():
            sys.stderr = stderr
            sys.stdout = stdout

        self.addCleanup(cleanup)
        sys.stderr = io.StringIO()
        sys.stdout = io.StringIO()

    def before(self, mode=ms.GRAPH_MODE):
        ms.set_context(mode=mode)

    # def test_comb_func1(self, mode=ms.GRAPH_MODE):
    #     ms.set_context(mode=mode)
    #     print(ms.get_context("mode"))

    @staticmethod
    def get_test_func(func_name, *args):
        def func(self):
            getattr(self, func_name)(*args)

        return func

    @classmethod
    def generate_test_cases(cls):
        """generate_test_cases"""
        modes = [ms.PYNATIVE_MODE, ms.GRAPH_MODE]  # specify parameter combinations
        arg_lists = list(itertools.product(modes))
        test_func_names = list(filter(lambda _: _.startswith("test_comb"), dir(cls)))
        for test_func_name in test_func_names:
            for args in arg_lists:
                new_func_name = f"test{test_func_name[9:]}_{'_'.join([str(_) for _ in args])}"
                setattr(cls, new_func_name, cls.get_test_func(test_func_name[5:], *args))
            setattr(cls, test_func_name[5:], getattr(cls, test_func_name))
            delattr(cls, test_func_name)


if __name__ == '__main__':
    MindsporeTestBase.generate_test_cases()
    unittest.main()
