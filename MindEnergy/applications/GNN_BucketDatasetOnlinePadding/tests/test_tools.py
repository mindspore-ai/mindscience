"""Unit tests for ``src/tools.py``: ``load_npy`` keying and ``load_yaml``.

The module needs pyyaml at import time, so the whole suite is skipped when it
is missing instead of erroring -- mirroring how the other suites stay runnable
on a bare numpy install.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

try:
    import yaml  # noqa: F401  pylint: disable=unused-import
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

if _HAS_YAML:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import tools as tools_module  # noqa: E402  pylint: disable=wrong-import-position


@unittest.skipUnless(_HAS_YAML, "pyyaml is not installed")
class ToolsTest(unittest.TestCase):
    """``load_npy`` derives keys with ``splitext``; both loaders fail with paths."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)

    def test_load_npy_uses_full_stem_as_key(self):
        """``train.data.npy`` must land under ``train.data``, not ``train``.

        ``split('.')[0]`` truncated dotted stems, silently merging files that
        differ only after the first dot.
        """
        np.save(os.path.join(self.tmp_dir, "x0.npy"), np.ones((2, 2), dtype=np.float32))
        np.save(os.path.join(self.tmp_dir, "train.data.npy"), np.zeros((3, 1), dtype=np.float32))

        data = tools_module.load_npy(self.tmp_dir)

        self.assertEqual(sorted(data.keys()), ["train.data", "x0"])
        np.testing.assert_array_equal(data["train.data"], np.zeros((3, 1), dtype=np.float32))

    def test_load_npy_missing_dir_raises_with_path(self):
        """A wrong dataset directory is reported with its path."""
        missing = os.path.join(self.tmp_dir, "does-not-exist")
        with self.assertRaises(FileNotFoundError) as ctx:
            tools_module.load_npy(missing)
        self.assertIn(missing, str(ctx.exception))

    def test_load_yaml_round_trip(self):
        """A small YAML document parses into the expected nested dict."""
        path = os.path.join(self.tmp_dir, "config.yaml")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("model:\n    x_features: 7\n    output: 3\n")

        config = tools_module.load_yaml(path)

        self.assertEqual(config, {"model": {"x_features": 7, "output": 3}})

    def test_load_yaml_missing_file_raises_with_path(self):
        """A missing config file is reported with its path."""
        missing = os.path.join(self.tmp_dir, "absent.yaml")
        with self.assertRaises(FileNotFoundError) as ctx:
            tools_module.load_yaml(missing)
        self.assertIn(missing, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
