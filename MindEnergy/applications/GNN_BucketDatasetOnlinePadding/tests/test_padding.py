"""Unit tests for ``src/padding.py``.

Pure numpy, no MindSpore and no NPU: ``python -m unittest discover -s tests``
finishes in milliseconds, so this suite can run in the pre-commit hook.

Node-axis failures show up as "shape right, one column wrong", which is why most
of the cases below assert the pad region of *every* registered column rather
than only the first one.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import padding  # noqa: E402  pylint: disable=wrong-import-position


class NodeAxisTest(unittest.TestCase):
    """Seven cases for ``pad_to_max`` / ``pad_nodes``."""

    def test_pad_to_max_is_noop_for_full_length(self):
        """A sample that already has ``max_node`` rows is returned as is.

        This is what makes the module a no-op on pre-padded datasets: not "the
        values happen to match" but "the very same object comes back", so the
        offline path pays neither a copy nor an allocation.
        """
        sample = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.assertIs(padding.pad_to_max(sample, 4, 0.0), sample)
        self.assertIs(padding.pad_to_max(sample, 3, 0.0), sample)

    def test_equal_length_batch_values_unchanged(self):
        """Equal-length input in, identical values out."""
        x0 = [np.arange(9, dtype=np.float32).reshape(3, 3) for _ in range(2)]
        labels = [np.arange(6, dtype=np.float32).reshape(3, 2) for _ in range(2)]
        padded, max_node = padding.pad_nodes({"x0": list(x0), "labels": list(labels)})
        self.assertEqual(max_node, 3)
        np.testing.assert_array_equal(padded["x0"], np.concatenate(x0, axis=0))
        np.testing.assert_array_equal(padded["labels"], np.concatenate(labels, axis=0))

    def test_variable_length_row_count_and_slots(self):
        """Row count is ``batch * max_node`` and sample *i* owns slot *i*."""
        lengths = [2, 3, 1]
        x0 = [np.full((n, 2), i, dtype=np.float32) for i, n in enumerate(lengths)]
        padded, max_node = padding.pad_nodes({"x0": x0})
        self.assertEqual(max_node, 3)
        self.assertEqual(padded["x0"].shape, (len(lengths) * max_node, 2))
        for i, n in enumerate(lengths):
            slot = padded["x0"][i * max_node:(i + 1) * max_node]
            np.testing.assert_array_equal(slot[:n], np.full((n, 2), i, dtype=np.float32))
            np.testing.assert_array_equal(slot[n:], np.zeros((max_node - n, 2), dtype=np.float32))

    def test_labels_are_padded_like_any_node_column(self):
        """``labels`` is node-level, so it must be padded, not concatenated raw.

        Concatenating it along axis 0 without padding also "works" -- it just
        returns ``sum(n_i)`` rows instead of ``batch * max_node``, and from then
        on label row ``k`` no longer belongs to node row ``k``. An equal-length
        batch hides the problem, because there the row counts coincide.
        """
        lengths = [4, 2]
        labels = [np.ones((n, 3), dtype=np.float32) * (i + 1) for i, n in enumerate(lengths)]
        padded, max_node = padding.pad_nodes({"labels": labels})
        self.assertEqual(max_node, 4)
        self.assertEqual(padded["labels"].shape, (2 * max_node, 3))
        np.testing.assert_array_equal(padded["labels"][:4], 1.0)
        np.testing.assert_array_equal(padded["labels"][4:6], 2.0)
        np.testing.assert_array_equal(padded["labels"][6:], 0.0)

    def test_pad_region_is_zero_for_every_column(self):
        """Every registered column is filled from the table, not left unset."""
        arrays = {name: [np.ones((1, 2), dtype=np.float32)] for name in padding.NODE_FILLS}
        padded, max_node = padding.pad_nodes(arrays, max_node=3)
        self.assertEqual(max_node, 3)
        for name in padding.NODE_FILLS:
            self.assertEqual(padded[name].shape, (3, 2), msg=name)
            np.testing.assert_array_equal(padded[name][1:], np.zeros((2, 2), dtype=np.float32),
                                          err_msg=f"{name} pad region must equal its fill value")

    def test_unregistered_column_falls_back_to_zero(self):
        """A column missing from the table is padded with 0 instead of raising."""
        padded, _ = padding.pad_nodes({"temporary": [np.ones((1, 2), dtype=np.float32)]}, max_node=2)
        np.testing.assert_array_equal(padded["temporary"][1:], np.zeros((1, 2), dtype=np.float32))

    def test_defensive_errors(self):
        """Empty batch and a too-small target both raise instead of truncating."""
        with self.assertRaises(ValueError):
            padding.pad_nodes({})
        with self.assertRaises(ValueError):
            padding.pad_nodes({"x0": [np.zeros((3, 2), dtype=np.float32)]}, max_node=2)


if __name__ == "__main__":
    unittest.main()
