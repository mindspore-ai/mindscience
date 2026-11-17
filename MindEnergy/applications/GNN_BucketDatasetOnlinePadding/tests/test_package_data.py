"""Unit tests for ``BucketDatasetGenerator._package_data``.

The batching function is the only piece of this application that needs to know
how a batch is laid out, so it is tested without MindSpore: the generator is
constructed directly and fed plain tuples of numpy arrays, exactly what
``create_tuple_iterator(output_numpy=True)`` would hand it.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import BucketDatasetGenerator as bucket_module  # noqa: E402  pylint: disable=wrong-import-position

COLUMNS = ["x0", "node_fea1", "edge_fea1", "node_fea2", "edge_fea2", "node_fea3", "mean_edge", "labels"]


def make_generator():
    """Build a generator with one bucket of batch size 2; no dataset is needed."""
    return bucket_module.BucketDatasetGenerator(
        dataset=None,
        sample_num_by_node={5: 4},
        bucket_boundaries=[10],
        batch_sizes=[2],
        column_names=list(COLUMNS),
        rank_size=1,
    )


def make_sample(node_count, edge_count, tag=1):
    """One sample as ``__getitem__`` returns it: (n, d) node columns, (e, 1) edge ones."""
    return (
        np.full((node_count, 7), tag, dtype=np.float32),                         # x0
        np.full((node_count, 1), tag, dtype=np.float32),                         # node_fea1
        np.full((edge_count, 1), tag * 10, dtype=np.float32),                    # edge_fea1
        np.full((node_count, 1), tag, dtype=np.float32),                         # node_fea2
        np.full((edge_count, 1), tag * 10, dtype=np.float32),                    # edge_fea2
        np.full((node_count, 1), tag, dtype=np.float32),                         # node_fea3
        np.full((edge_count, 1), tag * 10, dtype=np.float32),                    # mean_edge
        np.full((node_count, 3), tag, dtype=np.float32),                         # labels
    )


class PackageDataTest(unittest.TestCase):
    """Layout of one packaged batch."""

    def setUp(self):
        self.generator = make_generator()

    def test_equal_length_batch_matches_column_concat(self):
        """Equal node counts: the batch is a plain concatenation of the samples."""
        samples = [make_sample(4, 3, tag=1), make_sample(4, 3, tag=2)]
        batch = self.generator._package_data(samples, 2)

        self.assertEqual(len(batch), len(COLUMNS) + 2)
        for name, value in zip(COLUMNS, batch):
            np.testing.assert_array_equal(
                value, np.concatenate([sample[COLUMNS.index(name)] for sample in samples], axis=0),
                err_msg=name)

    def test_variable_length_nodes_are_padded(self):
        """Node columns get ``batch * max_node`` rows; edge columns do not."""
        samples = [make_sample(5, 2, tag=1), make_sample(3, 2, tag=2)]
        batch = self.generator._package_data(samples, 2)
        by_name = dict(zip(COLUMNS, batch))

        self.assertEqual(by_name["x0"].shape, (10, 7))
        self.assertEqual(by_name["labels"].shape, (10, 3))
        self.assertEqual(by_name["node_fea1"].shape, (10, 1))
        np.testing.assert_array_equal(by_name["labels"][:5], np.full((5, 3), 1.0))
        np.testing.assert_array_equal(by_name["labels"][5:8], np.full((3, 3), 2.0))
        np.testing.assert_array_equal(by_name["labels"][8:], np.zeros((2, 3)))

        for name in ("edge_fea1", "edge_fea2", "mean_edge"):
            self.assertEqual(by_name[name].shape, (4, 1), msg=name)

    def test_edge_batch_id_counts_edges_not_nodes(self):
        """``batch_id`` has ``sum(e_i)`` entries; the node count must not leak in.

        This is the shape a 5-node/2-edge graph produces. Seeding the id array
        from a node-level column gives 10 entries here instead of 5, and every
        edge-to-sample lookup after the first sample shifts.
        """
        samples = [make_sample(5, 2, tag=1), make_sample(5, 3, tag=2)]
        batch = self.generator._package_data(samples, 2)
        batch_id = batch[len(COLUMNS)]

        self.assertEqual(len(batch_id), 5)
        self.assertEqual(batch_id.tolist(), [0, 0, 1, 1, 1])

    def test_surplus_samples_in_bucket_are_trimmed(self):
        """A drained bucket may hold more than one batch: only ``batch_size`` is used."""
        samples = [make_sample(n, 2) for n in (2, 4, 6, 8)]
        batch = self.generator._package_data(samples, 2)
        by_name = dict(zip(COLUMNS, batch))

        self.assertEqual(by_name["x0"].shape[0], 2 * 4)
        self.assertEqual(by_name["labels"].shape[0], 2 * 4)

    def test_input_samples_are_not_modified(self):
        """Packaging twice yields the same batch; the samples stay untouched.

        Shifting ``edge_fea1`` in place would shift it a second time on the
        next pass, and ``mean_edge`` would stop matching
        ``(edge_fea1 + edge_fea2) / 2``.
        """
        samples = [make_sample(4, 2, tag=1), make_sample(4, 2, tag=2)]
        before = [tuple(array.copy() for array in sample) for sample in samples]

        first = self.generator._package_data(samples, 2)
        second = self.generator._package_data(samples, 2)

        for sample, snapshot in zip(samples, before):
            for array, expected in zip(sample, snapshot):
                np.testing.assert_array_equal(array, expected)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
