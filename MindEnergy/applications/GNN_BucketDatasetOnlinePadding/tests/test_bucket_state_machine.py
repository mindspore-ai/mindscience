"""Unit tests for the ``BucketDatasetGenerator`` iteration state machine.

``__iter__`` / ``__next__`` / ``_process_remaining_data`` / ``_get_bucket_index``
run against a stub dataset that only implements
``create_tuple_iterator(output_numpy=True)``, so the suite needs numpy only --
no MindSpore and no NPU, like the other suites in this directory.

Sample ``x0`` columns carry a nonzero tag in every real row, which lets a test
read the per-sample node counts back out of a padded batch: pad rows are the
zero fill, so the nonzero rows of slot *i* are sample *i*'s node count.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import BucketDatasetGenerator as bucket_module  # noqa: E402  pylint: disable=wrong-import-position

COLUMNS = ["x0", "node_fea1", "edge_fea1", "node_fea2", "edge_fea2", "node_fea3", "mean_edge", "labels"]


class StubDataset:
    """Minimal dataset: exactly what ``__iter__`` asks of the real one."""

    def __init__(self, samples):
        self.samples = samples

    def create_tuple_iterator(self, output_numpy=True):
        return iter(list(self.samples))


def make_sample(node_count, edge_count, tag):
    """One sample in ``DataGenerator.__getitem__`` layout, every row nonzero."""
    return (
        np.full((node_count, 7), tag, dtype=np.float32),      # x0
        np.full((node_count, 1), tag, dtype=np.float32),      # node_fea1
        np.full((edge_count, 1), tag * 10, dtype=np.float32),  # edge_fea1
        np.full((node_count, 1), tag, dtype=np.float32),      # node_fea2
        np.full((edge_count, 1), tag * 10, dtype=np.float32),  # edge_fea2
        np.full((node_count, 1), tag, dtype=np.float32),      # node_fea3
        np.full((edge_count, 1), tag * 10, dtype=np.float32),  # mean_edge
        np.full((node_count, 3), tag, dtype=np.float32),      # labels
    )


def make_generator(samples, boundaries, batch_sizes, sample_num_by_node=None, **kwargs):
    """Build the generator over a stub dataset; ``sample_num_by_node`` may lie."""
    if sample_num_by_node is None:
        sample_num_by_node = {n: 1 for n in range(1, max(boundaries))}
    return bucket_module.BucketDatasetGenerator(
        dataset=StubDataset(samples),
        sample_num_by_node=sample_num_by_node,
        bucket_boundaries=list(boundaries),
        batch_sizes=list(batch_sizes),
        column_names=COLUMNS,
        **kwargs,
    )


def drain(generator):
    """Iterate to exhaustion and return the list of batches."""
    return [batch for batch in generator]


def batch_node_counts(batch, batch_size):
    """Recover per-sample node counts of one batch from its padded ``x0``."""
    x0 = batch[COLUMNS.index("x0")]
    max_node = x0.shape[0] // batch_size
    return [int(np.count_nonzero(x0[i * max_node:(i + 1) * max_node, 0]))
            for i in range(batch_size)]


class StateMachineTest(unittest.TestCase):
    """End-to-end iteration over a stub dataset."""

    def setUp(self):
        np.random.seed(0)

    def test_every_sample_is_consumed_exactly_once(self):
        """Summing the batch-size column counts each sample once, no more."""
        counts = [3, 5, 7, 2, 6]
        samples = [make_sample(n, 2, tag=i + 1) for i, n in enumerate(counts)]
        generator = make_generator(samples, boundaries=[10], batch_sizes=[2])

        batches = drain(generator)

        self.assertEqual(sum(int(batch[-1]) for batch in batches), len(samples))
        # Every tagged row must appear exactly once across all batches.
        seen = {}
        for batch in batches:
            values = batch[COLUMNS.index("x0")][:, 0]
            for tag in np.unique(values[values > 0]):
                seen[int(tag)] = seen.get(int(tag), 0) + int((values == tag).sum())
        self.assertEqual(seen, {i + 1: n for i, n in enumerate(counts)})

    def test_batches_never_mix_buckets(self):
        """A drained batch draws from one bucket only, at that bucket's size."""
        samples = [make_sample(n, 2, tag=i + 1)
                   for i, n in enumerate([3, 4, 12, 13, 14, 15])]
        generator = make_generator(samples, boundaries=[10, 20], batch_sizes=[2, 2])

        batches = drain(generator)

        self.assertEqual(len(batches), 3)
        for batch in batches:
            batch_size = int(batch[-1])
            self.assertEqual(batch_size, 2)
            counts = batch_node_counts(batch, batch_size)
            in_first = all(n < 10 for n in counts)
            in_second = all(10 <= n < 20 for n in counts)
            self.assertTrue(in_first or in_second, msg=f"mixed batch: {counts}")

    def test_stop_iteration_is_repeatable(self):
        """Exhaustion raises ``StopIteration`` and keeps raising afterwards."""
        samples = [make_sample(4, 2, tag=1)]
        generator = make_generator(samples, boundaries=[10], batch_sizes=[2])

        drain(generator)
        self.assertRaises(StopIteration, generator.__next__)
        self.assertRaises(StopIteration, generator.__next__)

    def test_sample_beyond_boundary_raises(self):
        """A node count outside every bucket fails loudly, not into the last bucket."""
        samples = [make_sample(20, 2, tag=1)]
        generator = make_generator(samples, boundaries=[10], batch_sizes=[2])

        with self.assertRaises(ValueError) as ctx:
            for _ in generator:
                pass
        self.assertIn("beyond the maximum bucket boundary", str(ctx.exception))

    def test_estimate_shortfall_no_longer_crashes(self):
        """Actual batches beyond the 1.15x estimate cycle the drain list.

        The config claims 1 sample but the dataset holds 10 with bucket batch
        size 1, so 10 batches are emitted against a 2-entry ``random_list``
        (``ceil(1.15 * 1)``). This used to raise IndexError mid-epoch.
        """
        samples = [make_sample(5, 2, tag=i + 1) for i in range(10)]
        generator = make_generator(samples, boundaries=[10], batch_sizes=[1],
                                   sample_num_by_node={5: 1})

        batches = drain(generator)

        self.assertEqual(len(batches), 10)
        self.assertTrue(all(int(batch[-1]) == 1 for batch in batches))

    def test_construction_rejects_length_mismatch(self):
        """``batch_sizes`` shorter than ``bucket_boundaries`` is a ValueError."""
        samples = [make_sample(4, 2, tag=1)]
        with self.assertRaises(ValueError):
            make_generator(samples, boundaries=[10, 20], batch_sizes=[2])

    def test_construction_rejects_out_of_range_sample_counts(self):
        """``sample_num_by_node`` beyond the boundaries is rejected at build time."""
        samples = [make_sample(4, 2, tag=1)]
        with self.assertRaises(ValueError):
            make_generator(samples, boundaries=[10], batch_sizes=[2],
                           sample_num_by_node={50: 1})


if __name__ == "__main__":
    unittest.main()
