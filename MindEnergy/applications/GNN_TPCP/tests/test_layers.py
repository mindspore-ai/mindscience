"""Unit tests for ``distributed/layers.py`` on the ``size == 1`` path.

Everything here runs without any communication: the passthrough layers wrap a
native ``Linear`` and ``initialize_affine_weight`` only reads ``.size`` /
``.rank`` off its group handle, so a fake group covers the sharded slicing.

Like the other suites, the whole module skips when MindSpore cannot run here --
missing entirely, or a wheel whose kernels fail (the probe mirrors the one in
``test_grad.py``).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import mindspore as ms
    from mindspore import mint
    _probe = ms.Parameter(mint.ones((4, 6)), name="probe")
    _probe.copy_(mint.cat([mint.ones(6), mint.ones(6)])[0:12].reshape(4, 6))
    mint.matmul(mint.ones((2, 4)), mint.ones((4, 6)))
    mint.stack([mint.ones(1)]).sum()
    _probe = ms.Parameter(mint.ones((4, 6)), name="probe2")
    _HAS_MS = True
except Exception:  # pylint: disable=broad-except  any failure means "cannot run here"
    _HAS_MS = False

if _HAS_MS:
    from distributed.layers import (ColumnParallelLinear, RowParallelLinear,  # noqa: E402  pylint: disable=wrong-import-position
                                    clear_tags, initialize_affine_weight, is_grad_complete,
                                    is_sharded, mark_grad_complete, mark_sharded, release_tag,
                                    tag_of)
    from distributed import layers as layers_module  # noqa: E402  pylint: disable=wrong-import-position


class FakeGroup:
    """A size/rank stand-in: ``initialize_affine_weight`` only reads these."""

    def __init__(self, size, rank=0):
        self._size, self._rank = size, rank

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return self._rank


@unittest.skipUnless(_HAS_MS, "mindspore with working mint kernels is not available")
class LayersTest(unittest.TestCase):
    """Passthrough numerics, weight-init guards and the tag registry lifecycle.

    The registry outlives the test parameters, so scrub it around each case.
    """

    def setUp(self):
        layers_module._REGISTRY.clear()
        self.addCleanup(layers_module._REGISTRY.clear)

    def _copy_weights(self, parallel_layer, native_linear):
        """Make a passthrough layer compute exactly what the native one does."""
        for mine, theirs in zip(parallel_layer.trainable_params(), native_linear.trainable_params()):
            mine.set_data(theirs)

    def test_column_passthrough_matches_native_linear(self):
        """``size == 1`` must be bit-identical to the wrapped ``mint.nn.Linear``."""
        column = ColumnParallelLinear(4, 6, bias=True)
        native = mint.nn.Linear(4, 6)
        self._copy_weights(column, native)
        x = mint.ones((2, 4))
        out, ref = column(x), native(x)
        self.assertEqual(float((out - ref).abs().sum()), 0.0)

    def test_row_passthrough_matches_native_linear(self):
        """Same hard rule for the row-parallel layer."""
        row = RowParallelLinear(6, 4, bias=True)
        native = mint.nn.Linear(6, 4)
        self._copy_weights(row, native)
        x = mint.ones((2, 6))
        out, ref = row(x), native(x)
        self.assertEqual(float((out - ref).abs().sum()), 0.0)

    def test_size_one_exposes_attribute_aliases(self):
        """Introspection sees the same attribute surface as the parallel path."""
        column = ColumnParallelLinear(4, 6, bias=False)
        self.assertIsNone(column.in_map)
        self.assertIsNone(column.out_map)
        self.assertIs(column.weight, column.layer.weight)
        self.assertIsNone(column.bias)

        row = RowParallelLinear(6, 4, bias=True)
        self.assertIsNone(row.in_map)
        self.assertIsNone(row.out_map)
        self.assertIs(row.bias, row.layer.bias)

    def test_initialize_affine_weight_size_one_keeps_global_shape(self):
        """Without a group the full weight comes back, untagged."""
        param = initialize_affine_weight((4, 6))
        self.assertEqual(tuple(param.shape), (4, 6))
        self.assertFalse(is_sharded(param))

    def test_initialize_affine_weight_slices_by_group(self):
        """A size-2 group keeps a slice of the global shape, tagged ``sharded``."""
        param = initialize_affine_weight((4, 6), g=FakeGroup(2), partition_dim=0)
        self.assertEqual(tuple(param.shape), (2, 6))
        self.assertTrue(is_sharded(param))

    def test_initialize_affine_weight_rejects_bad_splits(self):
        """Indivisible axes and straddled groups both fail loudly."""
        with self.assertRaises(ValueError):
            initialize_affine_weight((5, 6), g=FakeGroup(2), partition_dim=0)
        with self.assertRaises(ValueError):
            initialize_affine_weight((6, 6), g=FakeGroup(2), partition_dim=0, num_groups=3)

    def test_tag_round_trip_and_cleanup(self):
        """Tag queries read the registry; release/clear free the references."""
        sharded = mark_sharded(ms.Parameter(mint.ones((3,))))
        complete = mark_grad_complete(ms.Parameter(mint.ones((3,))))
        self.assertTrue(is_sharded(sharded))
        self.assertTrue(is_grad_complete(complete))
        self.assertEqual(tag_of(sharded), "sharded")
        self.assertIsNone(tag_of(ms.Parameter(mint.ones((3,)))))

        release_tag(sharded)
        self.assertIsNone(tag_of(sharded))
        self.assertTrue(is_grad_complete(complete))

        clear_tags()
        self.assertIsNone(tag_of(complete))
        self.assertEqual(len(layers_module._REGISTRY), 0)


if __name__ == "__main__":
    unittest.main()
