"""Unit tests for the orthogonal grouping layout in ``distributed/context.py``.

Pure Python, no MindSpore and no numpy: ``context.py`` defers its MindSpore
imports to the functions that need them, so ``import distributed`` works on a
bare interpreter and ``python -m unittest discover -s tests`` runs this suite
anywhere.

The expected memberships are the three configurations measured in the README
("Orthogonal grouping"), so a regression here shows up as a diff against
documented behavior, not just against itself.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from distributed import Group, group, init_parallel  # noqa: E402  pylint: disable=wrong-import-position
from distributed import context as context_module  # noqa: E402  pylint: disable=wrong-import-position


class GroupMembersTest(unittest.TestCase):
    """``_group_members`` must partition the world exactly, in Megatron layout."""

    def partition(self, sizes, mask, world_size):
        """Run ``_group_members`` and assert the result is an exact partition."""
        groups = context_module._group_members(sizes, mask, world_size)
        flat = sorted(rank for members in groups for rank in members)
        self.assertEqual(flat, list(range(world_size)),
                         msg=f"not a partition of 0..{world_size - 1}: {groups}")
        group_size = 1
        for dim_size, selected in zip(sizes, mask):
            if selected:
                group_size *= dim_size
        for members in groups:
            self.assertEqual(len(members), group_size, msg=f"ragged group: {members}")
        return groups

    def test_tp2_cp2_pp1_world8_matches_readme(self):
        """The default topology must reproduce the README partition table."""
        sizes = [2, 2, 2, 1]  # tp, cp, dp, pp in "tp-cp-dp-pp" order
        self.assertEqual(self.partition(sizes, [True, False, False, False], 8),
                         [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(self.partition(sizes, [False, True, False, False], 8),
                         [[0, 2], [1, 3], [4, 6], [5, 7]])
        self.assertEqual(self.partition(sizes, [False, False, True, False], 8),
                         [[0, 4], [1, 5], [2, 6], [3, 7]])
        self.assertEqual(self.partition(sizes, [False, True, True, False], 8),
                         [[0, 2, 4, 6], [1, 3, 5, 7]])

    def test_tp4_cp2_pp1_world8_matches_readme(self):
        """With dp collapsed to 1, the dp groups degrade to singletons.

        ``dp`` is derived from the world size, so tp=4, cp=2, pp=1 at world=8
        gives per-axis sizes ``[4, 2, 1, 1]`` -- not ``[4, 2, 2, 1]``.
        """
        sizes = [4, 2, 1, 1]
        self.assertEqual(self.partition(sizes, [True, False, False, False], 8),
                         [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(self.partition(sizes, [False, True, False, False], 8),
                         [[0, 4], [1, 5], [2, 6], [3, 7]])
        self.assertEqual(self.partition(sizes, [False, False, True, False], 8),
                         [[rank] for rank in range(8)])

    def test_tp2_cp2_pp2_world8_matches_readme(self):
        """A pp axis of 2 changes the strides, not the partition property."""
        sizes = [2, 2, 1, 2]
        self.assertEqual(self.partition(sizes, [True, False, False, False], 8),
                         [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(self.partition(sizes, [False, True, False, False], 8),
                         [[0, 2], [1, 3], [4, 6], [5, 7]])
        self.assertEqual(self.partition(sizes, [False, False, True, False], 8),
                         [[rank] for rank in range(8)])

class GroupDegradationTest(unittest.TestCase):
    """Before ``init_parallel`` every handle degrades to a size-1 no-op group."""

    def test_uninitialized_group_is_size_one_no_op(self):
        """``group(kind)`` hands out a fresh size-1 handle until init ran."""
        handle = group("tp")
        self.assertIsInstance(handle, Group)
        self.assertEqual(handle.size, 1)
        self.assertEqual(handle.rank, 0)

    def test_name_requires_initialization(self):
        """``.name`` is the one attribute that cannot degrade: it fails loudly."""
        handle = Group("tp")
        with self.assertRaises(RuntimeError):
            _ = handle.name

    def test_every_kind_has_a_handle(self):
        """All four group kinds answer before init, not just "tp"."""
        for kind in ("tp", "cp", "dp", "dp-cp"):
            self.assertEqual(group(kind).size, 1, msg=kind)


class InitParallelValidationTest(unittest.TestCase):
    """Argument validation fires before any communication API is touched."""

    def test_invalid_order_is_rejected(self):
        """An unknown axis is a ValueError, not the old bare KeyError."""
        with self.assertRaises(ValueError):
            init_parallel(order="tp-cp-dp-xx")

    def test_duplicated_or_missing_axes_are_rejected(self):
        """Each axis must appear exactly once, whatever the permutation."""
        for order in ("tp-cp-dp", "tp-cp", "tp-cp-dp-pp-pp", "tp-tp-cp-pp"):
            with self.assertRaises(ValueError, msg=order):
                init_parallel(order=order)

    def test_reordered_order_is_accepted_shape(self):
        """A permutation of the four axes passes validation; without MindSpore
        it then fails with a clear RuntimeError instead of a traceback into
        mindspore internals."""
        with self.assertRaises(RuntimeError):
            init_parallel(order="pp-dp-cp-tp")

    def test_without_communication_it_fails_with_guidance(self):
        """Valid arguments without an initialized communicator: RuntimeError,
        never an ImportError leak or a silent no-op."""
        with self.assertRaises(RuntimeError):
            init_parallel(tp=2, cp=2, pp=1)


if __name__ == "__main__":
    unittest.main()
