"""Unit tests for ``distributed/grad.py``: plan classification, fused reduction, global norm.

Needs MindSpore for real tensors. Anything less than a working build -- no
mindspore at all, or a wheel whose CPU kernels cannot execute ``mint.stack`` --
skips the whole suite instead of erroring, mirroring the other suites here.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import mindspore as ms
    from mindspore import mint
    # Probe the kernels this module relies on (cat / slice / reshape / copy_ /
    # stack). Some CPU wheels ship with an incomplete kernel set where running
    # ``mint.stack`` silently breaks *subsequent* kernel selection, so the
    # sequence below deliberately exercises plain ops again *after* the stack
    # call: on such a wheel it raises here and the suite skips as a whole,
    # instead of failing midway with confusing per-test errors.
    _probe = mint.ones((2, 3))
    _probe.copy_(mint.cat([mint.ones(3), mint.ones(3)])[0:6].reshape(2, 3))
    mint.stack([mint.ones(1)]).sum()
    _probe = ms.Parameter(mint.ones((2, 3)), name="probe")
    _probe.copy_(mint.ones((2, 3)))
    _HAS_MS = True
except Exception:  # pylint: disable=broad-except  any failure means "cannot run here"
    _HAS_MS = False

if _HAS_MS:
    from distributed import build_plan, global_norm, reduce_grads  # noqa: E402  pylint: disable=wrong-import-position
    from distributed import layers as layers_module  # noqa: E402  pylint: disable=wrong-import-position
    from distributed.layers import mark_grad_complete, mark_sharded  # noqa: E402  pylint: disable=wrong-import-position


def make_param(shape, tag=None):
    """One trainable parameter, optionally tagged into a non-default bucket."""
    param = ms.Parameter(mint.ones(shape), name="p" + str(shape))
    if tag == "sharded":
        mark_sharded(param)
    elif tag == "complete":
        mark_grad_complete(param)
    return param


def make_grads(params):
    """Fresh gradients whose entry *i* is filled with value *i* + 1."""
    return [mint.ones(parameter.shape) * (i + 1) for i, parameter in enumerate(params)]


def expected_norm(params):
    """Square sum of ``make_grads(params)`` computed in plain Python."""
    return math.sqrt(sum((i + 1) ** 2 * math.prod(parameter.shape)
                         for i, parameter in enumerate(params)))


@unittest.skipUnless(_HAS_MS, "mindspore with working mint kernels is not available")
class GradTest(unittest.TestCase):
    """The registry outlives the test parameters, so scrub it around each case."""

    def setUp(self):
        layers_module._REGISTRY.clear()
        self.addCleanup(layers_module._REGISTRY.clear)


class BuildPlanTest(GradTest):

    def test_three_bucket_classification(self):
        """Tags decide the bucket; untagged falls to the strict ``partial``."""
        params = [make_param((2, 2)),                  # untagged -> partial
                  make_param((3,), tag="sharded"),
                  make_param((4,), tag="complete"),
                  make_param((5,), tag="sharded")]
        plan = build_plan(params, fuse=False)
        self.assertEqual(plan.partial, [0])
        self.assertEqual(plan.sharded, [1, 3])
        self.assertEqual(plan.complete, [2])

    def test_extra_partial_forces_into_partial_bucket(self):
        """A host-provided index lands in ``partial`` exactly once."""
        params = [make_param((2, 2)), make_param((3,))]
        plan = build_plan(params, fuse=False, extra_partial=[1, 1])
        self.assertEqual(plan.partial, [0, 1])

    def test_fused_plan_groups_by_dtype_with_offsets(self):
        """One dtype means one bucket; offsets/sizes/total describe the flat buffer."""
        params = [make_param((2, 3)), make_param((4,)), make_param((1, 2, 2))]
        plan = build_plan(params, fuse=True)
        self.assertEqual(len(plan.buckets), 1)
        bucket = plan.buckets[0]
        self.assertEqual(bucket["idx"], [0, 1, 2])
        self.assertEqual(bucket["sizes"], [6, 4, 4])
        self.assertEqual(bucket["offsets"], [0, 6, 10])
        self.assertEqual(bucket["total"], 14)


class ReduceGradsTest(GradTest):

    def test_identity_reduce_keeps_every_value_bit_identical(self):
        """With an identity all-reduce (the size==1 semantics) fusion is exact."""
        params = [make_param((2, 3)), make_param((4,)), make_param((1, 2))]
        grads = make_grads(params)
        plan = build_plan(params, fuse=True)
        out = reduce_grads(grads, plan, lambda x: x)
        for i, (g, p) in enumerate(zip(out, params)):
            self.assertIs(out[i], grads[i])
            self.assertAlmostEqual(float(g.reshape(-1)[0]), float(i + 1), places=6)
            self.assertEqual(tuple(g.shape), tuple(p.shape))

    def test_unfused_path_matches_fused_result(self):
        """Both code paths write the same values back under identity reduce."""
        params = [make_param((3,)), make_param((2, 2))]
        plan_fused = build_plan(params, fuse=True)
        plan_plain = build_plan(params, fuse=False)
        grads_a = make_grads(params)
        grads_b = make_grads(params)
        reduce_grads(grads_a, plan_fused, lambda x: x)
        reduce_grads(grads_b, plan_plain, lambda x: x)
        for left, right in zip(grads_a, grads_b):
            self.assertEqual(float(left.reshape(-1)[0]), float(right.reshape(-1)[0]))

    def test_empty_partial_bucket_returns_original_object(self):
        """No partial parameters means no communication and the same list."""
        params = [make_param((2,), tag="sharded")]
        grads = make_grads(params)
        plan = build_plan(params, fuse=False)
        self.assertIs(reduce_grads(grads, plan, lambda x: x), grads)


class GlobalNormTest(GradTest):

    def test_no_plan_plain_norm(self):
        """Without a plan this is the ordinary L2 norm of the gradients."""
        params = [make_param((2, 2)), make_param((3,))]
        norm = global_norm(make_grads(params))
        self.assertAlmostEqual(float(norm), expected_norm(params), places=5)

    def test_plan_without_sharded_needs_no_reduce(self):
        """An empty ``sharded`` bucket must not demand an ``all_reduce``."""
        params = [make_param((2, 2)), make_param((3,))]
        grads = make_grads(params)
        plan = build_plan(params, fuse=False)
        norm = global_norm(grads, plan=plan, all_reduce=None)
        self.assertAlmostEqual(float(norm), expected_norm(params), places=5)

    def test_sharded_with_identity_reduce_equals_plain_norm(self):
        """Identity all-reduce on the shard sum reproduces the full norm."""
        params = [make_param((2, 2)), make_param((3,), tag="sharded")]
        grads = make_grads(params)
        plan = build_plan(params, fuse=False)
        norm = global_norm(grads, plan=plan, all_reduce=lambda x: x)
        self.assertAlmostEqual(float(norm), expected_norm(params), places=5)

    def test_only_the_sharded_bucket_is_reduced(self):
        """A zero cross-group sum removes exactly the sharded contribution."""
        params = [make_param((2, 2)), make_param((3,), tag="sharded")]
        plan = build_plan(params, fuse=False)
        grads = make_grads(params)
        norm = global_norm(grads, plan=plan, all_reduce=lambda x: x * 0)
        sharded_sq = (2.0 ** 2) * math.prod(params[1].shape)
        self.assertAlmostEqual(float(norm), math.sqrt(expected_norm(params) ** 2 - sharded_sq),
                               places=5)

    def test_sharded_without_all_reduce_raises(self):
        """Missing ``all_reduce`` with sharded parameters fails loudly instead
        of returning a silently shard-factor-inflated norm."""
        params = [make_param((2, 2)), make_param((3,), tag="sharded")]
        grads = make_grads(params)
        plan = build_plan(params, fuse=False)
        with self.assertRaises(ValueError):
            global_norm(grads, plan=plan, all_reduce=None)


if __name__ == "__main__":
    unittest.main()
