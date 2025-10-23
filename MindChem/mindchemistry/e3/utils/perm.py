# Copyright 2022 Huawei Technologies Co., Ltd
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
"""permutation operators"""
import random
import math


def _is_perm(p):
    return sorted(set(p)) == list(range(len(p)))


def _identity(n):
    return tuple(i for i in range(n))


def _compose(p1, p2):
    r"""
    compute p1 . p2
    p: i |-> p[i]
    [p1.p2](i) = p1(p2(i)) = p1[p2[i]]
    """
    assert _is_perm(p1) and _is_perm(p2)
    assert len(p1) == len(p2)

    return tuple(p1[p2[i]] for i in range(len(p1)))


def _inverse(p):
    r"""
    compute the inverse permutation
    """
    return tuple(p.index(i) for i in range(len(p)))


def _rand(n):
    i = random.randint(0, math.factorial(n) - 1)
    return _from_int(i, n)


def _from_int(i, n):
    pool = list(range(n))
    p = []
    for _ in range(n):
        j = i % n
        i = i // n
        p.append(pool.pop(j))
        n -= 1
    return tuple(p)


def _to_int(p):
    n = len(p)
    pool = list(range(n))
    i = 0
    m = 1
    for j in p:
        k = pool.index(j)
        i += k * m
        m *= len(pool)
        pool.pop(k)
    return i


def _group(n):
    return {_from_int(i, n) for i in range(math.factorial(n))}


def _germinate(subset):
    while True:
        n = len(subset)
        subset = subset.union([_inverse(p) for p in subset])
        subset = subset.union([
            _compose(p1, p2)
            for p1 in subset
            for p2 in subset
        ])
        if len(subset) == n:
            return subset


def _is__(g):
    if len(g) == 0:
        return False

    n = len(next(iter(g)))

    for p in g:
        assert len(p) == n, p

    if not _identity(n) in g:
        return False

    for p in g:
        if not _inverse(p) in g:
            return False

    for p1 in g:
        for p2 in g:
            if not _compose(p1, p2) in g:
                return False

    return True


def _to_cycles(p):
    n = len(p)

    cycles = set()

    for i in range(n):
        c = [i]
        while p[i] != c[0]:
            i = p[i]
            c.append(i)
        if len(c) >= 2:
            i = c.index(min(c))
            c = c[i:] + c[:i]
            cycles.add(tuple(c))

    return cycles


def _sign(p):
    s = 1
    for c in _to_cycles(p):
        if len(c) % 2 == 0:
            s = -s
    return s
