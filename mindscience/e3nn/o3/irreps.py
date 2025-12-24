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
"""Irreducible representations and their direct sums for O(3)."""

import itertools
import collections
import dataclasses

import numpy as np

from mindspore import jit_class, Tensor, ops

from .wigner import wigner_D
from .rotation import matrix_to_angles
from ..utils.func import broadcast_args, _to_tensor, norm_keep, _expand_last_dims, narrow
from ..utils.perm import _inverse
from ..utils.linalg import _direct_sum

# pylint: disable=C0111

@jit_class
@dataclasses.dataclass(init=False, frozen=True)
class Irrep:
    r"""
    Irreducible representation of O(3).
    This class does not contain any data, it is a structure that
    describe the representation.
    It is typically used as argument of other classes of the library to
    define the input and output representations of functions.
    The irrep is labeled by a non-negative integer `l` (the degree) and
    a parity `p` (1 for even, -1 for odd).
    Common aliases: "e" for even parity, "o" for odd parity, "y" for
    parity (-1)^l.

    Args:
        l (Union[int, str]): non-negative integer, the degree of the representation, :math:`l = 0, 1, \dots`.
            Alternatively, a string such as ``"1o"`` or ``"2e"`` encoding both degree and parity.
        p (int, optional): the parity of the representation, :math:`p \in \{1, -1\}`.
            Ignored when ``l`` is a string. Default: ``None``.

    Raises:
        NotImplementedError: If method is not implemented.
        ValueError: If `l` is negative or `p` is not in {1, -1}.
        ValueError: If `l` cannot be converted to an `Irrep`.
        TypeError: If `l` is not int or str.

    Examples:
        >>> from mindscience.e3nn.o3 import Irrep
        >>> Irrep(0, 1)
        0e
        >>> Irrep("1y")
        1o
        >>> Irrep("2o").dim
        5
        >>> Irrep("2e") in Irrep("1o") * Irrep("1o")
        True
        >>> Irrep("1o") + Irrep("2o")
        1x1o+1x2o
    """
    l: int
    p: int

    def __init__(self, l, p=None):
        if p is None:
            if isinstance(l, Irrep):
                p = l.p
                l = l.l

            if isinstance(l, _MulIr):
                p = l.ir.p
                l = l.ir.l

            if isinstance(l, str):
                try:
                    name = l.strip()
                    l = int(name[:-1])
                    if l < 0:
                        raise ValueError("Irrep degree must be non-negative.")
                    p = {
                        'e': 1,
                        'o': -1,
                        'y': (-1) ** l,
                    }[name[-1]]
                except Exception as exc:
                    raise ValueError(f"Cannot convert string {l} to Irrep.") from exc
            elif isinstance(l, tuple):
                l, p = l

        if not isinstance(l, int):
            raise TypeError("Irrep degree must be int.")
        if l < 0:
            raise ValueError("Irrep degree must be non-negative.")
        if p not in [-1, 1]:
            raise ValueError("Irrep parity must be 1 or -1.")
        object.__setattr__(self, "l", l)
        object.__setattr__(self, "p", p)

    def __repr__(self):
        """Representation of the Irrep."""
        p = {+1: 'e', -1: 'o'}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls, lmax=None):
        for l in itertools.count():
            yield Irrep(l, (-1) ** l)
            yield Irrep(l, -(-1) ** l)

            if l == lmax:
                break

    def wigD_from_angles(self, alpha, beta, gamma, k=None):
        r"""
        Compute the Wigner-D matrix representation of O(3) from the three Euler angles
        :math:`(\alpha, \beta, \gamma)` that describe the rotation sequence:

        1. Rotate by :math:`\gamma` around the original Y axis.
        2. Rotate by :math:`\beta` around the new X axis.
        3. Rotate by :math:`\alpha` around the newest Y axis.

        Args:
            alpha (Union[Tensor[float32], list[float], tuple[float],
                         ndarray[np.float32], float]):
                Rotation :math:`\alpha` around Y axis, applied third.
            beta (Union[Tensor[float32], list[float], tuple[float],
                        ndarray[np.float32], float]):
                Rotation :math:`\beta` around X axis, applied second.
            gamma (Union[Tensor[float32], list[float], tuple[float],
                         ndarray[np.float32], float]):
                Rotation :math:`\gamma` around Y axis, applied first.
            k (Union[None, Tensor[float32], list[float], tuple[float],
                     ndarray[np.float32], float], optional):
                How many times the parity is applied. Default: ``None`` .

        Returns:
            Tensor, representation wigner D matrix of O(3). The shape of Tensor is :math:`(..., 2l+1, 2l+1)` .

        Examples:
            >>> m = Irrep(1, -1).wigD_from_angles(0, 0 ,0, 1)
            >>> print(m)
            [[-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]]
        """
        if k is None:
            k = ops.zeros_like(_to_tensor(alpha))

        alpha, beta, gamma, k = broadcast_args(alpha, beta, gamma, k)
        return wigner_D(self.l, alpha, beta, gamma) * self.p ** _expand_last_dims(k)

    def wigD_from_matrix(self, R):
        r"""
        Compute the Wigner-D matrix representation of O(3) from rotation matrices.

        Args:
            R (Tensor): Rotation matrices. The shape of Tensor is :math:`(..., 3, 3)`.

        Returns:
            Tensor, representation wigner D matrix of O(3). The shape of Tensor is :math:`(..., 2l+1, 2l+1)`.

        Raises:
            TypeError: If `R` is not a Tensor.

        Examples:
            >>> from mindspore import ops
            >>> m = Irrep(1, -1).wigD_from_matrix(-ops.eye(3))
            >>> print(m)
            [[-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]]
        """
        if not isinstance(R, Tensor):
            raise TypeError("R must be a Tensor.")
        d = Tensor(np.sign(np.linalg.det(R.asnumpy())))
        R = _expand_last_dims(d) * R
        k = (1. - d) / 2
        return self.wigD_from_angles(*matrix_to_angles(R), k)

    @property
    def dim(self) -> int:
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        r"""
        Check whether this irrep is the trivial (scalar) representation.

        Returns:
            bool, True if `l = 0` and parity `p = 1`, False otherwise.
        """
        return self.l == 0 and self.p == 1

    def __mul__(self, other):
        r"""
        Generate the irreps from the product of two irreps.

        Returns:
            generator of `Irrep`.
        """
        other = Irrep(other)
        p = self.p * other.p
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p)

    def __rmul__(self, other):
        r"""
        Return `Irreps` of multiple `Irrep`.

        Args:
            other (int): multiple number of the `Irrep`.

        Returns:
            `Irreps`, corresponding multiple `Irrep`.

        Raises:
            TypeError: If `other` is not int.
        """
        if not isinstance(other, int):
            raise TypeError
        return Irreps([(other, self)])

    def __add__(self, other):
        r"""Sum of two irreps."""
        return Irreps(self) + Irreps(other)

    def __radd__(self, other):
        r"""Sum of two irreps."""
        return Irreps(other) + Irreps(self)

    def __iter__(self):
        r"""Deconstruct the irrep into ``l`` and ``p``."""
        yield self.l
        yield self.p

    def __lt__(self, other):
        r"""Compare the order of two irreps."""
        return (self.l, self.p) < (other.l, other.p)

    def __eq__(self, other):
        """Compare two irreps."""
        other = Irrep(other)
        return (self.l, self.p) == (other.l, other.p)


@jit_class
@dataclasses.dataclass(init=False, frozen=True)
class _MulIr:
    """Multiple Irrep."""
    mul: int
    ir: Irrep

    def __init__(self, mul, ir=None):
        if ir is None:
            mul, ir = mul

        if not (isinstance(mul, int) and isinstance(ir, Irrep)):
            raise TypeError
        object.__setattr__(self, "mul", mul)
        object.__setattr__(self, "ir", ir)

    @property
    def dim(self):
        return self.mul * self.ir.dim

    def __repr__(self):
        """Representation of the irrep."""
        return f"{self.mul}x{self.ir}"

    def __iter__(self):
        """Deconstruct the mulirrep into `mul` and `ir`."""
        yield self.mul
        yield self.ir

    def __lt__(self, other):
        """Compare the order of two mulirreps."""
        return (self.ir, self.mul) < (other.ir, other.mul)

    def __eq__(self, other):
        """Compare two irreps."""
        return (self.mul, self.ir) == (other.mul, other.ir)


@jit_class
@dataclasses.dataclass(init=False, frozen=False)
class Irreps:
    r"""
    Direct sum of irreducible representations of O(3).
    This class does not contain any data, it is a structure that
    describe the representation.
    It is typically used as argument of other classes of the library to
    define the input and output representations of functions.
    The irreps are stored as a tuple of (_MulIr) objects, each
    containing a multiplicity and an Irrep.
    This allows for easy manipulation, such as addition, multiplication,
    and filtering of representations.

    Args:
        irreps (Union[str, Irrep, Irreps, list[tuple[int]]], optional):
            A string to represent the direct sum of irreducible
            representations. Default: ``None``.

    Raises:
        ValueError: If `irreps` cannot be converted to an `Irreps`.
        ValueError: If the mul part of `irreps` part is negative.
        TypeError: If the mul part of `irreps` part is not int.

    Examples:
        >>> from mindscience.e3nn.o3 import Irreps
        >>> x = Irreps([(100, (0, 1)), (50, (1, 1))])
        100x0e+50x1e
        >>> x.dim
        250
        >>> Irreps("100x0e+50x1e+0x2e")
        100x0e+50x1e+0x2e
        >>> Irreps("100x0e+50x1e+0x2e").lmax
        1
        >>> Irrep("2e") in Irreps("0e+2e")
        True
        >>> Irreps(), Irreps("")
        (, )
        >>> Irreps('2x1o+1x0o') * Irreps('2x1o+1x0e')
        4x0e+1x0o+2x1o+4x1e+2x1e+4x2e
    """
    __slots__ = ('data', 'dim', 'slice', 'slice_tuples')

    def __init__(self, irreps=None):
        if isinstance(irreps, Irreps):
            self.data = irreps.data
            self.dim = irreps.dim
            self.slice = irreps.slice
            self.slice_tuples = irreps.slice_tuples
            return

        out = ()
        if isinstance(irreps, Irrep):
            out += (_MulIr(1, Irrep(irreps)),)
        elif isinstance(irreps, _MulIr):
            out += (irreps,)
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mir in irreps.split('+'):
                        if 'x' in mir:
                            mul, ir = mir.split('x')
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mir)

                        if not isinstance(mul, int):
                            raise TypeError("Irrep multiplicity must be int.")
                        if mul < 0:
                            raise ValueError(
                                "Irrep multiplicity must be non-negative."
                            )
                        out += (_MulIr(mul, ir),)
            except Exception as exc:
                raise ValueError("Irreps string format is invalid.") from exc
        elif irreps is None:
            out = ()
        else:
            out = self._handle_irreps(irreps, out)

        self.data = out
        self.dim = self._dim()
        self.slice = self._slices()
        self.slice_tuples = [(s.start, s.stop - s.start) for s in self.slice]

    def _handle_irreps(self, irreps, out):
        for mir in irreps:

            if isinstance(mir, str):
                if 'x' in mir:
                    mul, ir = mir.split('x')
                    mul = int(mul)
                    ir = Irrep(ir)
                else:
                    mul = 1
                    ir = Irrep(mir)
            elif isinstance(mir, Irrep):
                mul = 1
                ir = mir
            elif isinstance(mir, _MulIr):
                mul, ir = mir
            elif isinstance(mir, int):
                mul, ir = 1, Irrep(l=mir, p=1)
            elif len(mir) == 2:
                mul, ir = mir
                ir = Irrep(ir)

            if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                raise ValueError("Irreps format is invalid.")

            out += (_MulIr(mul, ir),)
        return out

    def __iter__(self):
        return iter(self.data)

    def __hash__(self):
        return hash(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        """Representation of the irreps."""
        return "+".join(f"{mir}" for mir in self.data)

    def __eq__(self, other):
        """Compare two irreps."""
        other = Irreps(other)
        if not len(self) == len(other):
            return False
        for m_1, m_2 in zip(self.data, other.data):
            if not m_1 == m_2:
                return False
        return True

    def __contains__(self, ir):
        """Check if an irrep or an irreps is in the representation."""
        try:
            ir = Irrep(ir)
            return ir in (irrep for _, irrep in self.data)
        except TypeError:
            irreps = Irreps(ir)
        except ValueError:
            irreps = Irreps(ir)

        m, n = len(irreps), len(self)
        mask = [False] * n

        def dfs(i):
            if i == m:
                return True
            for j in range(n):
                if not mask[j]:
                    if irreps.data[i].mul <= self.data[j].mul and irreps.data[i].ir == self.data[j].ir:
                        mask[j] = True
                        found = dfs(i + 1)
                        if found:
                            return True
                        mask[j] = False
            return False

        return dfs(0)

    def __add__(self, irreps):
        irreps = Irreps(irreps)
        return Irreps(self.data.__add__(irreps.data))

    def __mul__(self, other):
        r"""
        Return `Irreps` of multiple `Irreps`.

        Args:
            other (int): multiple number of the `Irreps`.

        Returns:
            `Irreps`, corresponding multiple `Irreps`.

        Raises:
            NotImplementedError: If `other` is `Irreps`, please use `o3.TensorProduct`.
        """
        if isinstance(other, Irreps):
            res = Irreps()
            for mir_1 in self.data:
                for mir_2 in other.data:
                    out_ir = mir_1.ir * mir_2.ir
                    for ir in out_ir:
                        res += mir_1.mul * mir_2.mul * ir
            res, _, _ = res.simplify().sort()
            return res
        return Irreps([(mul * other, ir) for mul, ir in self.data])

    def __rmul__(self, other):
        r"""
        Return repeated `Irreps` of multiple `Irreps`.

        Args:
            other (int): multiple number of the `Irreps`.

        Returns:
            `Irreps`, repeated multiple `Irreps`.
        """
        return self * other

    def _dim(self):
        """The dimension of the representation, :math:`2 l + 1`."""
        return sum(mul * ir.dim for mul, ir in self.data)

    @property
    def num_irreps(self):
        return sum(mul for mul, _ in self.data)

    @property
    def ls(self):
        res = []
        for mul, (l, _) in self.data:
            res.extend([l] * mul)
        return res

    @property
    def lmax(self):
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def count(self, ir):
        r"""
        Multiplicity of `ir`.

        Args:
            ir (Irrep): `Irrep`

        Returns:
            int, total multiplicity of `ir`.

        Examples:
            >>> Irreps("1o + 3x2e").count("2e")
            3
        """
        ir = Irrep(ir)
        res = 0
        for mul, irrep in self.data:
            if ir == irrep:
                res += mul
        return res

    def simplify(self):
        """
        Simplify the representations.

        Returns:
            `Irreps`, simplified `Irreps`.

        Examples:
            >>> Irreps("1e + 1e + 0e").simplify()
            2x1e+1x0e
            >>> Irreps("1e + 1e + 0e + 1e").simplify()
            2x1e+1x0e+1x1e
        """
        out = []
        for mul, ir in self.data:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self):
        """
        Remove any irreps with multiplicities of zero.

        Returns:
            `Irreps`, irreps with multiplicities of zero removed.

        Examples:
            >>> Irreps("4x0e + 0x1o + 2x3e").remove_zero_multiplicities()
            4x0e+2x3e
        """
        out = [(mul, ir) for mul, ir in self.data if mul > 0]
        return Irreps(out)

    def _slices(self):
        r"""
        List of slices corresponding to indices for each irrep.

        Examples:
            >>> Irreps('2x0e + 1e').slices()
            [slice(0, 2, None), slice(2, 5, None)]
        """
        s = []
        i = 0
        for mir in self.data:
            s.append(slice(i, i + mir.dim))
            i += mir.dim
        return s

    def sort(self):
        r"""
        Sort the representations by increasing degree. 

        Returns:
            - `Irreps`, sorted `Irreps`.
            - `p (tuple[int])`, permute orders, `p[old_index] = new_index`.
            - `inv (tuple[int])`, inversed permute orders, `p[new_index] = old_index`.

        Examples:
            >>> Irreps("1e + 0e + 1e").sort().irreps
            1x0e+1x1e+1x1e
            >>> Irreps("2o + 1e + 0e + 1e").sort().p
            (3, 1, 0, 2)
            >>> Irreps("2o + 1e + 0e + 1e").sort().inv
            (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self.data)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = _inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    def filter(self, keep=None, drop=None):
        r"""
        Filter the `Irreps` by either `keep` or `drop`.

        Args:
            keep (Union[str, Irrep, Irreps, list[str, Irrep]], optional): 
                list of irrep to keep. Default: ``None``.
            drop (Union[str, Irrep, Irreps, list[str, Irrep]], optional): 
                list of irrep to drop. Default: ``None``.

        Returns:
            `Irreps`, filtered irreps.

        Raises:
            ValueError: If both `keep` and `drop` are not `None`.

        Examples:
            >>> Irreps("1o + 2e").filter(keep="1o")
            1x1o
            >>> Irreps("1o + 2e").filter(drop="1o")
            1x2e
        """
        if keep is None and drop is None:
            return self
        if keep is not None and drop is not None:
            raise ValueError("Cannot specify both keep and drop")
        if keep is not None:
            keep = Irreps(keep).data
            keep = {mir.ir for mir in keep}
            return Irreps([(mul, ir) for mul, ir in self.data if ir in keep])
        if drop is not None:
            drop = Irreps(drop).data
            drop = {mir.ir for mir in drop}
            return Irreps([(mul, ir) for mul, ir in self.data if not ir in drop])
        return None

    def decompose(self, v, batch=False):
        r"""
        Decompose a vector into irreducible components according to the current `Irreps` structure.

        This method reshapes the last axis of the input tensor `v` such that each slice
        corresponds to one of the irreducible representations listed in `self`. The
        resulting list contains one tensor per irrep, with shape
        `(..., multiplicity, irrep_dimension)`.

        Args:
            v (Tensor): the vector to be decomposed.
            batch (bool, optional): 
                whether reshape the result such that there is at least a batch dimension. Default: ``False``.

        Returns:
            List of Tensors, the decomposed vectors by `Irreps`.

        Raises:
            TypeError: If v is not Tensor.
            ValueError: If length of the vector `v` is not matching with dimension of `Irreps`.

        Examples:
            >>> import mindspore as ms
            >>> input = ms.Tensor([1, 2, 3])
            >>> m = Irreps("1o").decompose(input)
            >>> print(m)
            [Tensor(shape=[1,3], dtype=Int64, value=
            [[1,2,3]])]
        """
        if not isinstance(v, Tensor):
            raise TypeError(
                f"The input for decompose should be Tensor, but got {type(v)}.")
        len_v = v.shape[-1]
        if not self.dim == len_v:
            raise ValueError(
                f"the shape of input {v.shape[-1]} do not match irreps dimension {self.dim}.")

        res = []
        batch_shape = v.shape[:-1]
        for (s, l), mir in zip(self.slice_tuples, self.data):
            v_slice = narrow(v, -1, s, l)
            if v.ndim == 1 and batch:
                res.append(v_slice.reshape(
                    (1,) + batch_shape + (mir.mul, mir.ir.dim)))
            else:
                res.append(v_slice.reshape(
                    batch_shape + (mir.mul, mir.ir.dim)))

        return res

    @staticmethod
    def spherical_harmonics(lmax, p=-1):
        r"""
        Representation of the spherical harmonics.

        Args:
            lmax (int): maximum of `l`.
            p (int, optional): {1, -1}, the parity of the representation. Default: ``-1``.

        Returns:
            `Irreps`, representation of :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`.

        Examples:
            >>> Irreps.spherical_harmonics(3)
            1x0e+1x1o+1x2e+1x3o
            >>> Irreps.spherical_harmonics(4, p=1)
            1x0e+1x1e+1x2e+1x3e+1x4e
        """
        return Irreps([(1, (l, p ** l)) for l in range(lmax + 1)])

    def randn(self, *size, normalization='component'):
        r"""
        Generate a random tensor whose last dimension matches the total dimension of these irreps.
        The irreps structure is used to split the last axis into individual irrep blocks,
        each of which can be normalized either per-component or per-irrep norm.

        Args:
            \*size (list[int]): size of the output tensor, needs to contains a `-1`.
            normalization (str, optional): {'component', 'norm'},
                type of normalization method. Default: ``'component'``.

        Returns:
            Tensor, the shape is `size` where `-1` is replaced by `self.dim`.

        Raises:
            ValueError: If `normalization` is not 'component' or 'norm'.

        Examples:
            >>> Irreps("5x0e + 10x1o").randn(5, -1, 5, normalization='norm').shape
            (5, 35, 5)
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1:]

        if normalization == 'component':
            return ops.standard_normal((*lsize, self.dim, *rsize))
        if normalization == 'norm':
            x_list = []
            for _, (mul, ir) in zip(self.slice, self.data):
                if mul < 1:
                    continue
                r = ops.standard_normal((*lsize, mul, ir.dim, *rsize))
                r = r / norm_keep(r, axis=di + 1)

                x_list.append(r.reshape((*lsize, -1, *rsize)))
            return ops.concat(x_list, axis=di)
        raise ValueError("Normalization needs to be 'norm' or 'component'")

    def wigD_from_angles(self, alpha, beta, gamma, k=None):
        r"""
        Compute the Wigner-D matrix representation of O(3) from the three Euler angles
        :math:`(\alpha, \beta, \gamma)` that describe the rotation sequence:

        1. Rotate by :math:`\gamma` around the original Y axis.
        2. Rotate by :math:`\beta` around the new X axis.
        3. Rotate by :math:`\alpha` around the newest Y axis.

        The result is the direct sum of the Wigner-D matrices for each
        irrep contained in this `Irreps` object, repeated according to
        multiplicity.

        Args:
            alpha (Union[Tensor[float32], list[float], tuple[float],
                         ndarray[np.float32], float]):
                rotation :math:`\alpha` around Y axis, applied third.
            beta (Union[Tensor[float32], list[float], tuple[float],
                        ndarray[np.float32], float]):
                rotation :math:`\beta` around X axis, applied second.
            gamma (Union[Tensor[float32], list[float], tuple[float],
                         ndarray[np.float32], float]):
                rotation :math:`\gamma` around Y axis, applied first.
            k (Union[None, Tensor[float32], list[float], tuple[float],
                     ndarray[np.float32], float], optional):
                How many times the parity is applied. Default: ``None``.

        Returns:
            Tensor, representation wigner D matrix of O(3). The shape of Tensor is :math:`(..., 2l+1, 2l+1)`

        Examples:
            >>> m = Irreps("1o").wigD_from_angles(0, 0 ,0, 1)
            >>> print(m)
            [[-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]]
        """
        return _direct_sum(
            *[
                ir.wigD_from_angles(alpha, beta, gamma, k)
                for mul, ir in self
                for _ in range(mul)
            ]
        )

    def wigD_from_matrix(self, R):
        r"""
        Compute Wigner-D matrices of O(3) from rotation matrices.

        Args:
            R (Tensor): Rotation matrices. The shape of Tensor is :math:`(..., 3, 3)`.

        Returns:
            Tensor, representation wigner D matrix of O(3). The shape of Tensor is :math:`(..., 2l+1, 2l+1)`

        Raises:
            TypeError: If `R` is not a Tensor.

        Examples:
            >>> m = Irreps("1o").wigD_from_matrix(-ops.eye(3))
            >>> print(m)
            [[-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]]
        """
        if not isinstance(R, Tensor):
            raise TypeError("R needs to be a Tensor")
        d = Tensor(np.sign(np.linalg.det(R.asnumpy())))
        R = _expand_last_dims(d) * R
        k = (1 - d) / 2
        return self.wigD_from_angles(*matrix_to_angles(R), k)
