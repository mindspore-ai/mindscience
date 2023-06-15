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


@jit_class
@dataclasses.dataclass(init=False, frozen=True)
class Irrep:
    r"""
    Irreducible representation of O(3). This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of functions.

    Args:
        l (Union[int, str]): non-negative integer, the degree of the representation, :math:`l = 0, 1, \dots`. Or string to indicate the degree and parity.
        p (int): {1, -1}, the parity of the representation.

    Raises:
        NotImplementedError: If method is not implemented.
        ValueError: If `l` is negative or `p` is not in {1, -1}.
        ValueError: If `l` cannot be converted to an `Irrep`.
        TypeError: If `l` is not int or str.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
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
                        raise ValueError
                    p = {
                        'e': 1,
                        'o': -1,
                        'y': (-1) ** l,
                    }[name[-1]]
                except Exception:
                    raise ValueError
            elif isinstance(l, tuple):
                l, p = l

        if not isinstance(l, int):
            raise TypeError
        elif l < 0:
            raise ValueError
        if p not in [-1, 1]:
            raise ValueError
        object.__setattr__(self, "l", l)
        object.__setattr__(self, "p", p)

    def __repr__(self):
        """Representation of the Irrep."""
        p = {+1: 'e', -1: 'o'}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls, lmax=None):
        r"""
        Iterator through all the irreps of :math:`O(3)`.

        Examples:
            >>> it = Irrep.iterator()
            >>> next(it), next(it), next(it), next(it)
            (0e, 0o, 1o, 1e)
        """
        for l in itertools.count():
            yield Irrep(l, (-1) ** l)
            yield Irrep(l, -(-1) ** l)

            if l == lmax:
                break

    def wigD_from_angles(self, alpha, beta, gamma, k=None):
        r"""
        Representation wigner D matrices of O(3) from Euler angles.

        Args:
            alpha (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\alpha` around Y axis, applied third.
                tensor of shape :math:`(...)`
            beta (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\beta` around X axis, applied second.
                tensor of shape :math:`(...)`
            gamma (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\gamma` around Y axis, applied first.
                tensor of shape :math:`(...)`
            k (Union[None, Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): How many times the parity is applied. Default: None.
                tensor of shape :math:`(...)`

        Returns:
            Tensor, representation wigner D matrix of O(3).
                tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        if k is None:
            k = ops.zeros_like(_to_tensor(alpha))

        alpha, beta, gamma, k = broadcast_args(alpha, beta, gamma, k)
        return wigner_D(self.l, alpha, beta, gamma) * self.p ** _expand_last_dims(k)

    def wigD_from_matrix(self, R):
        r"""
        Representation wigner D matrices of O(3) from rotation matrices.

        Arg:
        R (Tensor): rotation matrices.
            tensor of shape :math:`(..., 3, 3)`
        k (Union[None, Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): How many times the parity is applied. Default: None.
                tensor of shape :math:`(...)`

        Returns:
            Tensor, representation wigner D matrix of O(3).
                tensor of shape :math:`(..., 2l+1, 2l+1)`

        Raises:
            TypeError: If `R` is not a Tensor.
        """
        if not isinstance(R, Tensor):
            raise TypeError
        d = Tensor(np.sign(np.linalg.det(R.asnumpy())))
        R = _expand_last_dims(d) * R
        k = (1. - d) / 2
        return self.wigD_from_angles(*matrix_to_angles(R), k)

    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        """Equivalent to `l == 0 and p == 1`."""
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

        Arg:
            other (int): multiple number of the `Irrep`.

        Returns:
            `Irreps` - corresponding multiple `Irrep`.

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
        """The dimension of the representations."""
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
    Direct sum of irreducible representations of O(3). This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of functions.

    Args:
        irreps (Union[str, Irrep, Irreps, List[Tuple[int]]]): a string to represent the direct sum of irreducible representations.

    Raises:
        ValueError: If `irreps` cannot be converted to an `Irreps`.
        ValueError: If the mul part of `irreps` part is negative.
        TypeError: If the mul part of `irreps` part is not int.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
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
        else:
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
                                raise TypeError
                            elif mul < 0:
                                raise ValueError
                            out += (_MulIr(mul, ir),)
                except Exception:
                    raise ValueError
            elif irreps is None:
                pass
            else:
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
                        raise ValueError

                    out += (_MulIr(mul, ir),)
            self.data = out
            self.dim = self._dim()
            self.slice = self._slices()
            self.slice_tuples = [(s.start, s.stop - s.start) for s in self.slice]

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
        except:
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
            `Irreps` - corresponding multiple `Irreps`.

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
            res, p, _ = res.simplify().sort()
            return res
        return Irreps([(mul * other, ir) for mul, ir in self.data])

    def __rmul__(self, other):
        r"""
        Return repeated `Irreps` of multiple `Irreps`.

        Args:
            other (int): multiple number of the `Irreps`.

        Returns:
            `Irreps` - repeated multiple `Irreps`.
        """
        return self * other

    def _dim(self):
        """The dimension of the representation, :math:`2 l + 1`."""
        return sum(mul * ir.dim for mul, ir in self.data)

    @property
    def num_irreps(self):
        """The total multiplications for each irrep."""
        return sum(mul for mul, _ in self.data)

    @property
    def ls(self):
        """List of degrees for each irrep."""
        res = []
        for mul, (l, _) in self.data:
            res.extend([l] * mul)
        return res

    @property
    def lmax(self):
        """Max degree of `Irreps`."""
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def count(self, ir):
        r"""
        Multiplicity of `ir`.

        Warning: do not suppose GRAPH_MODE in construct functions.

        Args:
            ir (Irrep): `Irrep`

        Returns:
            int, total multiplicity of `ir`.
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
            `Irreps`

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
            `Irreps`

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
            irreps (`Irreps`) - sorted `Irreps`
            p (tuple[int]) - permute orders. `p[old_index] = new_index`
            inv (tuple[int]) - inversed permute orders. `p[new_index] = old_index`

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

        Arg:
            keep (Union[str, Irrep, Irreps, List[str, Irrep]]): list of irrep to keep. Default: None.
            drop (Union[str, Irrep, Irreps, List[str, Irrep]]): list of irrep to drop. Default: None.

        Returns:
            `Irreps`, filtered irreps.

        Raises:
            ValueError: If both `keep` and `drop` are not `None`.
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
        Decompose a vector by `Irreps`.

        Args:
            v (Tensor): the vector to be decomposed.
            batch (bool): whether reshape the result such that there is at least a batch dimension. Default: `False`.

        Returns:
            List of Tensors, the decomposed vectors by `Irreps`.

        Raises:
            TypeError: If v is not Tensor.
            ValueError: If length of the vector `v` is not matching with dimension of `Irreps`.
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
            p (int): {1, -1}, the parity of the representation.

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
        Random tensor.

        Args:
            *size (List[int]): size of the output tensor, needs to contains a `-1`.
            normalization (str): {'component', 'norm'}, type of normalization method.

        Returns:
            Tensor, tensor of shape `size` where `-1` is replaced by `self.dim`.

        Examples:
            >>> Irreps("5x0e + 10x1o").randn(5, -1, 5, normalization='norm').shape
            (5, 35, 5)
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1:]

        if normalization == 'component':
            return ops.standard_normal((*lsize, self.dim, *rsize))
        elif normalization == 'norm':
            x_list = []
            for s, (mul, ir) in zip(self.slice, self.data):
                if mul < 1:
                    continue
                r = ops.standard_normal((*lsize, mul, ir.dim, *rsize))
                r = r / norm_keep(r, axis=di + 1)

                x_list.append(r.reshape((*lsize, -1, *rsize)))
            return ops.concat(x_list, axis=di)
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def wigD_from_angles(self, alpha, beta, gamma, k=None):
        r"""
        Representation wigner D matrices of O(3) from Euler angles.

        Args:
            alpha (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\alpha` around Y axis, applied third.
                tensor of shape :math:`(...)`
            beta (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\beta` around X axis, applied second.
                tensor of shape :math:`(...)`
            gamma (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\gamma` around Y axis, applied first.
                tensor of shape :math:`(...)`
            k (Union[None, Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): How many times the parity is applied. Default: None.
                tensor of shape :math:`(...)`

        Returns:
            Tensor, representation wigner D matrix of O(3).
                tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return _direct_sum(*[ir.wigD_from_angles(alpha, beta, gamma, k) for mul, ir in self for _ in range(mul)])

    def wigD_from_matrix(self, R):
        r"""
        Representation wigner D matrices of O(3) from rotation matrices.

        Args:
        R (Tensor): rotation matrices.
            tensor of shape :math:`(..., 3, 3)`
        k (Union[None, Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): How many times the parity is applied. Default: None.
                tensor of shape :math:`(...)`

        Returns:
            Tensor, representation wigner D matrix of O(3).
                tensor of shape :math:`(..., 2l+1, 2l+1)`

        Raises:
            TypeError: If `R` is not a Tensor.
        """
        if not isinstance(R, Tensor):
            raise TypeError
        d = Tensor(np.sign(np.linalg.det(R.asnumpy())))
        R = _expand_last_dims(d) * R
        k = (1 - d) / 2
        return self.wigD_from_angles(*matrix_to_angles(R), k)
