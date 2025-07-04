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
"""SphericalHarmonics"""
from mindspore import Tensor, nn, ops, float32
from .irreps import Irreps


def _sqrt(x, dtype=float32):
    sqrt = ops.Sqrt()
    return sqrt(Tensor(x, dtype=dtype))


class SphericalHarmonics(nn.Cell):
    r"""
    Return Spherical harmonics layer.

    Args:
        irreps_out (Union[str, `Irreps`]): irreducible representations of output for spherical harmonics.
        normalize (bool): whether to normalize the input Tensor to unit vectors that lie on the sphere before
            projecting onto the spherical harmonics.
        normalization (str): {'integral', 'component', 'norm'}, normalization method of the output tensors.
            Default: ``'integral'``.
        irreps_in (Union[str, `Irreps`, None]): irreducible representations of input for spherical harmonics.
            Default: ``None``.
        dtype (mindspore.dtype): The type of input tensor. Default: ``mindspore.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor for construct spherical harmonics. The shape of Tensor is :math:`(..., 3)`.

    Outputs:
        - **output** (Tensor) - the spherical harmonics :math:`Y^l(x)`. The shape of Tensor is :math:`(..., 2l+1)`.

    Raise:
        ValueError: If `normalization` is not in {'integral', 'component', 'norm'}.
        ValueError: If `irreps_in` for SphericalHarmonics is not neither a vector (`1x1o`) nor a pseudovector (`1x1e`).
        ValueError: If the `l` and `p` of `irreps_out` are not consistent with `irreps_in` for spherical harmonics.
            The output parity should have been p = {input_p**l}.
        NotImplementedError: If `l` is larger than 11.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.o3 import SphericalHarmonics
        >>> from mindspore import ops
        >>> sh = SphericalHarmonics(0, False, normalization='component')
        >>> x = ops.rand(2,3)
        >>> m = sh(x)
        [[1.]
        [1.]]
    """

    def __init__(self, irreps_out, normalize, normalization='integral', irreps_in=None, dtype=float32):
        super().__init__()
        self.normalize = normalize
        self.normalization = normalization
        if normalization not in ['integral', 'component', 'norm']:
            raise ValueError

        if isinstance(irreps_out, str):
            irreps_out = Irreps(irreps_out)
        if isinstance(irreps_out, Irreps) and irreps_in is None:
            for mul, (l, p) in irreps_out:
                if l % 2 == 1 and p == 1:
                    irreps_in = Irreps("1e")
        if irreps_in is None:
            irreps_in = Irreps("1o")

        irreps_in = Irreps(irreps_in)
        if irreps_in not in (Irreps("1x1o"), Irreps("1x1e")):
            raise ValueError
        self.irreps_in = irreps_in
        input_p = irreps_in.data[0].ir.p

        if isinstance(irreps_out, Irreps):
            ls = []
            for mul, (l, p) in irreps_out:
                if p != input_p ** l:
                    raise ValueError
                ls.extend([l] * mul)
        elif isinstance(irreps_out, int):
            ls = [irreps_out]
        else:
            ls = list(irreps_out)

        irreps_out = Irreps([(1, (l, input_p ** l)) for l in ls]).simplify()
        self.irreps_out = irreps_out
        self._ls_list = ls
        self._lmax = max(ls)
        self._is_range_lmax = ls == list(range(max(ls) + 1))
        self._prof_str = f'spherical_harmonics({ls})'
        self.ones = ops.Ones()

        if self.normalization == 'integral':
            self.norm_factors = [
                (_sqrt(2 * l + 1., dtype) / 3.5449077018110318) *
                self.ones(2 * l + 1, dtype)
                for l in self._ls_list
            ]
        elif self.normalization == 'component':
            self.norm_factors = [
                _sqrt(2 * l + 1., dtype) * self.ones(2 * l + 1, dtype)
                for l in self._ls_list
            ]

        self.l2_normalize = ops.L2Normalize(axis=-1, epsilon=0.000000000001)

    def construct(self, x):
        """
        Compute spherical harmonics of vector `x`.

        Args:
            x (Tensor): Tensor for construct spherical harmonics. The shape of Tensor is :math:`x` of shape ``(..., 3)``

        Returns:
            Tensor, the spherical harmonics :math:`Y^l(x)`. The shape of Tensor is ``(..., 2l+1)``

        Examples:
            >>> sh = SphericalHarmonics(irreps_out="1o + 2x2e", normalize=True)
            >>> input = ops.ones([1,3])
            >>> output = sh(input)
            >>> print(output)
            [[0.28209478 0.28209478 0.28209478 0.36418277 0.36418277 0
              0.36418277 0          0.36418277 0.36418277 0 0.36418277
              0]]
        """
        last_dim = x.shape[-1]
        if not last_dim == 3:
            raise ValueError

        if self.normalize:
            x = self.l2_normalize(x)

        sh = _spherical_harmonics(self._lmax, x[..., 0], x[..., 1], x[..., 2])

        if not self._is_range_lmax:
            sh = ops.concat([
                sh[..., l * l:(l + 1) * (l + 1)]
                for l in self._ls_list
            ], axis=-1)
        if self.normalization != 'norm':
            sh = ops.mul(sh, ops.concat(self.norm_factors))

        return sh

    def __repr__(self):
        return f'SphericalHarmonics {self._ls_list} ({self.irreps_in} -> {self.irreps_out})'


def spherical_harmonics(l, x, normalize=True, normalization='integral'):
    r"""
    Compute spherical harmonics.

    Spherical harmonics are polynomials defined on the 3d space :
        math:`Y^l: \mathbb{R}^3 \longrightarrow \mathbb{R}^{2l+1}`
    Usually restricted on the sphere (with ``normalize=True``) :
        math:`Y^l: S^2 \longrightarrow \mathbb{R}^{2l+1}`
    who satisfies the following properties:
        - are polynomials of the cartesian coordinates ``x, y, z``
        - is equivariant :math:`Y^l(R x) = D^l(R) Y^l(x)`
        - are orthogonal :math:`\int_{S^2} Y^l_m(x) Y^j_n(x) dx = \text{cste} \; \delta_{lj} \delta_{mn}`
    The value of the constant depends on the choice of normalization.

    It obeys the following property:
    .. math::
        Y^{l+1}_i(x) &= \text{cste}(l) \; & C_{ijk} Y^l_j(x) x_k
        \partial_k Y^{l+1}_i(x) &= \text{cste}(l) \; (l+1) & C_{ijk} Y^l_j(x)
    Where :math:`C` are the `wigner_3j`.

    Args:
        l (Union[int, List[int]]): degree of the spherical harmonics.
        x (Tensor): tensor for construct spherical harmonics.
            The shape of Tensor is :math:`x` of shape ``(..., 3)``
        normalize (bool): whether to normalize the ``x`` to unit vectors that lie on the sphere before projecting onto
            the spherical harmonics.
        normalization (str): {'integral', 'component', 'norm'}, normalization method of the output tensors.
            Default: 'intergral'.
            'component': :math:`\|Y^l(x)\|^2 = 2l+1, x \in S^2`
            'norm': :math:`\|Y^l(x)\| = 1, x \in S^2`, ``component / sqrt(2l+1)``
            'integral': :math:`\int_{S^2} Y^l_m(x)^2 dx = 1`, ``component / sqrt(4pi)``

    Returns:
        Tensor, the spherical harmonics :math:`Y^l(x)`. The shape of Tensor is ``(..., 2l+1)``.

    Raise:
        ValueError: If `normalization` is not in {'integral', 'component', 'norm'}.
        ValueError: If `irreps_in` for SphericalHarmonics is not neither a vector (`1x1o`) nor a pseudovector (`1x1e`).
        ValueError: If the `l` and `p` of `irreps_out` are not consistent with `irreps_in` for spherical harmonics.
            The output parity should have been p = {input_p**l}.
        ValueError: If the tensor `x` is not the shape of ``(..., 3)``.
        NotImplementedError: If `l` is larger than 11.

    """
    sh = SphericalHarmonics(l, normalize, normalization, dtype=x.dtype)
    return sh(x)


def _spherical_harmonics(lmax: int, x, y, z):
    """core functions of spherical harmonics"""

    sh_0_0 = ops.ones_like(x)
    if lmax == 0:
        return ops.stack([
            sh_0_0,
        ], axis=-1)

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2
        ], axis=-1)

    sh_2_0 = 1.7320508075688772 * x * z
    sh_2_1 = 1.7320508075688772 * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = 1.7320508075688772 * y * z
    sh_2_4 = 1.7320508075688772 / 2.0 * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], axis=-1)

    sh_3_0 = 0.9128709291752769 * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = 2.23606797749979 * sh_2_0 * y
    sh_3_2 = 0.6123724356957945 * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = 0.6123724356957945 * z * (4.0 * y2 - x2z2)
    sh_3_5 = 2.23606797749979 * sh_2_4 * y
    sh_3_6 = 0.9128709291752769 * (sh_2_4 * z - sh_2_0 * x)

    if lmax == 3:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
        ], axis=-1)

    sh_4_0 = 0.935414346693485 * sh_3_0 * z + 0.935414346693485 * sh_3_6 * x
    sh_4_1 = 0.661437827766148 * sh_3_0 * y + 0.810092587300982 * \
             sh_3_1 * z + 0.810092587300983 * sh_3_5 * x
    sh_4_2 = -0.176776695296637 * sh_3_0 * z + 0.866025403784439 * sh_3_1 * y + \
             0.684653196881458 * sh_3_2 * z + 0.684653196881457 * \
             sh_3_4 * x + 0.176776695296637 * sh_3_6 * x
    sh_4_3 = -0.306186217847897 * sh_3_1 * z + 0.968245836551855 * sh_3_2 * \
             y + 0.790569415042095 * sh_3_3 * x + 0.306186217847897 * sh_3_5 * x
    sh_4_4 = -0.612372435695795 * sh_3_2 * x + \
             sh_3_3 * y - 0.612372435695795 * sh_3_4 * z
    sh_4_5 = -0.306186217847897 * sh_3_1 * x + 0.790569415042096 * sh_3_3 * \
             z + 0.968245836551854 * sh_3_4 * y - 0.306186217847897 * sh_3_5 * z
    sh_4_6 = -0.176776695296637 * sh_3_0 * x - 0.684653196881457 * sh_3_2 * x + \
             0.684653196881457 * sh_3_4 * z + 0.866025403784439 * \
             sh_3_5 * y - 0.176776695296637 * sh_3_6 * z
    sh_4_7 = -0.810092587300982 * sh_3_1 * x + 0.810092587300982 * \
             sh_3_5 * z + 0.661437827766148 * sh_3_6 * y
    sh_4_8 = -0.935414346693485 * sh_3_0 * x + 0.935414346693486 * sh_3_6 * z
    if lmax == 4:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
        ], axis=-1)

    sh_5_0 = 0.948683298050513 * sh_4_0 * z + 0.948683298050513 * sh_4_8 * x
    sh_5_1 = 0.6 * sh_4_0 * y + 0.848528137423857 * \
             sh_4_1 * z + 0.848528137423858 * sh_4_7 * x
    sh_5_2 = -0.14142135623731 * sh_4_0 * z + 0.8 * sh_4_1 * y + 0.748331477354788 * \
             sh_4_2 * z + 0.748331477354788 * sh_4_6 * x + 0.14142135623731 * sh_4_8 * x
    sh_5_3 = -0.244948974278318 * sh_4_1 * z + 0.916515138991168 * sh_4_2 * y + \
             0.648074069840786 * sh_4_3 * z + 0.648074069840787 * \
             sh_4_5 * x + 0.244948974278318 * sh_4_7 * x
    sh_5_4 = -0.346410161513776 * sh_4_2 * z + 0.979795897113272 * sh_4_3 * \
             y + 0.774596669241484 * sh_4_4 * x + 0.346410161513776 * sh_4_6 * x
    sh_5_5 = -0.632455532033676 * sh_4_3 * x + \
             sh_4_4 * y - 0.632455532033676 * sh_4_5 * z
    sh_5_6 = -0.346410161513776 * sh_4_2 * x + 0.774596669241483 * sh_4_4 * \
             z + 0.979795897113273 * sh_4_5 * y - 0.346410161513776 * sh_4_6 * z
    sh_5_7 = -0.244948974278318 * sh_4_1 * x - 0.648074069840787 * sh_4_3 * x + \
             0.648074069840786 * sh_4_5 * z + 0.916515138991169 * \
             sh_4_6 * y - 0.244948974278318 * sh_4_7 * z
    sh_5_8 = -0.141421356237309 * sh_4_0 * x - 0.748331477354788 * sh_4_2 * x + \
             0.748331477354788 * sh_4_6 * z + 0.8 * \
             sh_4_7 * y - 0.141421356237309 * sh_4_8 * z
    sh_5_9 = -0.848528137423857 * sh_4_1 * x + \
             0.848528137423857 * sh_4_7 * z + 0.6 * sh_4_8 * y
    sh_5_10 = -0.948683298050513 * sh_4_0 * x + 0.948683298050513 * sh_4_8 * z
    if lmax == 5:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
        ], axis=-1)

    sh_6_0 = 0.957427107756337 * sh_5_0 * z + 0.957427107756338 * sh_5_10 * x
    sh_6_1 = 0.552770798392565 * sh_5_0 * y + 0.874007373475125 * \
             sh_5_1 * z + 0.874007373475125 * sh_5_9 * x
    sh_6_2 = -0.117851130197757 * sh_5_0 * z + 0.745355992499929 * sh_5_1 * y + \
             0.117851130197758 * sh_5_10 * x + 0.790569415042094 * \
             sh_5_2 * z + 0.790569415042093 * sh_5_8 * x
    sh_6_3 = -0.204124145231931 * sh_5_1 * z + 0.866025403784437 * sh_5_2 * y + \
             0.707106781186546 * sh_5_3 * z + 0.707106781186547 * \
             sh_5_7 * x + 0.204124145231931 * sh_5_9 * x
    sh_6_4 = -0.288675134594813 * sh_5_2 * z + 0.942809041582062 * sh_5_3 * y + \
             0.623609564462323 * sh_5_4 * z + 0.623609564462322 * \
             sh_5_6 * x + 0.288675134594812 * sh_5_8 * x
    sh_6_5 = -0.372677996249965 * sh_5_3 * z + 0.986013297183268 * sh_5_4 * \
             y + 0.763762615825972 * sh_5_5 * x + 0.372677996249964 * sh_5_7 * x
    sh_6_6 = -0.645497224367901 * sh_5_4 * x + \
             sh_5_5 * y - 0.645497224367902 * sh_5_6 * z
    sh_6_7 = -0.372677996249964 * sh_5_3 * x + 0.763762615825972 * sh_5_5 * \
             z + 0.986013297183269 * sh_5_6 * y - 0.372677996249965 * sh_5_7 * z
    sh_6_8 = -0.288675134594813 * sh_5_2 * x - 0.623609564462323 * sh_5_4 * x + \
             0.623609564462323 * sh_5_6 * z + 0.942809041582062 * \
             sh_5_7 * y - 0.288675134594812 * sh_5_8 * z
    sh_6_9 = -0.20412414523193 * sh_5_1 * x - 0.707106781186546 * sh_5_3 * x + \
             0.707106781186547 * sh_5_7 * z + 0.866025403784438 * \
             sh_5_8 * y - 0.204124145231931 * sh_5_9 * z
    sh_6_10 = -0.117851130197757 * sh_5_0 * x - 0.117851130197757 * sh_5_10 * z - \
              0.790569415042094 * sh_5_2 * x + 0.790569415042093 * \
              sh_5_8 * z + 0.745355992499929 * sh_5_9 * y
    sh_6_11 = -0.874007373475124 * sh_5_1 * x + 0.552770798392566 * \
              sh_5_10 * y + 0.874007373475125 * sh_5_9 * z
    sh_6_12 = -0.957427107756337 * sh_5_0 * x + 0.957427107756336 * sh_5_10 * z
    if lmax == 6:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
        ], axis=-1)

    sh_7_0 = 0.963624111659433 * sh_6_0 * z + 0.963624111659432 * sh_6_12 * x
    sh_7_1 = 0.515078753637713 * sh_6_0 * y + 0.892142571199771 * \
             sh_6_1 * z + 0.892142571199771 * sh_6_11 * x
    sh_7_2 = -0.101015254455221 * sh_6_0 * z + 0.699854212223765 * sh_6_1 * y + \
             0.82065180664829 * sh_6_10 * x + 0.101015254455222 * \
             sh_6_12 * x + 0.82065180664829 * sh_6_2 * z
    sh_7_3 = -0.174963553055942 * sh_6_1 * z + 0.174963553055941 * sh_6_11 * x + \
             0.82065180664829 * sh_6_2 * y + 0.749149177264394 * \
             sh_6_3 * z + 0.749149177264394 * sh_6_9 * x
    sh_7_4 = 0.247435829652697 * sh_6_10 * x - 0.247435829652697 * sh_6_2 * z + \
             0.903507902905251 * sh_6_3 * y + 0.677630927178938 * \
             sh_6_4 * z + 0.677630927178938 * sh_6_8 * x
    sh_7_5 = -0.31943828249997 * sh_6_3 * z + 0.95831484749991 * sh_6_4 * y + \
             0.606091526731326 * sh_6_5 * z + 0.606091526731326 * \
             sh_6_7 * x + 0.31943828249997 * sh_6_9 * x
    sh_7_6 = -0.391230398217976 * sh_6_4 * z + 0.989743318610787 * sh_6_5 * \
             y + 0.755928946018454 * sh_6_6 * x + 0.391230398217975 * sh_6_8 * x
    sh_7_7 = -0.654653670707977 * sh_6_5 * x + \
             sh_6_6 * y - 0.654653670707978 * sh_6_7 * z
    sh_7_8 = -0.391230398217976 * sh_6_4 * x + 0.755928946018455 * sh_6_6 * \
             z + 0.989743318610787 * sh_6_7 * y - 0.391230398217975 * sh_6_8 * z
    sh_7_9 = -0.31943828249997 * sh_6_3 * x - 0.606091526731327 * sh_6_5 * x + \
             0.606091526731326 * sh_6_7 * z + 0.95831484749991 * \
             sh_6_8 * y - 0.31943828249997 * sh_6_9 * z
    sh_7_10 = -0.247435829652697 * sh_6_10 * z - 0.247435829652697 * sh_6_2 * x - \
              0.677630927178938 * sh_6_4 * x + 0.677630927178938 * \
              sh_6_8 * z + 0.903507902905251 * sh_6_9 * y
    sh_7_11 = -0.174963553055942 * sh_6_1 * x + 0.820651806648289 * sh_6_10 * y - \
              0.174963553055941 * sh_6_11 * z - 0.749149177264394 * \
              sh_6_3 * x + 0.749149177264394 * sh_6_9 * z
    sh_7_12 = -0.101015254455221 * sh_6_0 * x + 0.82065180664829 * sh_6_10 * z + \
              0.699854212223766 * sh_6_11 * y - 0.101015254455221 * \
              sh_6_12 * z - 0.82065180664829 * sh_6_2 * x
    sh_7_13 = -0.892142571199772 * sh_6_1 * x + 0.892142571199772 * \
              sh_6_11 * z + 0.515078753637713 * sh_6_12 * y
    sh_7_14 = -0.963624111659431 * sh_6_0 * x + 0.963624111659433 * sh_6_12 * z
    if lmax == 7:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12,
            sh_7_13, sh_7_14
        ], axis=-1)

    sh_8_0 = 0.968245836551854 * sh_7_0 * z + 0.968245836551853 * sh_7_14 * x
    sh_8_1 = 0.484122918275928 * sh_7_0 * y + 0.90571104663684 * \
             sh_7_1 * z + 0.90571104663684 * sh_7_13 * x
    sh_8_2 = -0.0883883476483189 * sh_7_0 * z + 0.661437827766148 * sh_7_1 * y + \
             0.843171097702002 * sh_7_12 * x + 0.088388347648318 * \
             sh_7_14 * x + 0.843171097702003 * sh_7_2 * z
    sh_8_3 = -0.153093108923948 * sh_7_1 * z + 0.7806247497998 * sh_7_11 * x + \
             0.153093108923949 * sh_7_13 * x + 0.7806247497998 * \
             sh_7_2 * y + 0.780624749799799 * sh_7_3 * z
    sh_8_4 = 0.718070330817253 * sh_7_10 * x + 0.21650635094611 * sh_7_12 * x - \
             0.21650635094611 * sh_7_2 * z + 0.866025403784439 * \
             sh_7_3 * y + 0.718070330817254 * sh_7_4 * z
    sh_8_5 = 0.279508497187474 * sh_7_11 * x - 0.279508497187474 * sh_7_3 * z + \
             0.927024810886958 * sh_7_4 * y + 0.655505530106345 * \
             sh_7_5 * z + 0.655505530106344 * sh_7_9 * x
    sh_8_6 = 0.342326598440729 * sh_7_10 * x - 0.342326598440729 * sh_7_4 * z + \
             0.968245836551854 * sh_7_5 * y + 0.592927061281572 * \
             sh_7_6 * z + 0.592927061281571 * sh_7_8 * x
    sh_8_7 = -0.405046293650492 * sh_7_5 * z + 0.992156741649221 * \
             sh_7_6 * y + 0.75 * sh_7_7 * x + 0.405046293650492 * sh_7_9 * x
    sh_8_8 = -0.661437827766148 * sh_7_6 * x + \
             sh_7_7 * y - 0.661437827766148 * sh_7_8 * z
    sh_8_9 = -0.405046293650492 * sh_7_5 * x + 0.75 * sh_7_7 * z + \
             0.992156741649221 * sh_7_8 * y - 0.405046293650491 * sh_7_9 * z
    sh_8_10 = -0.342326598440728 * sh_7_10 * z - 0.342326598440729 * sh_7_4 * x - \
              0.592927061281571 * sh_7_6 * x + 0.592927061281571 * \
              sh_7_8 * z + 0.968245836551855 * sh_7_9 * y
    sh_8_11 = 0.927024810886958 * sh_7_10 * y - 0.279508497187474 * sh_7_11 * z - \
              0.279508497187474 * sh_7_3 * x - 0.655505530106345 * \
              sh_7_5 * x + 0.655505530106345 * sh_7_9 * z
    sh_8_12 = 0.718070330817253 * sh_7_10 * z + 0.866025403784439 * sh_7_11 * y - \
              0.216506350946109 * sh_7_12 * z - 0.216506350946109 * \
              sh_7_2 * x - 0.718070330817254 * sh_7_4 * x
    sh_8_13 = -0.153093108923948 * sh_7_1 * x + 0.7806247497998 * sh_7_11 * z + \
              0.7806247497998 * sh_7_12 * y - 0.153093108923948 * \
              sh_7_13 * z - 0.780624749799799 * sh_7_3 * x
    sh_8_14 = -0.0883883476483179 * sh_7_0 * x + 0.843171097702002 * sh_7_12 * z + \
              0.661437827766147 * sh_7_13 * y - 0.088388347648319 * \
              sh_7_14 * z - 0.843171097702002 * sh_7_2 * x
    sh_8_15 = -0.90571104663684 * sh_7_1 * x + 0.90571104663684 * \
              sh_7_13 * z + 0.484122918275927 * sh_7_14 * y
    sh_8_16 = -0.968245836551853 * sh_7_0 * x + 0.968245836551855 * sh_7_14 * z
    if lmax == 8:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12,
            sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12,
            sh_8_13, sh_8_14, sh_8_15, sh_8_16
        ], axis=-1)

    sh_9_0 = 0.97182531580755 * sh_8_0 * z + 0.971825315807551 * sh_8_16 * x
    sh_9_1 = 0.458122847290851 * sh_8_0 * y + 0.916245694581702 * \
             sh_8_1 * z + 0.916245694581702 * sh_8_15 * x
    sh_9_2 = -0.078567420131839 * sh_8_0 * z + 0.62853936105471 * sh_8_1 * y + 0.86066296582387 * \
             sh_8_14 * x + 0.0785674201318385 * sh_8_16 * x + 0.860662965823871 * sh_8_2 * z
    sh_9_3 = -0.136082763487955 * sh_8_1 * z + 0.805076485899413 * sh_8_13 * x + \
             0.136082763487954 * sh_8_15 * x + 0.74535599249993 * \
             sh_8_2 * y + 0.805076485899413 * sh_8_3 * z
    sh_9_4 = 0.749485420179558 * sh_8_12 * x + 0.192450089729875 * sh_8_14 * x - \
             0.192450089729876 * sh_8_2 * z + 0.831479419283099 * \
             sh_8_3 * y + 0.749485420179558 * sh_8_4 * z
    sh_9_5 = 0.693888666488711 * sh_8_11 * x + 0.248451997499977 * sh_8_13 * x - \
             0.248451997499976 * sh_8_3 * z + 0.895806416477617 * \
             sh_8_4 * y + 0.69388866648871 * sh_8_5 * z
    sh_9_6 = 0.638284738504225 * sh_8_10 * x + 0.304290309725092 * sh_8_12 * x - \
             0.304290309725092 * sh_8_4 * z + 0.942809041582063 * \
             sh_8_5 * y + 0.638284738504225 * sh_8_6 * z
    sh_9_7 = 0.360041149911548 * sh_8_11 * x - 0.360041149911548 * sh_8_5 * z + \
             0.974996043043569 * sh_8_6 * y + 0.582671582316751 * \
             sh_8_7 * z + 0.582671582316751 * sh_8_9 * x
    sh_9_8 = 0.415739709641549 * sh_8_10 * x - 0.415739709641549 * sh_8_6 * \
             z + 0.993807989999906 * sh_8_7 * y + 0.74535599249993 * sh_8_8 * x
    sh_9_9 = -0.66666666666666666667 * sh_8_7 * x + \
             sh_8_8 * y - 0.66666666666666666667 * sh_8_9 * z
    sh_9_10 = -0.415739709641549 * sh_8_10 * z - 0.415739709641549 * sh_8_6 * \
              x + 0.74535599249993 * sh_8_8 * z + 0.993807989999906 * sh_8_9 * y
    sh_9_11 = 0.974996043043568 * sh_8_10 * y - 0.360041149911547 * sh_8_11 * z - \
              0.360041149911548 * sh_8_5 * x - 0.582671582316751 * \
              sh_8_7 * x + 0.582671582316751 * sh_8_9 * z
    sh_9_12 = 0.638284738504225 * sh_8_10 * z + 0.942809041582063 * sh_8_11 * y - \
              0.304290309725092 * sh_8_12 * z - 0.304290309725092 * \
              sh_8_4 * x - 0.638284738504225 * sh_8_6 * x
    sh_9_13 = 0.693888666488711 * sh_8_11 * z + 0.895806416477617 * sh_8_12 * y - \
              0.248451997499977 * sh_8_13 * z - 0.248451997499977 * \
              sh_8_3 * x - 0.693888666488711 * sh_8_5 * x
    sh_9_14 = 0.749485420179558 * sh_8_12 * z + 0.831479419283098 * sh_8_13 * y - \
              0.192450089729875 * sh_8_14 * z - 0.192450089729875 * \
              sh_8_2 * x - 0.749485420179558 * sh_8_4 * x
    sh_9_15 = -0.136082763487954 * sh_8_1 * x + 0.805076485899413 * sh_8_13 * z + \
              0.745355992499929 * sh_8_14 * y - 0.136082763487955 * \
              sh_8_15 * z - 0.805076485899413 * sh_8_3 * x
    sh_9_16 = -0.0785674201318389 * sh_8_0 * x + 0.86066296582387 * sh_8_14 * z + \
              0.628539361054709 * sh_8_15 * y - 0.0785674201318387 * \
              sh_8_16 * z - 0.860662965823871 * sh_8_2 * x
    sh_9_17 = -0.9162456945817 * sh_8_1 * x + 0.916245694581702 * \
              sh_8_15 * z + 0.458122847290851 * sh_8_16 * y
    sh_9_18 = -0.97182531580755 * sh_8_0 * x + 0.97182531580755 * sh_8_16 * z
    if lmax == 9:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12,
            sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12,
            sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12,
            sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18
        ], axis=-1)

    sh_10_0 = 0.974679434480897 * sh_9_0 * z + 0.974679434480897 * sh_9_18 * x
    sh_10_1 = 0.435889894354067 * sh_9_0 * y + 0.924662100445347 * \
              sh_9_1 * z + 0.924662100445347 * sh_9_17 * x
    sh_10_2 = -0.0707106781186546 * sh_9_0 * z + 0.6 * sh_9_1 * y + 0.874642784226796 * \
              sh_9_16 * x + 0.070710678118655 * sh_9_18 * x + 0.874642784226795 * sh_9_2 * z
    sh_10_3 = -0.122474487139159 * sh_9_1 * z + 0.824621125123533 * sh_9_15 * x + \
              0.122474487139159 * sh_9_17 * x + 0.714142842854285 * \
              sh_9_2 * y + 0.824621125123533 * sh_9_3 * z
    sh_10_4 = 0.774596669241484 * sh_9_14 * x + 0.173205080756887 * sh_9_16 * x - \
              0.173205080756888 * sh_9_2 * z + 0.8 * \
              sh_9_3 * y + 0.774596669241483 * sh_9_4 * z
    sh_10_5 = 0.724568837309472 * sh_9_13 * x + 0.223606797749979 * sh_9_15 * x - \
              0.223606797749979 * sh_9_3 * z + 0.866025403784438 * \
              sh_9_4 * y + 0.724568837309472 * sh_9_5 * z
    sh_10_6 = 0.674536878161602 * sh_9_12 * x + 0.273861278752583 * sh_9_14 * x - \
              0.273861278752583 * sh_9_4 * z + 0.916515138991168 * \
              sh_9_5 * y + 0.674536878161602 * sh_9_6 * z
    sh_10_7 = 0.62449979983984 * sh_9_11 * x + 0.324037034920393 * sh_9_13 * x - \
              0.324037034920393 * sh_9_5 * z + 0.953939201416946 * \
              sh_9_6 * y + 0.62449979983984 * sh_9_7 * z
    sh_10_8 = 0.574456264653803 * sh_9_10 * x + 0.374165738677394 * sh_9_12 * x - \
              0.374165738677394 * sh_9_6 * z + 0.979795897113272 * \
              sh_9_7 * y + 0.574456264653803 * sh_9_8 * z
    sh_10_9 = 0.424264068711928 * sh_9_11 * x - 0.424264068711929 * sh_9_7 * \
              z + 0.99498743710662 * sh_9_8 * y + 0.741619848709567 * sh_9_9 * x
    sh_10_10 = -0.670820393249937 * sh_9_10 * z - \
               0.670820393249937 * sh_9_8 * x + sh_9_9 * y
    sh_10_11 = 0.99498743710662 * sh_9_10 * y - 0.424264068711929 * sh_9_11 * \
               z - 0.424264068711929 * sh_9_7 * x + 0.741619848709567 * sh_9_9 * z
    sh_10_12 = 0.574456264653803 * sh_9_10 * z + 0.979795897113272 * sh_9_11 * y - \
               0.374165738677395 * sh_9_12 * z - 0.374165738677394 * \
               sh_9_6 * x - 0.574456264653803 * sh_9_8 * x
    sh_10_13 = 0.62449979983984 * sh_9_11 * z + 0.953939201416946 * sh_9_12 * y - \
               0.324037034920393 * sh_9_13 * z - 0.324037034920393 * \
               sh_9_5 * x - 0.62449979983984 * sh_9_7 * x
    sh_10_14 = 0.674536878161602 * sh_9_12 * z + 0.916515138991168 * sh_9_13 * y - \
               0.273861278752583 * sh_9_14 * z - 0.273861278752583 * \
               sh_9_4 * x - 0.674536878161603 * sh_9_6 * x
    sh_10_15 = 0.724568837309472 * sh_9_13 * z + 0.866025403784439 * sh_9_14 * y - \
               0.223606797749979 * sh_9_15 * z - 0.223606797749979 * \
               sh_9_3 * x - 0.724568837309472 * sh_9_5 * x
    sh_10_16 = 0.774596669241484 * sh_9_14 * z + 0.8 * sh_9_15 * y - 0.173205080756888 * \
               sh_9_16 * z - 0.173205080756887 * sh_9_2 * x - 0.774596669241484 * sh_9_4 * x
    sh_10_17 = -0.12247448713916 * sh_9_1 * x + 0.824621125123532 * sh_9_15 * z + \
               0.714142842854285 * sh_9_16 * y - 0.122474487139158 * \
               sh_9_17 * z - 0.824621125123533 * sh_9_3 * x
    sh_10_18 = -0.0707106781186548 * sh_9_0 * x + 0.874642784226796 * sh_9_16 * z + \
               0.6 * sh_9_17 * y - 0.0707106781186546 * \
               sh_9_18 * z - 0.874642784226796 * sh_9_2 * x
    sh_10_19 = -0.924662100445348 * sh_9_1 * x + 0.924662100445347 * \
               sh_9_17 * z + 0.435889894354068 * sh_9_18 * y
    sh_10_20 = -0.974679434480898 * sh_9_0 * x + 0.974679434480896 * sh_9_18 * z
    if lmax == 10:
        return ops.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12,
            sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12,
            sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12,
            sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10,
            sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20
        ], axis=-1)

    sh_11_0 = 0.977008420918394 * sh_10_0 * z + 0.977008420918394 * sh_10_20 * x
    sh_11_1 = 0.416597790450531 * sh_10_0 * y + 0.9315409787236 * \
              sh_10_1 * z + 0.931540978723599 * sh_10_19 * x
    sh_11_2 = -0.0642824346533223 * sh_10_0 * z + 0.574959574576069 * sh_10_1 * y + \
              0.88607221316445 * sh_10_18 * x + 0.886072213164452 * \
              sh_10_2 * z + 0.0642824346533226 * sh_10_20 * x
    sh_11_3 = -0.111340442853781 * sh_10_1 * z + 0.84060190949577 * sh_10_17 * x + \
              0.111340442853781 * sh_10_19 * x + 0.686348585024614 * \
              sh_10_2 * y + 0.840601909495769 * sh_10_3 * z
    sh_11_4 = 0.795129803842541 * sh_10_16 * x + 0.157459164324444 * sh_10_18 * x - \
              0.157459164324443 * sh_10_2 * z + 0.771389215839871 * \
              sh_10_3 * y + 0.795129803842541 * sh_10_4 * z
    sh_11_5 = 0.74965556829412 * sh_10_15 * x + 0.203278907045435 * sh_10_17 * x - \
              0.203278907045436 * sh_10_3 * z + 0.838140405208444 * \
              sh_10_4 * y + 0.74965556829412 * sh_10_5 * z
    sh_11_6 = 0.70417879021953 * sh_10_14 * x + 0.248964798865985 * sh_10_16 * x - \
              0.248964798865985 * sh_10_4 * z + 0.890723542830247 * \
              sh_10_5 * y + 0.704178790219531 * sh_10_6 * z
    sh_11_7 = 0.658698943008611 * sh_10_13 * x + 0.294579122654903 * sh_10_15 * x - \
              0.294579122654903 * sh_10_5 * z + 0.9315409787236 * \
              sh_10_6 * y + 0.658698943008611 * sh_10_7 * z
    sh_11_8 = 0.613215343783275 * sh_10_12 * x + 0.340150671524904 * sh_10_14 * x - \
              0.340150671524904 * sh_10_6 * z + 0.962091385841669 * \
              sh_10_7 * y + 0.613215343783274 * sh_10_8 * z
    sh_11_9 = 0.567727090763491 * sh_10_11 * x + 0.385694607919935 * sh_10_13 * x - \
              0.385694607919935 * sh_10_7 * z + 0.983332166035633 * \
              sh_10_8 * y + 0.56772709076349 * sh_10_9 * z
    sh_11_10 = 0.738548945875997 * sh_10_10 * x + 0.431219680932052 * sh_10_12 * \
               x - 0.431219680932052 * sh_10_8 * z + 0.995859195463938 * sh_10_9 * y
    sh_11_11 = sh_10_10 * y - 0.674199862463242 * \
               sh_10_11 * z - 0.674199862463243 * sh_10_9 * x
    sh_11_12 = 0.738548945875996 * sh_10_10 * z + 0.995859195463939 * sh_10_11 * \
               y - 0.431219680932052 * sh_10_12 * z - 0.431219680932053 * sh_10_8 * x
    sh_11_13 = 0.567727090763491 * sh_10_11 * z + 0.983332166035634 * sh_10_12 * y - \
               0.385694607919935 * sh_10_13 * z - 0.385694607919935 * \
               sh_10_7 * x - 0.567727090763491 * sh_10_9 * x
    sh_11_14 = 0.613215343783275 * sh_10_12 * z + 0.96209138584167 * sh_10_13 * y - \
               0.340150671524904 * sh_10_14 * z - 0.340150671524904 * \
               sh_10_6 * x - 0.613215343783274 * sh_10_8 * x
    sh_11_15 = 0.658698943008611 * sh_10_13 * z + 0.9315409787236 * sh_10_14 * y - \
               0.294579122654903 * sh_10_15 * z - 0.294579122654903 * \
               sh_10_5 * x - 0.65869894300861 * sh_10_7 * x
    sh_11_16 = 0.70417879021953 * sh_10_14 * z + 0.890723542830246 * sh_10_15 * y - \
               0.248964798865985 * sh_10_16 * z - 0.248964798865985 * \
               sh_10_4 * x - 0.70417879021953 * sh_10_6 * x
    sh_11_17 = 0.749655568294121 * sh_10_15 * z + 0.838140405208444 * sh_10_16 * y - \
               0.203278907045436 * sh_10_17 * z - 0.203278907045435 * \
               sh_10_3 * x - 0.749655568294119 * sh_10_5 * x
    sh_11_18 = 0.79512980384254 * sh_10_16 * z + 0.77138921583987 * sh_10_17 * y - \
               0.157459164324443 * sh_10_18 * z - 0.157459164324444 * \
               sh_10_2 * x - 0.795129803842541 * sh_10_4 * x
    sh_11_19 = -0.111340442853782 * sh_10_1 * x + 0.84060190949577 * sh_10_17 * z + \
               0.686348585024614 * sh_10_18 * y - 0.111340442853781 * \
               sh_10_19 * z - 0.840601909495769 * sh_10_3 * x
    sh_11_20 = -0.0642824346533226 * sh_10_0 * x + 0.886072213164451 * sh_10_18 * z + \
               0.57495957457607 * sh_10_19 * y - 0.886072213164451 * \
               sh_10_2 * x - 0.0642824346533228 * sh_10_20 * z
    sh_11_21 = -0.9315409787236 * sh_10_1 * x + 0.931540978723599 * \
               sh_10_19 * z + 0.416597790450531 * sh_10_20 * y
    sh_11_22 = -0.977008420918393 * sh_10_0 * x + 0.977008420918393 * sh_10_20 * z
    return ops.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12,
        sh_7_13, sh_7_14,
        sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12,
        sh_8_13, sh_8_14, sh_8_15, sh_8_16,
        sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12,
        sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
        sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11,
        sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
        sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11,
        sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22
    ], axis=-1)
