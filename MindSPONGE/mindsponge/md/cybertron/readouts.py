# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
#
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""readouts"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from .units import units
from .neighbors import GatherNeighbors
from .base import SmoothReciprocal
from .cutoff import get_cutoff
from .aggregators import get_aggregator, get_list_aggregator
from .decoders import get_decoder

__all__ = [
    "Readout",
    "AtomwiseReadout",
    "GraphReadout",
    "LongeRangeReadout",
    "PairwiseReadout",
    "CoulombReadout",
    "InteractionsAggregator",
]


class InteractionsAggregator(nn.Cell):
    """InteractionsAggregator"""
    def __init__(self,
                 n_in,
                 n_out,
                 n_interactions,
                 activation=None,
                 list_aggregator='sum',
                 n_aggregator_hiddens=0,
                 decoders=None,
                 ):
        super().__init__()

        self.n_interactions = n_interactions
        self.decoders = None

        if decoders is not None:
            if isinstance(decoders, (tuple, list)):
                self.decoders = nn.CellList([
                    get_decoder(decoders[i], n_in, n_out, activation)
                    for i in range(n_interactions)
                ])
            elif isinstance(decoders, str):
                self.decoders = nn.CellList([
                    get_decoder(decoders, n_in, n_out, activation)
                    for i in range(n_interactions)
                ])
            else:
                raise TypeError(
                    "Unsupported Decoder type '{}'.".format(
                        type(decoders)))

        self.list_aggregator = get_list_aggregator(
            list_aggregator, n_out, n_interactions, n_aggregator_hiddens, activation)
        if self.list_aggregator is None:
            raise TypeError(
                "ListAggregator cannot be None at InteractionsAggregator")

    def construct(self, xlist, atom_mask=None):
        if self.decoders is not None:
            ylist = []
            n_interactions = len(xlist)
            for i in range(n_interactions):
                y = self.decoders[i](xlist[i])
                ylist.append(y)
            return self.list_aggregator(ylist, atom_mask)
        return self.list_aggregator(xlist, atom_mask)


class Readout(nn.Cell):
    """Readout"""
    def __init__(self,
                 n_in,
                 n_out=1,
                 atom_scale=1,
                 atom_shift=0,
                 mol_scale=1,
                 mol_shift=0,
                 axis=-2,
                 atom_ref=None,
                 scaled_by_atoms_number=True,
                 averaged_by_atoms_number=False,
                 activation=None,
                 decoder=None,
                 aggregator=None,
                 unit_energy='kJ/mol',
                 multi_aggregators=False,
                 read_all_interactions=False,
                 n_interactions=None,
                 interactions_aggregator='sum',
                 n_aggregator_hiddens=0,
                 interaction_decoders=None,
                 ):
        super().__init__()

        self.name = 'Readout'

        if unit_energy is None:
            self.output_is_energy = False
        else:
            self.output_is_energy = True
            if not isinstance(unit_energy, str):
                raise TypeError('Type of unit_energy must be str')
            unit_energy = units.check_energy_unit(unit_energy)
            units.set_energy_unit(unit_energy)
        self.unit_energy = unit_energy

        self.averaged_by_atoms_number = averaged_by_atoms_number

        self.atom_ref = atom_ref

        if not isinstance(n_in, int):
            raise TypeError('Type of n_in must be int')
        self.n_in = n_in

        n_out, num_output_dim = self._check_type_and_number(
            n_out, 'n_out', int)
        self.n_out = n_out
        self.multi_n_out = n_out
        self.total_out = n_out

        self.multi_output_number = False
        if num_output_dim > 1:
            n_out = list(set(n_out))
            if len(n_out) > 1:
                self.multi_output_number = True
            else:
                n_out = n_out[0]

        self.n_out = n_out

        atom_scale, num_atom_scale = self._check_type_and_number(
            atom_scale, 'atom_scale', (int, float, Tensor))
        self.atom_scale = atom_scale

        atom_shift, num_atom_shift = self._check_type_and_number(
            atom_shift, 'atom_shift', (int, float, Tensor))
        self.atom_shift = atom_shift

        if not isinstance(mol_scale, (float, int, Tensor)):
            raise TypeError('Type of mol_scale must be float, int or Tensor')
        self.mol_scale = mol_scale

        if not isinstance(mol_shift, (float, int, Tensor)):
            raise TypeError('Type of mol_shift must be float, int or Tensor')
        self.mol_shift = mol_shift

        if not isinstance(axis, int):
            raise TypeError('Type of mol_shift must be int')
        self.axis = axis

        activation, num_activation = self._check_type_and_number(
            activation, 'activation')
        self.activation = activation

        if not isinstance(scaled_by_atoms_number, bool):
            raise TypeError('Type of scaled_by_atoms_number must be bool')
        self.scaled_by_atoms_number = scaled_by_atoms_number

        if not isinstance(averaged_by_atoms_number, bool):
            raise TypeError('Type of averaged_by_atoms_number must be bool')

        if read_all_interactions and interaction_decoders is not None:
            decoder = None

        self.decoder = None
        self.multi_decoders = False
        self.num_decoder = 1
        if isinstance(decoder, (tuple, list)):
            self.multi_decoders = True
            self.num_decoder = len(decoder)
            if self.num_decoder == 1:
                decoder = decoder[0]
            elif self.num_decoder == 0:
                raise ValueError('Number of decoder cannot be zero')

        if num_output_dim > 1:
            self.multi_decoders = True
            if self.num_decoder != num_output_dim:
                if self.num_decoder == 1:
                    self.decoder = [decoder,] * num_output_dim
                    self.num_decoder = num_output_dim
                else:
                    raise ValueError('Number of decoder mismatch')

        self.aggregator = None
        self.multi_aggregators = multi_aggregators
        self.num_aggregator = 1
        if isinstance(aggregator, (tuple, list)):
            self.multi_aggregators = True
            self.num_aggregator = len(aggregator)
            if self.num_aggregator == 1:
                aggregator = aggregator[0]
            if self.num_aggregator == 0:
                raise ValueError('Number of aggregator cannot be zero')

        if self.num_aggregator == 1 and self.multi_aggregators:
            aggregator = [aggregator,] * self.num_decoder
            self.num_aggregator = self.num_decoder
        self.aggregator = aggregator

        if self.multi_decoders:
            if self.num_aggregator != self.num_decoder and self.num_aggregator != 1:
                raise ValueError('Number of aggregator mismatch')
        else:
            if self.multi_aggregators:
                raise ValueError(
                    'multi aggregators must be used with multi decoders')

        self.split_slice = ()
        if self.multi_decoders:
            if num_output_dim != self.num_decoder:
                if num_output_dim == 1:
                    self.multi_n_out = [n_out,] * self.num_decoder
                else:
                    raise ValueError('Number of n_out mismatch')

            sect = 0
            for i in range(len(self.multi_n_out) - 1):
                sect = self.multi_n_out[i] + sect
                self.split_slice += (sect,)

            self.total_out = 0
            for n in self.multi_n_out:
                self.total_out += n

            if num_activation != self.num_decoder:
                if num_activation == 1:
                    self.activation = [activation,] * self.num_decoder
                else:
                    raise ValueError('Number of activation missmatch')
        else:
            if num_atom_scale > 1:
                raise ValueError('Number of atom_scale mismatch')
            if num_atom_shift > 1:
                raise ValueError('Number of atom_shift mismatch')
            if num_activation > 1:
                raise ValueError('Number of activation mismatch')

        self.str_unit_energy = (
            " " + unit_energy) if self.output_is_energy else ""

        self.split = P.Split(-1, self.num_decoder)

        self.multi_atom_scale = None
        self.multi_atom_shift = None
        self.multi_atom_ref = None
        if self.multi_decoders:
            self.multi_atom_scale = self._split_by_decoders(
                self.atom_scale, 'atom_scale')
            self.multi_atom_shift = self._split_by_decoders(
                self.atom_shift, 'atom_scale')
            if atom_ref is not None:
                self.multi_atom_ref = self._split_by_decoders(
                    self.atom_ref, 'atom_ref')

        self.read_all_interactions = read_all_interactions
        self.n_interactions = n_interactions

        if read_all_interactions:
            if interaction_decoders is not None:
                if decoder is not None:
                    raise ValueError(
                        'decoder and interaction_decoders cannot be used at same time')
                if self.multi_decoders:
                    raise ValueError(
                        'Multiple decoders cannot support interaction_decoders')
                if n_interactions is None or n_interactions == 0:
                    raise ValueError(
                        'n_interactions must be setup when usingn interaction_decoders')

            self.interaction_decoders = interaction_decoders
            self.interactions_aggregator = InteractionsAggregator(
                n_in,
                n_out,
                n_interactions,
                activation=activation,
                list_aggregator=interactions_aggregator,
                n_aggregator_hiddens=n_aggregator_hiddens,
                decoders=interaction_decoders,
            )
        else:
            self.interactions_aggregator = None
            self.interaction_decoders = None

        self.concat = P.Concat(-1)

    def print_info(self):
        """print info"""

        self._print_plus_info()

        if self.read_all_interactions:
            print("------read all interactions with interactions aggregator: " +
                  str(self.interactions_aggregator.list_aggregator))
            if self.interaction_decoders:
                print("---------with independent decoders: ")
                for i in range(self.n_interactions):
                    decoder = self.interactions_aggregator.decoders[i]
                    print("---------" +
                          str(i +
                              1) +
                          '. decoder "' +
                          str(decoder) +
                          '" with activation "' +
                          str(decoder.activation) +
                          '".')
            else:
                print("---------without independent decoder. ")
        else:
            print("------read last interactions:")

        if self.multi_decoders:
            print("------with " +
                  str(self.multi_decoders) +
                  " multiple decoders and " +
                  ("aggregators." if self.multi_aggregators else ('common "' +
                                                                  str(self.aggregator) +
                                                                  '" aggreator.')))
            for i in range(self.num_decoder):
                print("------" +
                      str(i +
                          1) +
                      '. decoder "' +
                      str(self.decoder[i]) +
                      '" with activation "' +
                      str(self.decoder[i].activation) +
                      '".')
                if self.multi_aggregators:
                    print(
                        "---------with aggregator: " +
                        self.aggregator[i].name)
                print("---------with readout dimension: " +
                      str(self.multi_n_out[i]))
                print("---------with activation function: " +
                      str(self.activation[i]))

            print("------with multiple scale and shift:")
            for i in range(self.num_decoder):
                print("------" + str(i + 1) +
                      ". output with dimension: " + str(self.multi_n_out[i]))
                print("---------with atom scale: " +
                      str(self.multi_atom_scale[i]) + self.str_unit_energy)
                print("---------with atom shift: " +
                      str(self.multi_atom_shift[i]) + self.str_unit_energy)
            print("------scaled by atoms number: " +
                  str(self.scaled_by_atoms_number))

        else:
            if self.decoder is not None:
                print("------with decoder: " + str(self.decoder))
                print("------with activation function: " +
                      str(self.decoder.activation))
            print("------with readout dimension: " + str(self.n_out))
            print("------with atom scale: " +
                  str(self.atom_scale) + self.str_unit_energy)
            print("------with atom shift: " +
                  str(self.atom_shift) + self.str_unit_energy)
            print("------scaled by atoms number: " +
                  str(self.scaled_by_atoms_number))

        print("------with total readout dimension: " + str(self.total_out))
        print("------with molecular scale: " +
              str(self.mol_scale) + self.str_unit_energy)
        print("------with molecular shift: " +
              str(self.mol_shift) + self.str_unit_energy)
        print("------averaged by atoms number: " +
              ('Yes' if self.averaged_by_atoms_number else 'No'))

    def _print_plus_info(self):
        print('------calculate long range interaction: No')

    def _check_type_and_number(self, inputs, name, types=None):
        """_check_type_and_number"""
        num = 1
        if isinstance(inputs, (tuple, list)):
            num = len(inputs)
            if num == 0:
                raise ValueError("Size of " + name + " cannot be Zeros!")
            if num == 1:
                inputs = inputs[0]
                if (types is not None) and (not isinstance(inputs, types)):
                    raise TypeError(
                        "Unsupported " +
                        name +
                        " type '{}'.".format(
                            type(inputs)))
        elif (types is not None) and (not isinstance(inputs, types)):
            raise TypeError(
                "Unsupported " +
                name +
                " type '{}'.".format(
                    type(inputs)))

        return inputs, num

    def _split_by_decoders(self, inputs, name):
        """_split_by_decoders"""
        if self.multi_decoders:
            if isinstance(inputs, (tuple, list)):
                if len(inputs) != self.num_decoder:
                    if len(inputs) == 1:
                        inputs = inputs * self.num_decoder
                    else:
                        raise ValueError('Number of ' + name + ' mismatch')
            elif isinstance(inputs, (float, int)):
                if isinstance(input, float):
                    inputs = Tensor(inputs, ms.float32)
                if isinstance(input, int):
                    inputs = Tensor(inputs, ms.int32)
                inputs = [inputs,] * self.num_decoder
            elif isinstance(inputs, Tensor):
                if inputs.shape[-1] != self.total_out:
                    raise ValueError('Last dimension of ' + name + ' mismatch')

                if self.multi_output_number:
                    inputs = msnp.split(inputs, self.split_slice, -1)
                else:
                    inputs = self.split(inputs)
            else:
                raise TypeError(
                    "Unsupported Decoder type '{}'.".format(
                        type(inputs)))

        return inputs


class AtomwiseReadout(Readout):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
            self,
            n_in,
            n_out=1,
            atom_scale=1,
            atom_shift=0,
            mol_scale=1,
            mol_shift=0,
            axis=-2,
            atom_ref=None,
            scaled_by_atoms_number=False,
            averaged_by_atoms_number=False,
            activation=None,
            decoder='halve',
            aggregator='sum',
            unit_energy='kJ/mol',
            multi_aggregators=False,
            read_all_interactions=False,
            n_interactions=None,
            interactions_aggregator='sum',
            n_aggregator_hiddens=0,
            interaction_decoders=None,
        ):
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            atom_scale=atom_scale,
            atom_shift=atom_shift,
            mol_scale=mol_scale,
            mol_shift=mol_shift,
            axis=axis,
            atom_ref=atom_ref,
            scaled_by_atoms_number=scaled_by_atoms_number,
            averaged_by_atoms_number=averaged_by_atoms_number,
            activation=activation,
            decoder=decoder,
            aggregator=aggregator,
            unit_energy=unit_energy,
            multi_aggregators=multi_aggregators,
            read_all_interactions=read_all_interactions,
            n_interactions=n_interactions,
            interactions_aggregator=interactions_aggregator,
            n_aggregator_hiddens=n_aggregator_hiddens,
            interaction_decoders=interaction_decoders,
        )
        self.name = 'Atomwise'

        if self.multi_decoders:
            decoders = []
            for i in range(self.num_decoder):
                decoder_t = get_decoder(
                    self.decoder[i],
                    self.n_in,
                    self.multi_n_out[i],
                    self.activation[i])
                if decoder_t is not None:
                    decoders.append(decoder_t)
                else:
                    raise ValueError('Multi decoders cannot include None type')
            self.decoder = nn.CellList(decoders)
        else:
            self.decoder = get_decoder(
                decoder, self.n_in, self.n_out, self.activation)

        if self.decoder is None and self.interaction_decoders is None and self.n_in != self.n_out:
            raise ValueError(
                "When decoder is None, n_out (" +
                str(n_out) +
                ") must be equal to n_in (" +
                str(n_in) +
                ")")

        if self.multi_aggregators:
            aggregators = []
            for i in range(self.num_aggregator):
                aggregator_t = get_aggregator(
                    self.aggregator[i], self.multi_n_out[i], axis)
                if aggregator_t is not None:
                    aggregators.append(aggregator_t)
                else:
                    raise ValueError(
                        'Multi aggregators cannot include None type')
            self.aggregator = nn.CellList(aggregators)

        else:
            if self.multi_output_number:
                agg_dict = {}
                for n_out_t in self.n_out:
                    aggregator_t = get_aggregator(self.aggregator, n_out_t, axis)
                    if aggregator_t is not None:
                        agg_dict[n_out_t] = aggregator_t

                aggregators = []
                for i in range(self.num_decoder):
                    aggregators.append(agg_dict[self.multi_n_out[i]])
                self.aggregator = nn.CellList(aggregators)
                self.multi_aggregators = True
            else:
                self.aggregator = get_aggregator(aggregator, self.n_out, axis)

    def construct(
            self,
            x,
            xlist,
            atoms_types=None,
            atom_mask=None,
            atoms_number=None):
        r"""
        predicts atomwise property
        """

        if self.read_all_interactions:
            x = self.interactions_aggregator(xlist, atom_mask)

        y = None
        if self.multi_decoders:
            ytuple = ()
            for i in range(self.num_decoder):
                yi = self.decoder[i](x)
                if self.multi_aggregators:
                    yi = yi * \
                        self.multi_atom_scale[i] + self.multi_atom_shift[i]
                    if self.scaled_by_atoms_number:
                        yi = yi / atoms_number
                    if self.atom_ref is not None:
                        yi += F.gather(self.multi_atom_ref[i], atoms_types, 0)
                    yi = self.aggregator[i](yi, atom_mask, atoms_number)

                ytuple = ytuple + (yi,)

            y = self.concat(ytuple)

            if not self.multi_aggregators:
                y = y * self.atom_scale + self.atom_shift
                if self.scaled_by_atoms_number:
                    y = y / atoms_number
                if self.atom_ref is not None:
                    y += F.gather(self.atom_ref, atoms_types, 0)
                y = self.aggregator(y, atom_mask, atoms_number)
        else:
            if self.decoder is not None:
                y = self.decoder(x)
            else:
                y = x

            y = y * self.atom_scale + self.atom_shift
            if self.scaled_by_atoms_number:
                y = y / atoms_number

            if self.atom_ref is not None:
                y += F.gather(self.atom_ref, atoms_types, 0)

            if self.aggregator is not None:
                y = self.aggregator(y, atom_mask)

        y = y * self.mol_scale + self.mol_shift

        if self.averaged_by_atoms_number:
            if atoms_number is None:
                atoms_number = x.shape[self.axis]
            y = y / atoms_number

        return y


class GraphReadout(Readout):
    """

    Args:

    Returns:

    """

    def __init__(
            self,
            n_in,
            n_out=1,
            atom_scale=1,
            atom_shift=0,
            mol_scale=1,
            mol_shift=0,
            axis=-2,
            atom_ref=None,
            scaled_by_atoms_number=False,
            averaged_by_atoms_number=False,
            activation=None,
            decoder='halve',
            aggregator='mean',
            unit_energy=None,
            multi_aggregators=False,
            read_all_interactions=False,
            n_interactions=None,
            interactions_aggregator='sum',
            n_aggregator_hiddens=0,
            interaction_decoders=None,
        ):
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            atom_scale=atom_scale,
            atom_shift=atom_shift,
            mol_scale=mol_scale,
            mol_shift=mol_shift,
            axis=axis,
            atom_ref=atom_ref,
            scaled_by_atoms_number=scaled_by_atoms_number,
            averaged_by_atoms_number=averaged_by_atoms_number,
            activation=activation,
            decoder=decoder,
            aggregator=aggregator,
            unit_energy=unit_energy,
            multi_aggregators=multi_aggregators,
            read_all_interactions=read_all_interactions,
            n_interactions=n_interactions,
            interactions_aggregator=interactions_aggregator,
            n_aggregator_hiddens=n_aggregator_hiddens,
            interaction_decoders=interaction_decoders,
        )

        self.name = 'Graph'

        if self.interaction_decoders is not None:
            raise ValueError('GraphReadout cannot use interaction_decoders')

        if self.multi_aggregators:
            aggregators = []
            for i in range(self.num_aggregator):
                aggregator_t = get_aggregator(
                    self.aggregator[i], self.n_in, axis)
                if aggregator_t is not None:
                    aggregators.append(aggregator_t)
                else:
                    raise ValueError(
                        'Multi aggregators cannot include None type')
            self.aggregator = nn.CellList(aggregators)
        else:
            self.aggregator = get_aggregator(aggregator, self.n_in, axis)
            if self.aggregator is None:
                raise ValueError("aggregator cannot be None at GraphReadout")

        if self.multi_decoders:
            decoders = []
            for i in range(self.num_decoder):
                decoder_t = get_decoder(
                    self.decoder[i],
                    self.n_in,
                    self.multi_n_out[i],
                    self.activation[i])
                if decoder_t is not None:
                    decoders.append(decoder_t)
                else:
                    raise ValueError('Multi decoders cannot include None type')
            self.decoder = nn.CellList(decoders)
        else:
            self.decoder = get_decoder(
                decoder, self.n_in, self.n_out, self.activation)
            if self.decoder is None and n_in != n_out:
                raise ValueError(
                    "When decoder is None, n_out (" +
                    str(n_out) +
                    ") must be equal to n_in (" +
                    str(n_in) +
                    ")")

        self.reduce_sum = P.ReduceSum()

    def construct(
            self,
            x,
            xlist,
            atoms_types=None,
            atom_mask=None,
            atoms_number=None):
        r"""
        predicts graph property
        """

        if self.read_all_interactions:
            x = self.interactions_aggregator(xlist, atom_mask)

        y = None
        if self.multi_decoders:
            if self.multi_aggregators:
                agg = None
            else:
                agg = self.aggregator(x, atom_mask, atoms_number)

            ytuple = ()
            for i in range(self.num_decoder):
                if self.multi_aggregators:
                    agg = self.aggregator[i](x, atom_mask, atoms_number)

                yi = self.decoder[i](agg)

                ytuple = ytuple + (yi,)

            y = self.concat(ytuple)

            y = y * self.atom_scale + self.atom_shift
            if self.scaled_by_atoms_number:
                y = y / atoms_number

        else:
            agg = self.aggregator(x, atom_mask, atoms_number)

            if self.decoder is not None:
                y = self.decoder(agg)
            else:
                y = agg

            y = y * self.atom_scale + self.atom_shift
            if self.scaled_by_atoms_number:
                y = y / atoms_number

        if self.atom_ref is not None:
            ref = F.gather(self.atom_ref, atoms_types, 0)
            ref = self.reduce_sum(ref, self.axis)
            y += ref

        y = y * self.mol_scale + self.mol_shift

        if self.averaged_by_atoms_number:
            if atoms_number is None:
                atoms_number = x.shape[self.axis]
            y = y / atoms_number

        return y


class LongeRangeReadout(Readout):
    """LongeRangeReadout"""
    def __init__(self,
                 dim_feature,
                 atom_scale=1,
                 atom_shift=0,
                 mol_scale=1,
                 mol_shift=0,
                 axis=-2,
                 atom_ref=None,
                 activation=None,
                 decoder='halve',
                 longrange_decoder=None,
                 unit_energy='kcal/mol',
                 cutoff_function='gaussian',
                 cutoff_max=units.length(1, 'nm'),
                 cutoff_min=units.length(0.8, 'nm'),
                 fixed_neigh=False,
                 read_all_interactions=False,
                 n_interactions=None,
                 interactions_aggregator='sum',
                 n_aggregator_hiddens=0,
                 interaction_decoders=None,
                 ):
        super().__init__(
            n_in=dim_feature,
            n_out=1,
            atom_scale=atom_scale,
            atom_shift=atom_shift,
            mol_scale=mol_scale,
            mol_shift=mol_shift,
            axis=axis,
            atom_ref=atom_ref,
            scaled_by_atoms_number=False,
            averaged_by_atoms_number=False,
            activation=activation,
            decoder=decoder,
            aggregator='sum',
            unit_energy=unit_energy,
            multi_aggregators=False,
            read_all_interactions=read_all_interactions,
            n_interactions=n_interactions,
            interactions_aggregator=interactions_aggregator,
            n_aggregator_hiddens=n_aggregator_hiddens,
            interaction_decoders=interaction_decoders,
        )

        self.name = 'longrange'
        self.longrange_decoder = longrange_decoder
        self.coulomb_const = units.Coulomb()

        if self.multi_decoders:
            raise ValueError('LongRangeReadout cannot use multiple decoders')

        self.aggregator = get_aggregator('sum', 1, axis)

        self.gather_neighbors = GatherNeighbors(dim_feature, fixed_neigh)
        self.squeeze = P.Squeeze(-1)
        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(keep_dims=True)

        self.smooth_reciprocal = SmoothReciprocal()

        if cutoff_function is not None:
            self.cutoff_function = get_cutoff(
                cutoff_function,
                r_max=cutoff_max,
                r_min=cutoff_min,
                return_mask=False,
                reverse=True)
        else:
            self.cutoff_function = None

    def set_fixed_neighbors(self, flag=True):
        self.fixed_neigh = flag
        self.gather_neighbors.fixed_neigh = flag

    def _print_plus_info(self):
        print('------calculate long range interaction: Yes')
        print('---------with method for long range interaction: ' + self.name)
        print('---------with coulomb constant: ' + str(self.coulomb_const * 2) + ' ' + self.unit_energy + '*' +
              self.unit_dis)


class CoulombReadout(LongeRangeReadout):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
            self,
            dim_feature,
            atom_scale=1,
            atom_shift=0,
            mol_scale=1,
            mol_shift=0,
            axis=-2,
            atom_ref=None,
            activation=None,
            decoder='halve',
            longrange_decoder=None,
            unit_energy='kcal/mol',
            cutoff_function='gaussian',
            cutoff_max=units.length(1, 'nm'),
            cutoff_min=units.length(0.8, 'nm'),
            fixed_neigh=False,
            read_all_interactions=False,
            n_interactions=None,
            interactions_aggregator='sum',
            n_aggregator_hiddens=0,
            interaction_decoders=None,
        ):
        super().__init__(
            dim_feature=dim_feature,
            atom_scale=atom_scale,
            atom_shift=atom_shift,
            mol_scale=mol_scale,
            mol_shift=mol_shift,
            axis=axis,
            atom_ref=atom_ref,
            activation=activation,
            decoder=decoder,
            longrange_decoder=longrange_decoder,
            unit_energy=unit_energy,
            cutoff_function=cutoff_function,
            cutoff_max=cutoff_max,
            cutoff_min=cutoff_min,
            fixed_neigh=fixed_neigh,
            read_all_interactions=read_all_interactions,
            n_interactions=n_interactions,
            interactions_aggregator=interactions_aggregator,
            n_aggregator_hiddens=n_aggregator_hiddens,
            interaction_decoders=interaction_decoders,
        )
        self.name = 'Coulumb'

        self.decoder = get_decoder(decoder, dim_feature, 2, self.activation)
        if self.decoder is None:
            raise ValueError("Decoder in CoulombReadout cannot be None")

        if longrange_decoder is not None:
            raise ValueError("CoulombReadout cannot support longrange_decoder")

        self.split = P.Split(-1, 2)

    def construct(
            self,
            x,
            xlist,
            atoms_types=None,
            atom_mask=None,
            atoms_number=None,
            distances=None,
            neighbors=None,
            neighbor_mask=None):
        r"""
        predicts atomwise property
        """

        if self.read_all_interactions:
            x = self.interactions_aggregator(xlist, atom_mask)

        y = self.decoder(x)

        # [B,A,2] -> [B,A,1] * 2
        (ei, qi) = self.split(y)
        # [B,A,1] -> [B,A,N,1]
        qij = self.gather_neighbors(qi, neighbors)
        # [B,A,N,1] -> [B,A,N]
        qij = self.squeeze(qij)
        # [B,A,N] -> [B,A,N] * [B,A,1]
        qiqj = qij * qi

        sij = self.smooth_reciprocal(distances, neighbor_mask)
        if self.cutoff_function is not None:
            sij = sij * self.cutoff_function(distances, neighbor_mask)

        eq = qiqj * sij
        eq = self.reduce_sum(eq, -1)
        eq = self.aggregator(eq, atom_mask, atoms_number) * \
            self.coulomb_const / 2.

        ei = ei * self.atom_scale + self.atom_shift

        if self.atom_ref is not None:
            ei += F.gather(self.atom_ref, atoms_types, 0)

        ei = self.aggregator(ei, atom_mask, atoms_number)

        ei = ei * self.mol_scale + self.mol_shift

        return ei + eq


class PairwiseReadout(LongeRangeReadout):
    """PairwiseReadout"""
    def __init__(self,
                 dim_feature,
                 atom_scale=1,
                 atom_shift=0,
                 mol_scale=1,
                 mol_shift=0,
                 axis=-2,
                 atom_ref=None,
                 activation=None,
                 decoder='halve',
                 longrange_decoder=None,
                 unit_energy='kcal/mol',
                 cutoff_function='gaussian',
                 cutoff_max=units.length(1, 'nm'),
                 cutoff_min=units.length(0.8, 'nm'),
                 fixed_neigh=False,
                 read_all_interactions=False,
                 n_interactions=None,
                 interactions_aggregator='sum',
                 n_aggregator_hiddens=0,
                 interaction_decoders=None,
                 ):
        super().__init__(
            dim_feature=dim_feature,
            atom_scale=atom_scale,
            atom_shift=atom_shift,
            mol_scale=mol_scale,
            mol_shift=mol_shift,
            axis=axis,
            atom_ref=atom_ref,
            activation=activation,
            decoder=decoder,
            longrange_decoder=longrange_decoder,
            unit_energy=unit_energy,
            cutoff_function=cutoff_function,
            cutoff_max=cutoff_max,
            cutoff_min=cutoff_min,
            fixed_neigh=fixed_neigh,
            read_all_interactions=read_all_interactions,
            n_interactions=n_interactions,
            interactions_aggregator=interactions_aggregator,
            n_aggregator_hiddens=n_aggregator_hiddens,
            interaction_decoders=interaction_decoders,
        )

        self.name = 'pairwise'

        self.decoder = get_decoder(decoder, dim_feature, 1, self.activation)
        if self.decoder is None:
            raise ValueError("Decoder in CoulombReadout cannot be None")

        self.longrange_decoder = get_decoder(
            decoder, dim_feature, 1, self.activation)
        if self.longrange_decoder is None:
            raise ValueError(
                "longrange_decoder in CoulombReadout cannot be None")

        self.squeeze = P.Squeeze(-1)

    def construct(
            self,
            x,
            xlist,
            atoms_types=None,
            atom_mask=None,
            atoms_number=None,
            distances=None,
            neighbors=None,
            neighbor_mask=None):
        """construct"""

        if self.read_all_interactions:
            x = self.interactions_aggregator(xlist, atom_mask)

        ei = self.decoder(x)

        # [B,A,V] -> [B,A,1,V]
        xi = F.expand_dims(x, -2)
        # [B,A,N,V]
        xij = self.gather_neighbors(x, neighbors)
        # [B,A,N,V] = [B,A,N,V] * [B,A,1,V]
        xixj = xij * xi

        # [B,A,N,1]
        qiqj = self.longrange_decoder(xixj)

        qiqj = self.squeeze(qiqj)
        sij = self.smooth_reciprocal(distances, neighbor_mask)

        if self.cutoff_function is not None:
            cij = self.cutoff_function(distances, neighbor_mask)
            sij = sij * cij

        eq = qiqj * sij
        eq = self.reduce_sum(eq, -1)
        eq = self.aggregator(eq, atom_mask, atoms_number) * \
            self.coulomb_const / 2.

        ei = ei * self.atom_scale + self.atom_shift

        if self.atom_ref is not None:
            ei += F.gather(self.atom_ref, atoms_types, 0)

        ei = self.aggregator(ei, atom_mask, atoms_number)

        ei = ei * self.mol_scale + self.mol_shift

        return ei + eq
