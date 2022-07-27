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
"""structure module"""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindsponge.cell.initializer import lecun_init


class PredictedLDDTHead(nn.Cell):
    """Head to predict the per-residue LDDT to be used as a confidence measure."""

    def __init__(self, config, seq_channel):
        super().__init__()
        self.config = config
        self.input_layer_norm = nn.LayerNorm([seq_channel,], epsilon=1e-5)
        self.act_0 = nn.Dense(seq_channel, self.config.num_channels,
                              weight_init=lecun_init(seq_channel, initializer_name='relu')
                              ).to_float(mstype.float16)
        self.act_1 = nn.Dense(self.config.num_channels, self.config.num_channels,
                              weight_init=lecun_init(self.config.num_channels, initializer_name='relu')
                              ).to_float(mstype.float16)
        self.logits = nn.Dense(self.config.num_channels, self.config.num_bins, weight_init='zeros'
                               ).to_float(mstype.float16)
        self.relu = nn.ReLU()

    def construct(self, rp_structure_module):
        """Builds ExperimentallyResolvedHead module."""
        act = rp_structure_module
        act = self.input_layer_norm(act.astype(mstype.float32))
        act = self.act_0(act)
        act = self.relu(act.astype(mstype.float32))
        act = self.act_1(act)
        act = self.relu(act.astype(mstype.float32))
        logits = self.logits(act)
        return logits


class DistogramHead(nn.Cell):
    """Head to predict a distogram.

    Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
    """

    def __init__(self, config, pair_dim):
        super().__init__()
        self.config = config
        self.half_logits = nn.Dense(pair_dim, self.config.num_bins, weight_init='zeros')
        self.first_break = self.config.first_break
        self.last_break = self.config.last_break
        self.num_bins = self.config.num_bins

    def construct(self, pair):
        """Builds DistogramHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'pair': pair representation, shape [N_res, N_res, c_z].

        Returns:
          Dictionary containing:
            * logits: logits for distogram, shape [N_res, N_res, N_bins].
            * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
        """
        half_logits = self.half_logits(pair)

        logits = half_logits + mnp.swapaxes(half_logits, -2, -3)
        breaks = mnp.linspace(self.first_break, self.last_break, self.num_bins - 1)

        return logits, breaks


class ExperimentallyResolvedHead(nn.Cell):
    """Predicts if an atom is experimentally resolved in a high-res structure.

    Only trained on high-resolution X-ray crystals & cryo-EM.
    Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'
    """

    def __init__(self, seq_channel):
        super().__init__()
        self.logits = nn.Dense(seq_channel, 37, weight_init='zeros')

    def construct(self, single):
        """Builds ExperimentallyResolvedHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'single': Single representation, shape [N_res, c_s].

        Returns:
          Dictionary containing:
            * 'logits': logits of shape [N_res, 37],
                log probability that an atom is resolved in atom37 representation,
                can be converted to probability by applying sigmoid.
        """
        logits = self.logits(single)
        return logits


class MaskedMsaHead(nn.Cell):
    """Head to predict MSA at the masked locations.

    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """

    def __init__(self, config, msa_channel):
        super().__init__()
        self.config = config
        self.logits = nn.Dense(msa_channel, self.config.num_output, weight_init='zeros')

    def construct(self, msa):
        """Builds MaskedMsaHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'msa': MSA representation, shape [N_seq, N_res, c_m].

        Returns:
          Dictionary containing:
            * 'logits': logits of shape [N_seq, N_res, N_aatype] with
                (unnormalized) log probabilies of predicted aatype at position.
        """
        # del batch
        logits = self.logits(msa)
        return logits


class PredictedAlignedErrorHead(nn.Cell):
    """Head to predict the distance errors in the backbone alignment frames.

    Can be used to compute predicted TM-Score.
    Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
    """

    def __init__(self, config, pair_dim):
        super().__init__()
        self.config = config
        self.num_bins = self.config.num_bins
        self.max_error_bin = self.config.max_error_bin
        self.logits = nn.Dense(pair_dim, self.num_bins, weight_init='zeros')

    def construct(self, pair):
        """Builds PredictedAlignedErrorHead module.

        Arguments:
            * 'pair': pair representation, shape [N_res, N_res, c_z].

        Returns:
            * logits: logits for aligned error, shape [N_res, N_res, N_bins].
            * breaks: array containing bin breaks, shape [N_bins - 1].
        """
        logits = self.logits(pair)
        breaks = mnp.linspace(0, self.max_error_bin, self.num_bins - 1)
        return logits, breaks
