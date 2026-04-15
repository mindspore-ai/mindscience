#    Copyright 2025 Yuanhanyu Luo & Linchang Zhu

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
This module defines the MedFormer model for gene expression prediction.
"""

import mindspore as ms
from mindspore import nn
from mindspore import Parameter
from mindspore import ops

class GenePertFormer(ms.nn.Cell):
    """
    GenePertFormer model for gene expression prediction, combining gene, drug, and cell features.
    """
    def __init__(self, gene_vocab_size=23185, drug_dim=1024, cell_dim=82,
                 hidden_dim=256, n_layers=4, n_heads=1, dropout=0.1,
                 cell_input_dim=978, use_cell_expr=False):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embeddings
        self.gene_embedding = nn.Embedding(gene_vocab_size, hidden_dim)
        self.expr_embedding = nn.Dense(1, hidden_dim)
        self.drug_embedding = nn.Dense(drug_dim, hidden_dim)

        self.use_cell_expr = use_cell_expr
        if use_cell_expr:
            self.cell_embedding = nn.SequentialCell([
                nn.Dense(cell_input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Dense(512, hidden_dim)
            ])
        else:
            self.cell_embedding = nn.Dense(cell_dim, hidden_dim)

        # CLS Token & positional embedding
        self.cls_token = Parameter(ops.StandardNormal()((1, 1, hidden_dim)), name='cls_token')
        # Line too long fixed by splitting the long line
        self.pos_embedding = \
            Parameter(ops.StandardNormal()((1, gene_vocab_size + 3, hidden_dim)), name='pos_embedding')

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                                   dim_feedforward=4 * hidden_dim, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction Heads
        self.to_gene_pred = nn.Dense(hidden_dim, 1)
        self.cls_head = nn.Dense(hidden_dim, hidden_dim)
        self.recon_head = nn.SequentialCell([
            nn.Dense(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, cell_input_dim)
        ])

    def construct(self, gene_ids, gene_expr, drug_fp, cell_feat, mask=None):
        """
        Forward pass for the GenePertFormer model.

        Args:
            gene_ids (Tensor): Tensor of gene IDs.
            gene_expr (Tensor): Tensor of gene expressions.
            drug_fp (Tensor): Tensor of drug fingerprints.
            cell_feat (Tensor): Tensor of cell features.
            mask (Tensor, optional): Mask for the transformer encoder. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Predicted gene expression, CLS token output, and reconstructed cell features.
        """
        batch_size, _ = gene_ids.shape  # Renamed B to batch_size, G to _ (unused)

        id_embed = self.gene_embedding(gene_ids)       # [batch_size, G, H]
        expr_embed = self.expr_embedding(gene_expr)    # [batch_size, G, H]
        gene_embed = id_embed + expr_embed             # [batch_size, G, H]

        drug_token = self.drug_embedding(drug_fp).expand_dims(1)  # [batch_size, 1, H]

        if self.use_cell_expr:
            cell_raw = ops.Squeeze(-1)(cell_feat)        # [batch_size, G]
            cell_embed = self.cell_embedding(cell_raw)   # [batch_size, H]
        else:
            cell_embed = self.cell_embedding(cell_feat)  # [batch_size, H]
        cell_token = cell_embed.expand_dims(1)           # [batch_size, 1, H]

        cls = ops.BroadcastTo((batch_size, 1, self.hidden_dim))(self.cls_token)
        tokens = ops.Concat(axis=1)((cls, drug_token, cell_token, gene_embed))
        tokens = tokens + self.pos_embedding[:, :tokens.shape[1], :]

        x = self.encoder(tokens, src_key_padding_mask=mask)

        pred_gene = self.to_gene_pred(x[:, 3:, :]).squeeze(-1)  # [batch_size, G]
        cls_out = self.cls_head(x[:, 0, :])
        recon = self.recon_head(cell_embed)

        return pred_gene, cls_out, recon
