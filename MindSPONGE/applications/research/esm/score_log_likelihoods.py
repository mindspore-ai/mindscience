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
"""Score sequences for a given structure"""

import argparse
from pathlib import Path
import os
import stat
from biotite.sequence.io.fasta import FastaFile, get_sequences
import numpy as np
from tqdm import tqdm
import mindspore as ms
import src
import src.pretrained
import src.util


def main():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    parser = argparse.ArgumentParser(
        description='Score sequences based on a given structure.'
    )
    parser.add_argument(
        '--pdbfile', type=str,
        help='input filepath, either .pdb or .cif', default='./data/5YH2.cif',
    )
    parser.add_argument(
        '--seqfile', type=str,
        help='input filepath for variant sequences in a .fasta file', default='./data/5YH2_mutated_seqs.fasta',
    )
    parser.add_argument(
        '--outpath', type=str,
        help='output filepath for scores of variant sequences',
        default='output/sequence_scores.csv',
    )
    parser.add_argument(
        '--chain', type=str,
        help='chain id for the chain of interest', default='C',
    )
    parser.add_argument('--device_id', help='device id', type=int, default=2)
    args = parser.parse_args()
    ms.set_context(device_target='GPU', device_id=args.device_id)

    model, alphabet = src.pretrained.esm_if1_gvp4_t16_142m_ur50()
    model.set_train(False)
    coords, seq = src.util.load_coords(args.pdbfile, args.chain)
    print('Native sequence loaded from structure file:')
    print(seq)
    print('\n')

    ll, _ = src.util.score_sequence(
        model, alphabet, coords, seq)
    print('Native sequence')
    print(f'Log likelihood: {ll:.2f}')
    print(f'Perplexity: {np.exp(-ll):.2f}')

    print('\nScoring variant sequences from sequence file..\n')
    infile = FastaFile()
    infile.read(args.seqfile)
    seqs = get_sequences(infile)
    Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(args.outpath, flags, modes), 'w') as fout:
        fout.write('seqid,log_likelihood\n')
        for header, seq in tqdm(seqs.items()):
            ll, _ = src.util.score_sequence(
                model, alphabet, coords, str(seq))
            fout.write(header + ',' + str(ll) + '\n')
    print(f'Results saved to {args.outpath}')


if __name__ == '__main__':
    main()
