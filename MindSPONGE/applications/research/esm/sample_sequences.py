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
"""Sample sequence designs for a given structure"""

import argparse
from pathlib import Path
import os
import stat
import numpy as np
import mindspore as ms
import src.pretrained as pretrained
import src.util as util


def main():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")


    parser = argparse.ArgumentParser(
        description='Sample sequences based on a given structure.'
    )
    parser.add_argument(
        'pdbfile', type=str,
        help='input filepath, either .pdb or .cif',
        default='data/4uv3.cif',)
    parser.add_argument(
        '--chain', type=str,
        help='chain id for the chain of interest', default=None,
    )
    parser.add_argument(
        '--temperature', type=float,
        help='temperature for sampling, higher for more diversity',
        default=1.,
    )
    parser.add_argument(
        '--outpath', type=str,
        help='output filepath for saving sampled sequences',
        default='output/sampled_seqs.fasta',
    )
    parser.add_argument(
        '--num-samples', type=int,
        help='number of sequences to sample',
        default=1,
    )
    parser.add_argument('--device_id', help='device id', type=int, default=2)
    args = parser.parse_args()
    ms.set_context(device_target='GPU', device_id=args.device_id)

    model, _ = pretrained.esm_if1_gvp4_t16_142m_ur50()
    model.set_train(False)
    coords, seq = util.load_coords(args.pdbfile, args.chain)
    print('Sequence loaded from file:')
    print(seq)

    print(f'Saving sampled sequences to {args.outpath}.')

    Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(args.outpath, flags, modes), 'w') as f:
        for i in range(args.num_samples):
            print(f'\nSampling.. ({i+1} of {args.num_samples})')
            sampled_seq = model.sample(coords, temperature=args.temperature)
            print('Sampled sequence:')
            print(sampled_seq)
            f.write(f'>sampled_seq_{i+1}\n')
            f.write(sampled_seq + '\n')

            recovery = np.mean([(a == b) for a, b in zip(seq, sampled_seq)])
            print('Sequence recovery:', recovery)


if __name__ == '__main__':
    main()
