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
"""inference model of cVAE"""
import argparse
import scipy.io as sio
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import minmax_scale
import mindspore as ms
from model import VAE, MDA

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dir', help='dataset directory',
                        default='dataset')
    parser.add_argument('--vae_encoder_layer_sizes', type=list, default=[1519, 1000], help='encoder layer sizes')
    parser.add_argument('--vae_latent_size', help='latent size', type=int, default=100)
    parser.add_argument('--vae_decoder_layer_sizes', help='decoder layer sizes', type=list, default=[1000, 1519])
    parser.add_argument('--cvae_checkpoint_dir', help='checkpoint directory of mda', type=str,
                        default='checkpoint/cvae.ckpt')
    parser.add_argument('--device_id', help='device id', type=int, default=0)
    parser.add_argument('--mda_select_nets', help='select nets of mda', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 10])
    parser.add_argument('--mda_checkpoint_dir', help='checkpoint directory of mda', type=str,
                        default='checkpoint/mda.ckpt')
    parser.add_argument('--ppmi_dir', help='PPMI directory', type=str,
                        default='dataset/PPMI/')
    parser.add_argument('--mda_feature_out', help='output directory of mda features', type=str,
                        default='dataset/')
    parser.add_argument('--model', help='select which model to run', type=str)
    args = parser.parse_args()
    ms.set_context(device_target='GPU', device_id=args.device_id)

    if args.model == 'mda':
        print("MDA start!")

        ORG = 'drug'
        SELECTNETS = args.mda_select_nets  # a number 1-10 (see below)

        # all possible combinations for architectures
        arch = {}
        arch['mda'] = {}
        arch['mda']['drug'] = {}
        arch['mda']['drug'] = {1: [9 * 100],
                               2: [9 * 1000, 9 * 100, 9 * 1000],
                               3: [9 * 1000, 9 * 500, 9 * 100, 9 * 500, 9 * 1000],
                               4: [9 * 1000, 9 * 500, 9 * 200, 9 * 100, 9 * 200, 9 * 500, 9 * 1000],
                               5: [9 * 1000, 9 * 800, 9 * 500, 9 * 200, 9 * 100, 9 * 200, 9 * 500, 9 * 800, 9 * 1000],
                               }

        # load PPMI matrices
        NETS = []
        input_dims = []
        for i in SELECTNETS:
            print("### [%d] Loading network..." % (i))
            N = sio.loadmat(args.ppmi_dir + ORG + '_net_' + str(i) + '.mat', squeeze_me=True)
            Net = N['Net'].todense()
            print("Net %d, NNofile_keywords=%d \n" % (i, np.count_nonzero(Net)))
            NETS.append(minmax_scale(Net))
            input_dims.append(Net.shape[1])

        NETS = ms.Tensor(np.array(NETS).astype(np.float32))
        model = MDA(input_dims, arch.get('mda').get('drug').get(3), train=False)
        print(model)
        ms.load_checkpoint(args.mda_checkpoint_dir, model)

        mid_feature = model(NETS)
        feature = mid_feature.asnumpy()
        feature = minmax_scale(feature)
        np.savetxt(args.mda_feature_out + ORG + 'Features.txt', feature, delimiter='\t', fmt='%s', newline='\n')
        print('MDA Done!')

    elif args.model == 'cvae':
        print('cVAE start!')
        print('dataset directory: ' + args.dir)
        directory = args.dir

        PATH = '{}/drugDisease.txt'.format(directory)
        print('train data path: ' + PATH)

        R = np.loadtxt(PATH)
        RTENSOR = R.transpose()

        whole_positive_index = []
        whole_negative_index = []
        for i in range(np.shape(RTENSOR)[0]):
            for j in range(np.shape(RTENSOR)[1]):
                if int(RTENSOR[i][j]) == 1:
                    whole_positive_index.append([i, j])
                elif int(RTENSOR[i][j]) == 0:
                    whole_negative_index.append([i, j])
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)

        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
        count = 0
        for i in whole_positive_index:
            data_set[count][0] = i[0]
            data_set[count][1] = i[1]
            data_set[count][2] = 1
            count += 1
        for i in negative_sample_index:
            data_set[count][0] = whole_negative_index[i][0]
            data_set[count][1] = whole_negative_index[i][1]
            data_set[count][2] = 0
            count += 1
        test_auc_fold = []
        test_aupr_fold = []
        DTITEST = data_set
        X = np.zeros((np.shape(RTENSOR)[0], np.shape(RTENSOR)[1]))
        for ele in DTITEST:
            X[ele[0], ele[1]] = ele[2]
        RTENSOR = ms.Tensor.from_numpy(X.astype('float32'))

        model = VAE(args.vae_encoder_layer_sizes, args.vae_latent_size, args.vae_decoder_layer_sizes)
        ms.load_checkpoint(args.cvae_checkpoint_dir, model)
        print(model)

        score, _, _, _, _ = model(RTENSOR)
        print(score.asnumpy().shape)
        ZSCORE = score.asnumpy()

        pred_list = []
        ground_truth = []
        for ele in DTITEST:
            pred_list.append(ZSCORE[ele[0], ele[1]])
            ground_truth.append(ele[2])
        test_auc = roc_auc_score(ground_truth, pred_list)
        test_aupr = average_precision_score(ground_truth, pred_list)
        print('test auc aupr', test_auc, test_aupr)
        print('cVAE Done!')
