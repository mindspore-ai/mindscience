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
"""Evaluation for UFold"""
import time
import collections
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.dataset import GeneratorDataset
from mindspore import dtype as mstype
from src.utils import get_args, evaluate_exact_new
from src.model import Unet as FCNNet
from src.config import process_config
from src.data_generator import RNASSDataGenerator
from src.data_generator import DatasetCutConcatNewCanonicle as Dataset_FCN


args = get_args()
if args.nc:
    from src.postprocess import postprocess_new_nc as postprocess
else:
    from src.postprocess import postprocess_new as postprocess


def model_eval_all_test(contact_net, test_generator):
    """eval function"""
    result_no_train = list()
    seq_lens_list = list()
    batch_n = 0
    nc_map_nc = 0
    contacts_batch = 0
    map_no_train_nc = 0
    map_no_train = 0
    result_nc = list()
    result_nc_tmp = list()
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []
    cast = ops.Cast()

    for contacts, seq_embeddings, _, seq_lens, seq_ori, seq_name, nc_map, _ in test_generator:

        nc_map_nc = cast(nc_map, mstype.float32) * contacts
        if seq_lens.item() > 1500:
            continue
        if batch_n % 1000 == 0:
            print('Batch number: ', batch_n)

        batch_n += 1
        contacts_batch = ms.Tensor(cast(contacts, mstype.float32))
        seq_embedding_batch = ms.Tensor(cast(seq_embeddings, mstype.float32))
        seq_ori = ms.Tensor(cast(seq_ori, mstype.float32))
        seq_names.append(seq_name)
        seq_lens_list.append(seq_lens.item())
        tik = time.time()

        pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
                                 seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)

        nc_no_train = cast(nc_map, mstype.float32) * u_no_train
        map_no_train = cast((u_no_train > 0.5), mstype.float32)
        map_no_train_nc = cast((nc_no_train > 0.5), mstype.float32)
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)

        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train[i],
                                                                    contacts_batch[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp

        nc_map_nc = cast(nc_map_nc, mstype.float32)
        if nc_map_nc.sum() != 0:
            result_nc_tmp = list(map(lambda i: evaluate_exact_new(map_no_train_nc[i],
                                                                  nc_map_nc[i]), range(contacts_batch.shape[0])))
            result_nc += result_nc_tmp
            nc_name_list.append(seq_name)

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result_no_train)

    nt_exact_p_list = list(nt_exact_p)
    nt_exact_r_list = list(nt_exact_r)
    nt_exact_f1_list = list(nt_exact_f1)

    nt_exact_p = list(map(lambda i: nt_exact_p_list[i].asnumpy(), range(len(nt_exact_p_list))))
    nt_exact_r = list(map(lambda i: nt_exact_r_list[i].asnumpy(), range(len(nt_exact_r_list))))
    nt_exact_f1 = list(map(lambda i: nt_exact_f1_list[i].asnumpy(), range(len(nt_exact_f1_list))))

    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))


def main():
    ms.context.set_context(device_target=args.device_target, device_id=args.device_id)

    config_file = args.config
    test_file = args.test_files

    config = process_config(config_file)
    model_saved = args.ckpt_file
    batch_size_1 = config.batch_size_stage_1

    print('Loading test file: ', test_file)
    if test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('data/', test_file+'.pickle')
    else:
        test_data = RNASSDataGenerator('data/', test_file+'.cPickle')

    seq_len = test_data.data_y.shape[-2]
    print('Max seq length ', seq_len)

    test_set = Dataset_FCN(test_data)
    test_generator = GeneratorDataset(test_set, column_names=['contacts', 'seq_embeddings', 'matrix_reps',
                                                              'seq_lens', 'seq_ori', 'seq_name', 'nc_map', 'l_len',],
                                      num_parallel_workers=6,
                                      shuffle=True).batch(batch_size=batch_size_1, drop_remainder=True)

    contact_net = FCNNet(img_ch=17)

    print('==========Start Loading==========')
    param_dict = ms.load_checkpoint(model_saved)
    ms.load_param_into_net(contact_net, param_dict)
    print('==========Finish Loading==========')

    model_eval_all_test(contact_net, test_generator)

if __name__ == '__main__':
    # See module-level docstring for a description of the script.
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    main()
