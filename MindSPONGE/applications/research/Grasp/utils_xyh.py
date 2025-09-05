import numpy as np
import pandas as pd
import pprint

def show_npdict(npdict, tag=None):
    '''print Dict elegantly'''
    if tag:
        print('*'*80)
        print(f'*{tag:^78}*')
        print('*'*80)
        print('\n')

    for k in sorted(list(npdict.keys())):
        v = npdict[k]
        if isinstance(v, np.ndarray):
            print(f'{f"{k}: {v.shape}, {v.dtype}":-<80}')
            if len(v.shape) == 0:
                print(v)
                continue
            v1 = v.copy()
            while len(v1.shape) > 1 and v1.shape[0] > 0:
                v1 = v1[0]
            print(v1[:min(10, len(v1))])
        else:
            print(f'{f"{k}, {type(v)}":-<80}')
            pprint.pprint(v)
        print('')

def reduce_dim(x, num):
    if not isinstance(x, np.ndarray):
        x = x.asnumpy()
    while len(x.shape) > num:
        x = x[0]
    return x

def print_restraint_info(d1):
    '''print sampled restraints' information'''
    d = d1.copy()
    contact_mask_input = reduce_dim(d["contact_mask_input"], 2)
    contact_mask_output = reduce_dim(d["contact_mask_output"], 2)
    true_contact = contact_mask_input * contact_mask_output
    false_contact = contact_mask_input * (1 - contact_mask_output)
    asym_id = reduce_dim(d['asym_id'], 1)
    is_intra = (asym_id[None] == asym_id[:, None])
    true_inter = (true_contact * (1 - is_intra)).sum() / 2
    true_intra = (true_contact * is_intra).sum() / 2
    false_inter = (false_contact * (1 - is_intra)).sum() / 2
    false_intra = (false_contact * is_intra).sum() / 2
    df = pd.DataFrame([[true_inter, true_intra], [false_inter, false_intra]], columns=['inter', 'intra'], index=['true', 'false'])
    df['sum'] = df.sum(1)
    df.loc['sum'] = df.sum(0)
    print(df)
