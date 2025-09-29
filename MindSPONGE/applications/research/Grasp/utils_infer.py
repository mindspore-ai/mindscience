import os
import re
import gc
import time
import glob
import pickle
import datetime
import numpy as np
import pandas as pd
import mindspore.context as context
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype

from model import MegaFold
from model import compute_confidence, compute_ranking_score
from common.protein import to_pdb, from_prediction
from common.protein import to_pdb, from_prediction, PDB_CHAIN_IDS
from common.utils import trans_ckpt
from data import MultimerFeature
from restraint_sample import BINS

from mindspore import Tensor
from mindspore import load_checkpoint, nn, load_param_into_net
from mindspore.communication import init, get_rank

from mindsponge1.common.protein import from_pdb_string_all_chains
from mindsponge1.data.data_transform import pseudo_beta_fn
from mindsponge1.cell.amp import amp_convert
from mindsponge1.common.config_load import load_config
from mindsponge1.common import residue_constants
# import mindspore.train.amp as amp
# from mindspore.ops import operations as P
# amp.AMP_AUTO_BLACK_LIST.clear()
# amp.AMP_AUTO_BLACK_LIST.extend([P.LayerNorm, P.Softmax])
# amp.AMP_AUTO_WHITE_LIST.extend([P.BiasAdd])
SEED = 20230820


# def get_seqs_from_fasta(fasta):
#     with open(fasta, 'r') as f:
#         cont = [i.strip() for i in f.readlines()]
#     seqs = cont[1::2]
#     return seqs

def parse_fasta(fasta):
    with open(fasta, 'r') as f:
        cont = [i.strip() for i in f.readlines()]
    seqdict = {}
    desc = None
    for line in cont:
        if line.startswith('>'):
            if desc is not None:
                seqdict[desc] = seq
            seq = ''
            desc = line[1:].strip()
        else:
            seq += line
    seqdict[desc] = seq
    return seqdict

def get_mapping_from_fasta(fasta):
    mapping = parse_fasta(fasta)
    mapping = {k.split('_')[-1][0]: v for k, v in mapping.items()}
    return mapping
    # with open(fasta, 'r') as f:
    #     cont = [i.strip() for i in f.readlines()]
    # mapping = {}
    # for i in range(0, len(cont), 2):
    #     k = cont[i].split('_')[-1][0]
    #     v = cont[i+1]
    #     mapping[k] = v
    # print(mapping)
    # return mapping

def get_order_from_seqs(seqs):
    seqdict = {}
    for seq in seqs:
        if seq not in seqdict:
            seqdict[seq] = 1
        else:
            seqdict[seq] += 1

    p = 0 # pointer
    for seq, k in seqdict.items():
        ls = []
        for i in range(k):
            ls.append(range(p, p+len(seq)))
            p += len(seq)
        seqdict[seq] = ls
    ls = []
    for seq in seqs:
        ls.append(seqdict[seq][0])
        seqdict[seq].pop(0)
    return ls

def reorder(x, slices, axis):
    return np.concatenate([np.take(x, i, axis) for i in slices], axis=axis)
    
def reorder_features(feats, seqs):
    ord = get_order_from_seqs(seqs)
    seqlen = feats['aatype'].shape[0]
    for k, v in feats.items():
        # print(k, v.shape)
        for i, s in enumerate(v.shape):
            if s == seqlen:
                v = reorder(v, ord, i)
        feats[k] = v
        
        
def np_pad(array, seqlen, axis=None):
    pad_width = []
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        axis = range(len(array.shape))
    # print("=================array===================: ", array.shape)
    for i, n in enumerate(array.shape):
        if i in axis:
            pad_width.append((0, seqlen - n))
        else:
            pad_width.append((0, 0))
    return np.pad(array=array, pad_width=pad_width)

def get_dist_from_protein(prot):
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(prot.aatype, prot.atom_positions, prot.atom_mask)
    pred_dist = np.sqrt(((pseudo_beta[:, None] - pseudo_beta[None]) ** 2).sum(-1) + 1e-8)
    pseudo_beta_mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None]
    return pred_dist, pseudo_beta_mask_2d

def get_nbdist_avg_ca(prot, asym_id, break_thre=5.0):
    """compute averaged neihbour ca distance for each residue"""
    # atom_types = [
    #     'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    #     'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    #     'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    #     'CZ3', 'NZ', 'OXT'
    # ]
    ca_idx = 1
    ca_pos = prot.atom_positions[..., ca_idx, :] #[nres, natom, 3]
    mask = prot.atom_mask[..., ca_idx]
    nbdist = np.sqrt(((ca_pos[1:]-ca_pos[:-1])**2).sum(-1)+1e-8)
    mask_nbdist = 4.0
    for i in np.where(1-mask)[0]:
        print(i)
        nbdist[i] = mask_nbdist
        if i>0:
            nbdist[i-1] = mask_nbdist
    nbdist_leftadd = np.concatenate([[nbdist[0]], nbdist])
    nbdist_rightadd = np.concatenate([nbdist, [nbdist[-1]]])
    is_chain_start = asym_id!=np.concatenate(([[-1], asym_id[:-1]]))
    is_chain_end = asym_id!=np.concatenate((asym_id[1:], [100000]))
    nbdist_left = np.where(is_chain_start, nbdist_rightadd, nbdist_leftadd)
    nbdist_right = np.where(is_chain_end, nbdist_leftadd, nbdist_rightadd)
    nbdist_avg = (nbdist_left+nbdist_right)/2

    break_num = int((nbdist_left>break_thre).sum())
    max_nb_dist = nbdist_left.max()

    return nbdist_avg, break_num, max_nb_dist
    

def dist_onehot(dist, bins):
    x = (dist[..., None] > bins).sum(-1)
    return np.eye(len(bins) + 1)[x]

def get_range(x):
    lowers = np.concatenate([[0], BINS])
    uppers = np.concatenate([BINS, [np.inf]])
    intervals = [(i, j) for i, j, k in zip(lowers, uppers, x) if k]
    ys = []
    last = None
    for i, j in intervals:
        if (last is not None) and (last == i):
            ys[-1][-1] = j
        else:
            ys.append([i, j])
        last = j
    return ','.join([f'{i}-{j}' for i, j in ys])


def compute_recall(satis, mask, conf):
    if mask.sum() <= 0:
        return None, None
    
    recall = (satis*mask).sum()/(mask.sum()+1e-8)
    recall_conf = (satis*mask*conf).sum()/((mask*conf).sum())
    return recall, recall_conf

def compute_rm_score(values, thres):
    score = 0
    scale = 1
    assert len(values) == len(thres), (values, thres)
    for value, thre in zip(values, thres):
        if (thre is not None):
            if (value>thre):
                score += (value-thre)*scale
            scale *= 100
    return score


def generate_terminal_mask(asym_id, n):
    is_end = asym_id != np.concatenate([asym_id[1:], [-1]])
    is_start = asym_id != np.concatenate([[0], asym_id[: -1]])
    end_idx = np.where(is_end)[0]
    start_idx = np.where(is_start)[0]
    term_idx = np.concatenate([end_idx, start_idx])
    idx = np.arange(len(asym_id))
    mask = (np.abs(idx[:, None] - term_idx[None]) >= n).all(axis=-1).astype(int)
    mask = mask[None] # only mask the other side which is different from the interface.
    return mask


def filter_restraints(restraints, restraints0, prot, nbdist_ca_thre=5, max_rm_ratio=0.2, viol_thre=5, mask_terminal_residues=0):
    # restraints0: initial restraints.
    # restraints: current restraints.

    plddt = prot.b_factors.max(-1)
    pred_dist, pseudo_beta_mask_2d = get_dist_from_protein(prot)
    mask_intrachain = restraints['asym_id'][None] == restraints['asym_id'][:, None]
    terminal_residue_mask = generate_terminal_mask(restraints['asym_id'], mask_terminal_residues)

    d = pred_dist + mask_intrachain*1000 + (1-pseudo_beta_mask_2d) * 1000 + (1-terminal_residue_mask) * 1000
    # dist_thre=10.0
    # plddts_2d = (d<=dist_thre)*plddt[None]
    # plddt_otherside = plddts_2d.max(axis=1)
    sbr = restraints['sbr']
    sbr_high = (sbr > (1 / sbr.shape[-1]))

    not_high_bin = 1-sbr_high
    upper_1d = np.concatenate([BINS, [100,]])
    sbr_upper_thre = (upper_1d-1e6*not_high_bin).max(-1)
    sbr_upper_viol_dist = (pred_dist-sbr_upper_thre)
    sbr_max_viol_dist = (sbr_upper_viol_dist * restraints['sbr_mask']).max()
    sbr_viol_num = ((sbr_upper_viol_dist * restraints['sbr_mask']) > 0).sum() / 2
    interface_viol_dist = ((d.min(axis=-1)-8.0)*restraints['interface_mask'])
    interface_max_viol_dist = interface_viol_dist.max()
    interface_viol_num = (interface_viol_dist>0).sum()
    viol_num = sbr_viol_num + interface_viol_num
    max_viol_dist = max(sbr_max_viol_dist, interface_max_viol_dist)
    pred_dist_onehot = dist_onehot(pred_dist, BINS)
    sbr_satis = (sbr_high * pred_dist_onehot).sum(-1) * pseudo_beta_mask_2d
    nbdist_avg_ca, break_num, max_nb_dist = get_nbdist_avg_ca(prot, asym_id=restraints['asym_id'])
    includ_mat = np.zeros_like(restraints['sbr_mask'])
    includ_if = np.zeros_like(restraints['interface_mask'])
    
    
    def resi(i, ds=None):
        cid = PDB_CHAIN_IDS[int(restraints['asym_id'][i])-1]
        rid = prot.residue_index[i]
        y = f'{cid}{rid}/conf{plddt[i]:.2f}/nbdist_avg_ca{nbdist_avg_ca[i]:.2f}'
        if ds is not None:
            y += f'/dist_cb{ds[i]:.2f}'
        return y
    
    def print_pair(ps):
        ps = [(i, j) for i, j in ps if i<j]
        satisfied_num = 0
        included_num = 0

        nbdists = [(nbdist_avg_ca[i]+nbdist_avg_ca[j])/2 for i, j in ps]
        viol_dists = [sbr_upper_viol_dist[i, j] for i, j in ps]
        rm_scores = [compute_rm_score((viol_dist, nb_dist), (viol_thre, nbdist_ca_thre)) for nb_dist, viol_dist in zip(nbdists, viol_dists)]
        rm_thre = np.quantile(rm_scores, 1-max_rm_ratio)
        
        for (i, j), rm_score in zip(ps, rm_scores):
            if (rm_score <= rm_thre):
                includ_mat[i,j] = 1
                includ_mat[j,i] = 1
                included_num += 1
                filter_info = 'Included!'
            else:
                filter_info = 'Excluded!'
            if sbr_satis[i,j]:
                satisfied_num += 1
                satis_info = 'Satisfied!'
            else:
                satis_info = 'Violated! '
            print(f'{filter_info} {satis_info} {resi(i)}<==>{resi(j, pred_dist[i])}, range: {get_range(sbr_high[i,j])}, rm_score {rm_score}, rm_thre {rm_thre}')
        print(f'>>>>> Total {len(ps)}: {included_num} included, {satisfied_num} satisfied')
    
    # print interface info ==========================================================
    if_num = int(restraints['interface_mask'].sum())
    if if_num>0:
        print('interface restraints:')
        included_num = 0
        satisfied_num = 0
        nbdists = [nbdist_avg_ca[i] for i in np.where(restraints['interface_mask'])[0]]
        viol_dists = [d[i].min()-8.0 for i in np.where(restraints['interface_mask'])[0]]
        rm_scores = [compute_rm_score((viol_dist, nb_dist), (viol_thre, nbdist_ca_thre)) for nb_dist, viol_dist in zip(nbdists, viol_dists)]
        rm_thre = np.quantile(rm_scores, 1-max_rm_ratio)
        for i, rm_score in zip(np.where(restraints['interface_mask'])[0], rm_scores):
            # js = np.where((plddts_2d[i])>0)[0]
            if d[i].min()<=8.0:
                satisfied_num += 1
                satis_info = 'Satisfied!'
            else:
                satis_info = 'Violated! '
                
            # if len(js)==0:
            #     print(f'Excluded! {satis_info} {resi(i)}<==>{resi(np.argmin(ds), ds)}')
            # else:
                # jmax = np.argmax(plddts_2d[i])
            
            if (rm_score<=rm_thre):
                includ_if[i] = 1
                included_num += 1
                filter_info = 'Included!'
            else:
                filter_info = 'Excluded!'
            print(f'{filter_info} {satis_info} {resi(i)} {d[i].min()}, rm_score{rm_score}, rm_thre{rm_thre}')

        print(f'>>>>> Total {if_num}, {included_num} included, {satisfied_num} satisfied')
    
    # print sbr info =================================================================
    intra_ps = np.transpose(np.where(restraints['sbr_mask']*mask_intrachain))
    inter_ps = np.transpose(np.where(restraints['sbr_mask']*(1-mask_intrachain)))
    intra_sbr = int(len(intra_ps)/2)
    inter_sbr = int(len(inter_ps)/2)
    tot_sbr = intra_sbr+inter_sbr
    if tot_sbr >0:          
        print(f'inter-residue restraints: {tot_sbr}({inter_sbr} inter-chain + {intra_sbr} intra-chain)')
    if inter_sbr > 0:
        print('Inter-chain restraints')
        print_pair(inter_ps)
    if intra_sbr > 0:
        print('Intra-chain restraints')
        print_pair(intra_ps)
    
    # update restraints based on plddts ==============================================
    tot_before = int(tot_sbr+if_num)
    restraints['interface_mask'] = includ_if * restraints['interface_mask']
    restraints['sbr_mask'] = includ_mat * restraints['sbr_mask']
    restraints['sbr'] = restraints['sbr'] * restraints['sbr_mask'][:,:,None]
    tot_after = int((restraints['interface_mask']).sum() + (restraints['sbr_mask']).sum()/2)
    rm_num = int(tot_before - tot_after)

    # compute recall, breakage
    sbr_mask0 = restraints0['sbr_mask']
    sbr0 = restraints0['sbr']
    sbr_high0 = (sbr0 > (1 / sbr0.shape[-1]))
    sbr_satis0 = (sbr_high0 * pred_dist_onehot).sum(-1) * pseudo_beta_mask_2d


    
    
    interface_mask0 = restraints0['interface_mask']
    interface_satis0 = d.min(axis=1)<=8
    conf_2d = (plddt[None]+plddt[:, None])/2

    recall_dict = {
        'interchain': (*compute_recall(sbr_satis0, sbr_mask0*np.triu(sbr_mask0)*(1-mask_intrachain), conf_2d), 1),
        'intrachain': (*compute_recall(sbr_satis0, sbr_mask0*np.triu(sbr_mask0)*mask_intrachain, conf_2d), 0.5),
        'interface':  (*compute_recall(interface_satis0, interface_mask0, plddt), 1)
    }
    
    recall_dict = {
        k: v for k, v in recall_dict.items() if v[0] is not None
    }


    print('Breakage info ==========')
    print(f'Break number: {break_num}, Max neighbour CA dist: {max_nb_dist}\n')

    print('Recall info=============')
    recall = 0
    recall_conf = 0
    w = 0

    for k, v in recall_dict.items():
        if v[0] is None:
            continue
        print(f'{k} (w {v[2]}): recall {v[0]}, recall weighted by confidence: {v[1]}')
        recall += v[0]*v[2]
        recall_conf += v[1]*v[2]
        w += v[2]

    if w == 0:
        # no restraints
        recall = None
        recall_conf = None
    else:
        recall /= w
        recall_conf /= w

    return rm_num, break_num, max_nb_dist, recall, recall_conf, viol_num, max_viol_dist






def generate_split_first_num(split_file):
    with open(split_file, 'r') as f:
        split = [i.strip().split(',') for i in f.readlines()]
    print(split)
    return len(split[0])

def generate_index(lenlsls):
    ls = []
    for lenls in lenlsls:
        last = 0
        for l in lenls:
            a = np.arange(l)+last
            ls.append(a)
            last = a[-1]+200
    return np.concatenate(ls, axis=0)

def dict_update_keepdtype(d1, d2):
    for k, v in d2.items():
        if k in d1:
            d1[k] = v.astype(d1[k].dtype)

def generate_id(seqs, first_num):
    s1, s2 = [''.join(s) for s in [seqs[:first_num], seqs[first_num:]]]
    if s1 == s2:
        entity_id = np.repeat(1, len(s1)+len(s2))
        sym_id = np.repeat([1, 2], (len(s1), len(s2)))
    else:
        entity_id = np.repeat([1, 2], (len(s1), len(s2)))
        sym_id = np.repeat(1, len(s1)+len(s2))
    asym_id = np.repeat([1, 2], (len(s1), len(s2)))
    return asym_id, sym_id, entity_id

    
def update_feature_make_two_chains(feat, first_num, seqs):
    lenls = np.array([len(i) for i in seqs])
    lenlsls = [lenls[:first_num], lenls[first_num:]]

    asym_id, sym_id, entity_id = generate_id(seqs, first_num)
    d_update = {
        'residue_index': generate_index(lenlsls),
        'asym_id': asym_id,
        'sym_id': sym_id,
        'entity_id': entity_id,
        'assembly_num_chains': np.array(2)
    }
    dict_update_keepdtype(feat, d_update)
    
def get_distri(cutoff, fdr):
    xbool = np.concatenate([BINS, [np.inf]])<=cutoff
    x = np.ones(len(BINS)+1)
    x[xbool] = (1-fdr) * (x[xbool]/x[xbool].sum())
    x[~xbool] = fdr * (x[~xbool]/x[~xbool].sum())
    assert x[xbool].max() > x[~xbool].max(), (x[xbool].max(), x[~xbool].max())
    return x

class SplitNamelist:
    def __init__(self, rank_id, rank_size, outdir, rotate_split=False, key=None):
        self.rank_id = rank_id
        self.rank_size = rank_size
        self.rotate_split = rotate_split
        self.outdir = outdir
        self.key = key
        os.makedirs(self.outdir, exist_ok=True)
        self.completion_flag_file = self.get_flag(self.rank_id)

    def get_flag(self, rank_id):
        if self.key is None:
            completion_flag_file = f'{self.outdir}/.complete_flag_rank{rank_id}.tmp'
        else:
            completion_flag_file = f'{self.outdir}/.complete_flag_rank{rank_id}_{self.key}.tmp'
        return completion_flag_file

    def split_namelist(self, namelist):
        if not self.rotate_split:
            d, m = divmod(len(namelist), self.rank_size)
            nums = np.repeat([d+1, d], [m, self.rank_size-m])
            start = int(nums[:self.rank_id].sum())
            namelist_slice = namelist[start: start+nums[self.rank_id]]
        else:
            namelist_slice = []
            for i in range(self.rank_id, len(namelist), self.rank_size):
                namelist_slice.append(namelist[i])
        print(f'Rank {self.rank_id}/{self.rank_size}: {len(namelist_slice)}/{len(namelist)}: {namelist_slice[:2]} ...', flush=True)
        return namelist_slice
    
    def start_job(self):
        print(f'start job completion monitor for rank id {self.rank_id}')
        if os.path.isfile(self.completion_flag_file):
            os.remove(self.completion_flag_file)
        assert (not os.path.exists(self.completion_flag_file))
        
    def complete(self):
        print(f'job complete for rank id {self.rank_id}')
        print(f'generate temporary complete flag file {self.completion_flag_file}')
        with open(self.completion_flag_file, 'w') as f:
            f.write(f'job complete for rank {self.rank_id}')

    def check_all_complete(self):
        comp_files = [i for i in range(self.rank_size) if os.path.exists(self.get_flag(i))]
        not_finish = list(set(range(self.rank_size)) - set(comp_files))
        not_finish.sort()
        print(f'current completion status: {len(comp_files)}/{self.rank_size}, not finished: {not_finish}')
        return len(comp_files) == self.rank_size
    
        

class DataGenerator:
    def __init__(self, raw_feat_dir, fasta_dir=None, reorder=False):
        self.raw_feat_dir = raw_feat_dir
        self.fasta_dir = fasta_dir
        self.reorder = reorder
        self.files_dict = {}

    def get_pattern(self, pdb_id):
        pat_dict = {
            'raw_feat': f'{self.raw_feat_dir}/{pdb_id}*.pkl',
            'fasta': f'{self.fasta_dir}/{pdb_id}*.fasta'
        }
        return pat_dict

    def _glob_file(self, pattern):
        files = glob.glob(pattern)
        assert len(files)<=1, files
        return files[0] if len(files)==1 else None
    
    def get_files(self, pdb_id):
        if pdb_id in self.files_dict:
            return self.files_dict[pdb_id]
        pat_dict = self.get_pattern(pdb_id)
        file_dict = {k: self._glob_file(v) for k, v in pat_dict.items()}
        self.files_dict[pdb_id] = file_dict
        return file_dict
    
    def get_feat(self, pdb_id):
        # raw feat
        raw_pkl = self.get_files(pdb_id)['raw_feat']
        with open(raw_pkl, "rb") as f:
            raw_feature = pickle.load(f)
        if self.reorder:
            print('reorder features')
            seqs = list(self.get_seqs_dict(pdb_id).values())
            reorder_features(raw_feature, seqs)
        return raw_feature
    
    def get_len(self, pdb_id):
        raw_feat = self.get_feat(pdb_id)
        return raw_feat['msa'].shape[1] if raw_feat is not None else 100000
    
    def get_seqs_dict(self, pdb_id):
        fasta_file = self.get_files(pdb_id)['fasta']
        return parse_fasta(fasta_file)
    
    def get_data(self):
        # overwrite for specific cases
        raise NotImplementedError
    
class ModelGenerator:
    def __init__(self, arguments, ckpt_dir):
        data_cfg = load_config(arguments.data_config)
        model_cfg = load_config(arguments.model_config)
        # print("this is model_cfg: ", model_cfg)
        self.seq_length = int(arguments.seq_len)
        data_cfg.eval.crop_size = self.seq_length
        model_cfg.seq_length = self.seq_length
        slice_key = "seq_" + str(model_cfg.seq_length)
        slice_val = vars(model_cfg.slice)[slice_key]
        model_cfg.slice = slice_val
        data_cfg.common.target_feat_dim = 21 # TARGET_FEAT_DIM
        model_cfg.common.target_feat_dim = 21 # TARGET_FEAT_DIM
        self.arguments = arguments
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.processed_feature = MultimerFeature(arguments.mixed_precision)
        # ckpt
        self.ckpt_dir = ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            raise ValueError(f'checkpoint directory {self.ckpt_dir} does not exist')
        self.last_ckpt = None
        if os.path.isdir(self.ckpt_dir):
            self.ckpt = None
        else:
            self.ckpt = self.ckpt_dir

    def get_ckpt(self, ckpt_id):
        ckpt = f'{self.ckpt_dir}/step_{ckpt_id}.ckpt'
        if not os.path.isfile(ckpt):
            ckpt = f'{self.ckpt_dir}/{ckpt_id}.ckpt'
        return ckpt
        
    def get_model(self, ckpt_id):
        if self.ckpt is not None:
            print(f'loading model from {self.ckpt}, not use ckpt id{ckpt_id}')
            ckpt = self.ckpt
        else:
            ckpt = self.get_ckpt(ckpt_id)
        if self.last_ckpt is None or self.last_ckpt !=  ckpt:
            print(f'Initializing model from {ckpt}')
            megafold_multimer = MegaFold(self.model_cfg, mixed_precision=self.arguments.mixed_precision, 
                                         device_num=self.arguments.device_num)
            megafold_multimer.to_float(mstype.float16)
            # print("debug network", megafold_multimer)
            # fp32_white_list = (nn.Softmax, nn.LayerNorm)
            # amp_convert(megafold_multimer, fp32_white_list)
            # megafold_multimer = amp.auto_mixed_precision(megafold, amp_level="auto", dtype=mstype.float16)
            self.model = megafold_multimer
            params = load_checkpoint(ckpt)
            params_infer = trans_ckpt(params)
            
            # print("preprocess_msa: ", params_infer['preprocess_msa.weight'].asnumpy())

            for key in params_infer.keys():
                if "msa_row_attention_with_pair_bias.query_norm_gammas" in key:
                    print("debug", key)
            load_param_into_net(self.model, params_infer)
        return self.model
    
    def model_process_data(self, raw_feature):
        feat = self.processed_feature.pipeline(self.model_cfg, self.data_cfg, raw_feature)
        return feat

def distance(points):
    return np.sqrt(np.sum((points[:, None] - points[None, :])**2,
                            axis=-1))

def mask_mean(mask, value, eps=1e-10):
    mask_shape = mask.shape
    value_shape = value.shape

    axis = list(range(len(mask_shape)))

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
    
    return (np.sum(mask * value, axis=tuple(axis)) /
            (np.sum(mask, axis=tuple(axis)) * broadcast_factor + eps))


def recycle_cond(i, prev, next_in, feat, recycle_early_stop_tolerance):
    print("start recycle_cond")

    ca_idx = residue_constants.atom_order['CA']
    sq_diff = np.square(distance(prev[:, ca_idx, :].astype(np.float64)) -
                         distance(next_in[:, ca_idx, :].astype(np.float64)))
    seq_mask_idx = 8
    mask = feat[seq_mask_idx][:, None] * feat[seq_mask_idx][None, :]
    sq_diff = mask_mean(mask.astype(np.float64), sq_diff)
    diff = np.sqrt(sq_diff + 1e-8)
    has_exceeded_tolerance = (
        (i == 0) | bool(diff > recycle_early_stop_tolerance)
    )
    print(f"recycle {i} diff: {diff}")
    print("end recycle_cond: ", has_exceeded_tolerance)
    # mydict = {
    #     'i': i,
    #     'sq_diff': sq_diff.asnumpy(),
    #     'diff': diff.asnumpy(),
    #     'prev': prev.asnumpy(),
    #     'next_in': next_in.asnumpy(),
    #     'mask': mask.asnumpy()
    # }
    # with open(f'/job/file/rec_{i}.pkl', 'wb') as f:
    #     pickle.dump(mydict, f)
    return has_exceeded_tolerance, diff.item()

def grasp_infer_quick(model_gen: ModelGenerator, ckpt_id, raw_feature: dict, restraints: dict, output_prefix, 
                nbdist_ca_thre=5.0, viol_thre=5.0, mask_terminal_residues=0, iter=5, max_rm_ratio=0.2, left_ratio=0.2, same_msa_across_recycle=True,
                num_recycle=20, dtype=np.float16, seed=None, recycle_early_stop_tolerance=0.5):
    print('Using quick inference')
    ori_res_length = raw_feature['msa'].shape[1]
    # run with no restraints provided
    if restraints is None:
        restraints = {
            'sbr': np.zeros((ori_res_length, ori_res_length, len(BINS) + 1)),
            'sbr_mask': np.zeros((ori_res_length, ori_res_length)),
            'interface_mask': np.zeros(ori_res_length),
            'asym_id': raw_feature['asym_id']
        }

    mydicts = []

    restraints0 = restraints.copy()
    
    t0 = time.time()
    megafold_multimer = model_gen.get_model(ckpt_id)
    seq_length = model_gen.seq_length
    
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    if seed is not None:
        np.random.seed(seed)

    feat_list = []

    left_thre = (restraints['interface_mask'].sum() + restraints['sbr_mask'].sum()/2)*left_ratio
    left_thre = int(np.ceil(left_thre))
    print(f'At least {left_thre} restraints will be used in the final iteration')

    # initialize prevs
    prev_pos = Tensor(np.zeros([seq_length, 37, 3]).astype(dtype))
    prev_msa_first_row = Tensor(np.zeros([seq_length, 256]).astype(dtype))
    prev_pair = Tensor(np.zeros([seq_length, seq_length, 128]).astype(dtype))
    prev_prev_pos = prev_pos.asnumpy()
    next_in_prev_pos = prev_pos.asnumpy()
    it = 0
    num_recycle_cur_iter = 0
    max_recycle_per_iter = 4

    for i in range(num_recycle):

        print("now its num_recycle", i)

        # pad restraints to fixed length
        sbr = Tensor(np_pad(restraints['sbr'], seq_length, axis=(0, 1)).astype(dtype))
        sbr_mask = Tensor(np_pad(restraints['sbr_mask'], seq_length, axis=(0, 1)).astype(dtype))
        interface_mask = Tensor(np_pad(restraints['interface_mask'], seq_length, axis=0).astype(dtype))
        
        # process data
        f_i = 0 if same_msa_across_recycle else i
        if len(feat_list)-1 < f_i:
            feat_list.append(model_gen.model_process_data(raw_feature))
        feat = feat_list[f_i]

        # inference
        prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits, aligned_error_logits, aligned_error_breaks = megafold_multimer(*feat_i,
                                                                                        sbr, sbr_mask, interface_mask,
                                                                                        prev_pos,
                                                                                        prev_msa_first_row,
                                                                                        prev_pair)
        prev_prev_pos = next_in_prev_pos
        next_in_prev_pos = prev_pos.asnumpy()
        # compute diff
        has_exceeded_tolerance, diff = recycle_cond(i, prev_prev_pos, next_in_prev_pos, feat, recycle_early_stop_tolerance)
        num_recycle_cur_iter += 1

        end_cur_iter = (not has_exceeded_tolerance) or (num_recycle_cur_iter >= max_recycle_per_iter)
        print(f"iter: {it+1}, recycle: {i}, diff: {diff}, has_exceeded_tolerance: {has_exceeded_tolerance}, end_cur_iter: {end_cur_iter}", flush=True)

        if end_cur_iter:
            print(f"early stop: {i}, diff: {diff}, iter: {it+1} =============================")

            # extract results
            final_atom_positions, predicted_lddt_logits  = [i.asnumpy()[:ori_res_length] for i in (prev_pos, predicted_lddt_logits)]
            final_atom_mask = feat[16][:ori_res_length]
            confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)
            b_factors = plddt[:, None] * final_atom_mask
            aligned_error_logits = aligned_error_logits.asnumpy()[:ori_res_length, :ori_res_length]
            ranking_score = compute_ranking_score(aligned_error_logits, aligned_error_breaks.asnumpy(), raw_feature['asym_id'])
            ranking_score = round(ranking_score*100, 2)

            unrelaxed_protein = from_prediction(final_atom_positions,
                                                final_atom_mask,
                                                feat[0][:ori_res_length],
                                                feat[1][:ori_res_length],
                                                b_factors,
                                                feat[5][:ori_res_length] - 1,
                                                remove_leading_feature_dimension=False)
            
            

            # Write sturcutres into pdb files
            pdb_file = to_pdb(unrelaxed_protein)
            pdb_path = f'{output_prefix}_iter{it+1}.pdb'
            with open(pdb_path, 'w') as f:
                f.write(pdb_file)
                
            # filter restraints
            print(f'Filter Restraints Iteration {it+1}')
            rm_num, break_num, max_nb_dist, recall, recall_conf, viol_num, max_viol_dist = filter_restraints(restraints, restraints0, unrelaxed_protein, nbdist_ca_thre=nbdist_ca_thre, max_rm_ratio=max_rm_ratio, viol_thre=viol_thre, mask_terminal_residues=mask_terminal_residues)
            print(f'Filter out {rm_num} restraint(s), confidence {confidence}, 0.8iptm+0.2ptm {ranking_score}')
            rest = int(restraints['interface_mask'].sum() + restraints['sbr_mask'].sum()/2)
            
            # record
            assert rm_num >=0, rm_num
            mydict = {
                'Iter': it+1,
                'Conf': round(confidence, 3),
                'RankScore': ranking_score,
                'Total': rm_num+rest,
                'Remove': rm_num,
                'Rest': rest,
                'MaxNbDist': max_nb_dist,
                'BreakNum': break_num,
                'Recall': recall,
                'RecallByConf': recall_conf,
                'Recycle_num': num_recycle_cur_iter,
                'Diff': round(diff, 3),
                'ViolNum': int(viol_num),
                'MaxViolDist': round(max_viol_dist, 2),
                'Time': round(time.time()-t0, 2)
            }
            
            mydicts.append(mydict)
            if (rest <= left_thre) or (rm_num == 0) or (it>=iter-1):
                break
            t0 = time.time()
            it += 1
            num_recycle_cur_iter = 0
    df = pd.DataFrame(mydicts)
    return df

def grasp_infer(model_gen: ModelGenerator, ckpt_id, raw_feature: dict, restraints: dict, output_prefix, 
                nbdist_ca_thre=5.0, viol_thre=5.0, mask_terminal_residues=0, iter=5, max_rm_ratio=0.2, left_ratio=0.2, baseline=False, same_msa_across_recycle=True,
                num_recycle=20, dtype=np.float16, seed=None, recycle_early_stop_tolerance=0.5, device_num=8):
    
    ori_res_length = raw_feature['msa'].shape[1]
    
    # run with no restraints provided
    if restraints is None:
        restraints = {
            'sbr': np.zeros((ori_res_length, ori_res_length, len(BINS) + 1)),
            'sbr_mask': np.zeros((ori_res_length, ori_res_length)),
            'interface_mask': np.zeros(ori_res_length),
            'asym_id': raw_feature['asym_id']
        }

    mydicts = []

    restraints0 = restraints.copy()
    
    t0 = time.time()
    megafold_multimer = model_gen.get_model(ckpt_id)
    seq_length = model_gen.seq_length
    print("seq_length: ", seq_length)
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    if seed is not None:
        np.random.seed(seed)

    feat_list = []

    left_thre = (restraints['interface_mask'].sum() + restraints['sbr_mask'].sum()/2)*left_ratio
    left_thre = int(np.ceil(left_thre))
    print(f'At least {left_thre} restraints will be used in the final iteration')
    print(f"iter is {iter}")
    for it in range(iter):
        rank = get_rank()
        step = seq_length // device_num
        # print("it: ", it, "iter: ", iter)
        # pad restraints to fixed length
        
        sbr = Tensor(np_pad(restraints['sbr'], seq_length, axis=(0, 1)).astype(dtype))
        sbr_mask = Tensor(np_pad(restraints['sbr_mask'], seq_length, axis=(0, 1)).astype(dtype))
        interface_mask = Tensor(np_pad(restraints['interface_mask'], seq_length, axis=0).astype(dtype))

        # +++++++++++++++ Fixed ++++++++++++++++++
        sbr = Tensor(sbr[rank*step : (rank + 1)*step, :, :])
        sbr_mask = Tensor(sbr_mask[rank*step : (rank + 1)*step, :])
        interface_mask = Tensor(interface_mask[rank*step : (rank + 1)*step])

        
        # initialize prevs
        prev_pos = Tensor(np.zeros([seq_length, 37, 3]).astype(dtype))
        prev_msa_first_row = Tensor(np.zeros([seq_length, 256]).astype(dtype))
        prev_pair = Tensor(np.zeros([seq_length, seq_length, 128]).astype(dtype))

        prev_prev_pos = prev_pos.asnumpy()
        next_in_prev_pos = prev_pos.asnumpy()
        # # +++++++++++++++ Fixed ++++++++++++++++++
        # prev_pos = Tensor(prev_pos[rank*step : (rank + 1)*step])
        prev_msa_first_row = Tensor(prev_msa_first_row[rank*step : (rank + 1)*step, :])
        prev_pair = Tensor(prev_pair[:, rank*step : (rank + 1)*step, :])

        first_dim = [0, 1, 5, 6, 7, 8, 10, 15, 16]
        print(f"num_recycle is {num_recycle}")
        for i in range(num_recycle):
            f_i = 0 if same_msa_across_recycle else i
            if len(feat_list)-1 < f_i:
                feat_list.append(model_gen.model_process_data(raw_feature))
            
            feat = feat_list[f_i]
            has_exceeded_tolerance, diff = recycle_cond(i, prev_prev_pos, next_in_prev_pos, feat, recycle_early_stop_tolerance)
            
            
            # =============== Fixed ==============
            # feat_i = [Tensor(x) for x in feat]

            # +++++++++++++++ Fixed ++++++++++++++

            feat_i = []
            for index, x in enumerate(feat):
                if index in first_dim:
                    feat_i.append(Tensor(x[rank*step : (rank+1)*step]))
                else:
                    feat_i.append(Tensor(x[:, rank*step : (rank + 1)*step]))
            
            diff = round(diff, 3)
            if not has_exceeded_tolerance:
                print(f"early stop: {i}")
                break
            
            print("--------------------start----------------------")
            # +++++++++++++++ Fixed ++++++++++++++++++
            prev_pos = Tensor(prev_pos[rank*step : (rank + 1)*step])
            # prev_msa_first_row = Tensor(prev_msa_first_row[rank*step : (rank + 1)*step, :])
            # prev_pair = Tensor(prev_pair[:, rank*step : (rank + 1)*step, :])
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits, aligned_error_logits, aligned_error_breaks = megafold_multimer(*feat_i,
                                sbr, sbr_mask, interface_mask,
                                prev_pos,
                                prev_msa_first_row,
                                prev_pair)
            print("--------------------end------------------------")
            
            prev_prev_pos = next_in_prev_pos
            next_in_prev_pos = prev_pos.asnumpy()
            del feat_i
            gc.collect()
            
        
        prev_pos, predicted_lddt_logits = [i.asnumpy()[:ori_res_length] for i in (prev_pos, predicted_lddt_logits)]
        final_atom_positions = prev_pos
        final_atom_mask = feat[16][:ori_res_length]

        confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)
        b_factors = plddt[:, None] * final_atom_mask
        aligned_error_logits = aligned_error_logits.asnumpy()[:ori_res_length, :ori_res_length]
        # ranking_score = compute_ranking_score(aligned_error_logits, aligned_error_breaks.asnumpy()[:ori_res_length, :ori_res_length], raw_feature['asym_id'])
        # ranking_score = round(ranking_score*100, 2)

        unrelaxed_protein = from_prediction(final_atom_positions,
                                            final_atom_mask,
                                            feat[0][:ori_res_length],
                                            feat[1][:ori_res_length],
                                            b_factors,
                                            feat[5][:ori_res_length] - 1,
                                            remove_leading_feature_dimension=False)

        # Write sturcutres into pdb files
        pdb_file = to_pdb(unrelaxed_protein)
        # pdb_path = f'{output_prefix}_score{ranking_score}_iter{it+1}.pdb'
        pdb_path = f'{output_prefix}_iter{it+1}_recycle{num_recycle}_graph_parallel.pdb'
        print(" ===================== pdb_path ==================== ", pdb_path)
        with open(pdb_path, 'w') as f:
            f.write(pdb_file)
            
        # filter restraints
        print(f'Filter Restraints Iteration {it+1} =============================================')
        rm_num, break_num, max_nb_dist, recall, recall_conf, viol_num, max_viol_dist = filter_restraints(restraints, restraints0, unrelaxed_protein, nbdist_ca_thre=nbdist_ca_thre, max_rm_ratio=max_rm_ratio, viol_thre=viol_thre, mask_terminal_residues=mask_terminal_residues)
        # print(f'Filter out {rm_num} restraint(s), confidence {confidence}, 0.8iptm+0.2ptm {ranking_score}')
        rest = int(restraints['interface_mask'].sum() + restraints['sbr_mask'].sum()/2)
                
        
        # record
        assert rm_num >=0, rm_num

        tys = []
        if ((confidence < 50) and (i == num_recycle-1) and (break_num > 20)):
            tys.append('Failed')
        if (recall is not None) and (recall < 0.01):
            tys.append('LowRecall')
        if (rest <= left_thre):
            tys.append('RemoveThre')
        if (rm_num == 0):
            tys.append('Converged')
        if (it == iter-1):
            tys.append('LastIter')

        if len(tys) == 0:
            ty = 'Continue'
        else:
            ty = ','.join(tys)
        
        # mydict = {
        #     'Iter': it+1,
        #     'Conf': round(confidence, 3),
        #     'RankScore': ranking_score,
        #     'Total': rm_num+rest,
        #     'Remove': rm_num,
        #     'Rest': rest,
        #     'MaxNbDist': max_nb_dist,
        #     'BreakNum': break_num,
        #     'Recall': None if recall is None else round(recall, 2),
        #     'RecallByConf': None if recall_conf is None else round(recall_conf, 3),
        #     'Recycle_num': i+1,
        #     'Diff': round(diff, 3),
        #     'ViolNum': int(viol_num),
        #     'MaxViolDist': None if max_viol_dist is None else round(max_viol_dist, 2),
        #     'Time': round(time.time()-t0, 2),
        #     'Type': ty
        # }

        # mydicts.append(mydict)
        # t0 = time.time()
        if len(tys)>0:
            print('Stop iteration:', ty, flush=True)
            break
    # df = pd.DataFrame(mydicts)
    # return df
    return

def infer_batch(model_gen: ModelGenerator,data_gen: DataGenerator, sn: SplitNamelist, pdb_ids, ckpt_ids, res_dir, num_seed=5, 
                baseline=False, nbdist_ca_thre=5.0, viol_thre=5.0, mask_terminal_residues=2, iter=5, 
                num_recycle=20, recycle_early_stop_tolerance=0.5,
                check_tsv_exist=True, quick=False):
    
    os.makedirs(res_dir, exist_ok=True)

    ori_pdb_id_num = len(pdb_ids)
    pdb_ids = [i for i in pdb_ids if data_gen.get_len(i)<=model_gen.seq_length]
    ckpt_ids = [i for i in ckpt_ids if os.path.isfile(model_gen.get_ckpt(i))]
    ckpt_ids.sort()
    print(f'Total pdbs: {len(pdb_ids)}, with {ori_pdb_id_num - len(pdb_ids)} pdb_ids removed because of length exceeding {model_gen.seq_length}')
    print(f'Total ckpts: {len(ckpt_ids)}, {ckpt_ids}')

    print("res_dir", res_dir)
    if check_tsv_exist:
        all_cases = [(ckpt_id, pdb_id) for ckpt_id in ckpt_ids for pdb_id in pdb_ids if len(glob.glob(f'{res_dir}/ckpt_{ckpt_id}_{pdb_id}*_info.tsv'))<num_seed]
    else:
        all_cases = [(ckpt_id, pdb_id) for ckpt_id in ckpt_ids for pdb_id in pdb_ids]

    print(f'Total cases left: {len(all_cases)}')
    
    all_cases = sn.split_namelist(namelist=all_cases)
    
    for ckpt_id, pdb_id in all_cases:

        for i in range(num_seed):

            seed = hash((14, i))%100000 
            # On my environment the first ten seeds are:
            # 32981, 15506, 67931, 50456, 63081,
            # 45606, 98031, 80556, 93181, 75706
            
            output_prefix=f'{res_dir}/ckpt_{ckpt_id}_{pdb_id}_seed{seed}'
            infofile = f'{output_prefix}_info.tsv'
            if os.path.isfile(infofile):
                print(f'{infofile} exists, skip!')
                continue
            t1 = time.time()     
            print(f'[{datetime.datetime.now()}] Start infer ckpt {ckpt_id}, {pdb_id} for seed {seed} ...')
            
            # preprocess features
            raw_feature, restraints = data_gen.get_data(pdb_id)
            if 'asym_id' not in restraints:
                restraints['asym_id'] = raw_feature['asym_id']
            if raw_feature['aatype'].shape[0] > model_gen.seq_length:
                print(f'length out of range {pdb_id}: sequence length {raw_feature["aatype"].shape[0]} > {model_gen.seq_length}')
                continue
            
            t2 = time.time()
            if quick:
                df = grasp_infer_quick(model_gen, ckpt_id, raw_feature, restraints, output_prefix, iter=iter, nbdist_ca_thre=nbdist_ca_thre, viol_thre=viol_thre, mask_terminal_residues=mask_terminal_residues, seed=seed, num_recycle=num_recycle,
                                       recycle_early_stop_tolerance=recycle_early_stop_tolerance)
            else:
                df = grasp_infer(model_gen, ckpt_id, raw_feature, restraints, output_prefix, iter=iter, nbdist_ca_thre=nbdist_ca_thre, viol_thre=viol_thre, mask_terminal_residues=mask_terminal_residues, baseline=baseline, seed=seed,
                                num_recycle=num_recycle, recycle_early_stop_tolerance=recycle_early_stop_tolerance)
            df.to_csv(infofile, sep='\t', index=False)
            t3 = time.time()
            timings = f"[{datetime.datetime.now()}] ckpt step_{ckpt_id} prot_name {pdb_id} seed {seed}, pre_process_time {round(t2 - t1, 2)}, predict time {round(t3 - t2, 2)} , all_time {round(t3 - t1, 2)}"
            print(df.to_string())
            print(timings)

def infer_config(rotate_split, outdir, key=None):
    # context.set_context(mode=context.PYNATIVE,
    #                     device_target="Ascend",
    #                     mempool_block_size="31GB",
    #                     max_call_depth=6000)
    
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    rank_id = int(os.getenv('RANK_ID', '0'))
    device_id = int(os.getenv("DEVICE_ID", '0'))
    rank_size = int(os.getenv('RANK_SIZE', '1'))

    print('{}, rank id: {}, device id: {}, device num: {}, start to run...'.format(
        datetime.datetime.now(), rank_id, device_id, rank_size), flush=True)
    sn = SplitNamelist(rank_id, rank_size, outdir, rotate_split=rotate_split, key=key)
    return sn