import numpy as np
import traceback
# from scipy.stats import lognorm
BINS = np.arange(4, 33, 1)

def normalize_number_in_bins(dist, bins):
    upper_edges = np.array(list(bins) + [np.inf])
    lower_edges = np.array([0] + list(bins))
    num_in_bins = ((dist.flatten()<=upper_edges[..., None])*(dist.flatten()>lower_edges[..., None])).sum(-1)
    dist_which_bin = (dist[..., None]>bins).sum(-1)
    p_norm = (1/(num_in_bins+1e-8))[dist_which_bin]
    return p_norm


# def sample_dist_1(dist, mask, num, thre, fdr, bins=BINS):
#     d = dist * mask
#     idx = np.where(d>1) # remove masked sites
#     d = d[idx]
#     true_num = (d<=thre).sum()
#     total_max = np.ceil(true_num/(1-fdr))
#     num = int(min(total_max, num, len(d)))
#     p_norm = normalize_number_in_bins(d, bins)
#     p = fdr * (d>thre)/(bins>=thre).sum() + (1-fdr)*(d<=thre)/(bins<=thre).sum()
#     p *= p_norm
#     p /= p.sum()
#     chosen_idx = np.random.choice(np.arange(p.size), size=num, p=p.ravel(), replace=False)

#     idx = np.transpose(idx)[chosen_idx]
#     idx = (idx[:,0], idx[:, 1])

#     return idx

def sample_dist(dist, mask, num, thre, fdr, bins=BINS):
    d = dist * mask
    idx = np.where(d>1) # remove masked sites
    d = d[idx]
    
    true_num = (d<=thre).sum()
    false_num = d.size - true_num

    true_num = int(min(true_num, round(num*(1-fdr))))
    false_num = int(min(false_num, round(num*fdr)))
    
    p_norm = normalize_number_in_bins(d, bins)

    chosen_idx = []
    # sample true
    if true_num>0:
        p = np.ones(d.shape) * (d<=thre)
        p *= p_norm
        p /= p.sum()
        idx_temp = np.random.choice(np.arange(p.size), size=true_num, p=p.ravel(), replace=False)
        chosen_idx.extend(idx_temp)
    
    # sample false
    if false_num>0:
        p = np.ones(d.shape) * (d>thre)
        p *= p_norm
        p /= p.sum()
        idx_temp = np.random.choice(np.arange(p.size), size=false_num, p=p.ravel(), replace=False)
        chosen_idx.extend(idx_temp)
    
    idx = np.transpose(idx)[chosen_idx]
    idx = (idx[:,0], idx[:, 1])

    return idx   

def get_crop_index(asym_id, residue_index, chain_index):
    unique_chain_index = np.unique(chain_index)
    unique_chain_index.sort()
    crop_index = []
    seq_len = 0
    for i in unique_chain_index:
        # print(i, seq_len)
        res_idx_i = residue_index[asym_id == (i+1)]
        res_idx_i += seq_len
        crop_index.extend(res_idx_i)
        seq_len += (chain_index == i).sum()
        # print(crop_index)
    return crop_index

def get_sample_num(start, reduce, end, k):
    probs = np.concatenate((np.zeros(start), np.ones(reduce-start), np.exp(-(np.arange(end-reduce)/k))),axis=0)
    probs = normalize(probs)
    num = np.random.choice(np.arange(end), p=probs)
    return num

def normalize(probs):
    probs[probs<0] = 0
    probs /= probs.sum()
    return probs

def generate_mask(dist, pseudo_beta_mask, asym_id, residue_index):
    ''' add mask info'''
    remote_residue_threshold = 6
    greater_residue_index_diff = (np.abs(residue_index[:, None] - residue_index[None]) > remote_residue_threshold).astype(np.float32)
    pseudo_beta_mask_2d =  (pseudo_beta_mask[:,None] * pseudo_beta_mask[None]) * (dist > 0.01)
    upper_mask = np.triu(np.ones_like(pseudo_beta_mask_2d), 1)
    mask_intra = (asym_id[:, None] == asym_id[None]).astype(np.float32)
    mask_inter = (1.0 - mask_intra) * pseudo_beta_mask_2d
    mask_interface = (dist < 8.0) * mask_inter
    interface_dist = (dist+(1-mask_inter)*1e8).min(-1)
    mask_interface = mask_interface.any(-1)
    mask_inter *= upper_mask
    mask_intra = mask_intra * greater_residue_index_diff * pseudo_beta_mask_2d * upper_mask
    return mask_inter, mask_intra, mask_interface, interface_dist

def sample_interface_by_asym_id(asym_id, mask, num):
    num = min(num, mask.sum())
    mask_interface = np.zeros_like(mask)
    if num > 0:
        asym_id_same = asym_id[:, None] == asym_id[None]
        num_interface_each_chain = (asym_id_same * mask[None]).sum(-1)
        probs = mask / (num_interface_each_chain + 1e-8)
        probs = normalize(probs)
        idx = np.random.choice(np.arange(len(asym_id)), num, replace=False, p=probs)
        mask_interface[idx] = 1.0
    return mask_interface

def single_bin(dist, fdr, bins):
    r = np.eye(len(bins)+1)[(dist[..., None] > bins).sum(-1).astype(np.int32)]
    r = r*(1-fdr) + (1-r)*fdr/((1-r).sum(-1, keepdims=True))
    return r

def uniform_cutoff(thre, fdr, bins):
    r = np.ones((len(bins)+1))
    num_lower = (thre >= bins).sum()
    r[:num_lower] = r[:num_lower]/r[:num_lower].sum() * (1-fdr)
    r[num_lower:] = r[num_lower:]/r[num_lower:].sum() * fdr
    return r
    
def print_rpr(dist, mask, sbr, thre):
    if mask.sum()>0:
        d = dist[mask>0.5]
        print(f'Total:{d.size}, FDR: {(d>thre).sum()/d.size}, Thre: {thre}, Dist: {d}')
        # if not (sbr[mask>0.5].sum(-1)==1).all():
        #     print(sbr[mask>0.5].sum(-1))
        # assert (sbr[mask>0.5].sum(-1)==1).all()
    else:
        print('No restraint')


def generate_interface_and_restraints(d, num_inter=0, num_intra=0, num_interface=0, thre=8, fdr=0.05,
                                      mixed_precision=True, training=True,
                                      seed=None, fix_afm=True, bins = BINS):
    if seed is not None:
        np.random.seed(seed)

    # assert 'pseudo_beta' in d
    # assert 'pseudo_beta_mask' in d
    # assert 'asym_id' in d
    # assert 'residue_index' in d
    # assert 'chain_index' in d
    
    if training:
        asym_id = d['asym_id'][0]    
        residue_index = d['residue_index'][0]
        chain_index = d['chain_index']
        crop_index = get_crop_index(asym_id, residue_index, chain_index)
        seqlen = len(asym_id) #384

        # check crop index
        aatype_pdb = d['aatype_per_chain'][crop_index]
        aatype_pdb = np.pad(aatype_pdb, ((0, seqlen - aatype_pdb.shape[0]),))
        aatype = d['aatype'][0]
        delta = (np.abs(aatype - aatype_pdb) * (aatype_pdb < 20)).sum()
        if delta > 0:
            print('error! crop index is wrong!')
            print(aatype)
            print(aatype_pdb)
            raise ValueError
        
        pseudo_beta = d["pseudo_beta"][crop_index]
        pseudo_beta_mask = d['pseudo_beta_mask'][crop_index]
        # pad to fixed length
        pseudo_beta = np.pad(pseudo_beta, ((0, seqlen - pseudo_beta.shape[0]), (0, 0)))
        pseudo_beta_mask = np.pad(pseudo_beta_mask, ((0, seqlen - pseudo_beta_mask.shape[0]),))
        dist = np.sqrt((np.square(pseudo_beta[None]-pseudo_beta[: ,None])).sum(-1) + 1e-8)

    else:
        asym_id = d['asym_id']
        seqlen = len(asym_id)
        pseudo_beta = d['pseudo_beta'] if 'pseudo_beta' in d else np.zeros((seqlen, 3))
        if 'pseudo_beta_mask' in d:
            pseudo_beta_mask = d['pseudo_beta_mask']
        elif 'mask_2d' in d:
            pseudo_beta_mask = (d['mask_2d'].sum(0) > 0.5).astype(d['mask_2d'].dtype)
        else:
            np.ones_like(asym_id)
        dist = np.sqrt((np.square(pseudo_beta[None]-pseudo_beta[: ,None])).sum(-1) + 1e-8)
        dist = d['dist'] if 'dist' in d else dist
        residue_index = d['residue_index'] if 'residue_index' in d else np.arange(seqlen)
     
    
    sbr = np.zeros((seqlen, seqlen, len(bins) + 1))
    sbr_mask = np.zeros((seqlen, seqlen))
    mask_interface = np.zeros(seqlen)
    try:
        
        if training:

            num_inter = 0
            num_intra = 0
            num_interface = 0

            sample_ratio = 0.5
            if np.random.rand() < sample_ratio:
                num_inter = get_sample_num(start=1, reduce=20, end=40, k=4)

            if np.random.rand() < sample_ratio:
                num_intra = get_sample_num(start=1, reduce=80, end=160, k=16)

            if np.random.rand() < sample_ratio:
                num_interface = get_sample_num(start=1, reduce=40, end=80, k=8)
          
            if fix_afm and num_inter+num_intra+num_interface==0:
                num_inter = get_sample_num(start=1, reduce=20, end=40, k=4)
                num_intra = get_sample_num(start=1, reduce=80, end=160, k=16)
                num_interface = get_sample_num(start=1, reduce=40, end=80, k=8)

            # Only one chain
            if len(np.unique(asym_id)) == 1:
                num_interface = 0
                num_inter = 0

        mask_inter, mask_intra, mask_interface, interface_dist = generate_mask(dist, pseudo_beta_mask, asym_id, residue_index)

        if training:
            single_bin_ratio = 0.5
            if np.random.rand() < single_bin_ratio:
                thre = 30
                r = single_bin(dist, fdr=0.05, bins=bins)
                
            else:
                thre = np.random.randint(low=8, high=31)
                r = uniform_cutoff(thre, fdr=fdr, bins=bins)
                r = np.tile(r, (*dist.shape, 1))

        else:
            r = uniform_cutoff(thre, fdr=fdr, bins=bins)
            r = np.tile(r, (*dist.shape, 1))

        intra_pair = sample_dist(dist, mask_intra, num_intra, thre=thre, fdr=fdr, bins=bins)
        inter_pair = sample_dist(dist, mask_inter, num_inter, thre=thre, fdr=fdr, bins=bins)
        sbr[intra_pair] = r[intra_pair]
        sbr[inter_pair] = r[inter_pair]
        sbr += sbr.swapaxes(0, 1)

        mask_interface  = sample_interface_by_asym_id(asym_id, mask_interface, num_interface)

        dtype = np.float32
        if mixed_precision:
            dtype = np.float16
        sbr = sbr.astype(dtype)
        sbr_mask = (sbr.sum(-1) > 0.5).astype(dtype)

        mask_interface = mask_interface.astype(dtype)
        
        # show info
        print('inter rpr: =======================================')
        print_rpr(dist, mask_inter*sbr_mask, sbr, thre)
        print('intra rpr: =======================================')
        print_rpr(dist, mask_intra*sbr_mask, sbr, thre)
        print('interface: =======================================')
        print(f'Total: {int(mask_interface.sum())}, Dist: {interface_dist[mask_interface>0.5]}')
        
    except Exception as e:
        sbr = np.zeros((seqlen, seqlen, len(bins) + 1))
        sbr_mask = np.zeros((seqlen, seqlen))
        mask_interface = np.zeros(seqlen)
        print('Error in sample restraints:', e)
        traceback.print_exc()

    return sbr, sbr_mask, mask_interface

