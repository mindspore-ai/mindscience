# MindSpore & Sharker 版本的数据处理工具

# 设置导入路径
import setup_paths

from sharker.data import Graph as Data
from sharker.loader import Dataloader as DataLoader
from mindchemistry.e3.nn.one_hot import soft_one_hot_linspace
# 其他必要的mindspore和sharker导入

# 这里实现build_data, get_target, get_neighbors等函数，接口与原data.py一致
# 具体实现需参考原data.py和sharker的API

# 示例：
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from scipy.signal import savgol_filter
from ase.neighborlist import neighbor_list
from ase import Atom
import os
import sys
import pandas as pd
import ase.io
from tqdm import tqdm
import matplotlib.pyplot as plt

# MindSpore默认使用float32

# 定义频率范围
Freq_final = np.arange(0.25, 101, 2)
Freq_final_E = np.arange(-50, 50, 1)

# 创建原子类型编码
type_encoding = {}
specie_am = []
for Z in range(1, 119):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)

type_onehot = ms.ops.eye(len(type_encoding), len(type_encoding), ms.float32)
am_onehot = ms.ops.diag(ms.Tensor(specie_am, dtype=ms.float32))

def build_data(entry, r_max=4.0, embed_ph_dos=True, embed_e_dos=True, fine=False):
    """
    使用MindSpore和Sharker构建适合图模型的数据对象
    
    参数:
    - entry: 包含分子或晶体结构信息的输入对象
    - r_max: 邻居列表计算的截断半径
    - embed_ph_dos: 是否嵌入声子态密度数据
    - embed_e_dos: 是否嵌入电子态密度数据
    - fine: 声子态密度的精细度标志
    
    返回:
    - sharker.data.Data: 包含节点特征、边索引等属性的Sharker数据对象
    """
    symbols = list(entry.structure.symbols).copy()
    positions = ms.Tensor(entry.structure.positions.copy(), dtype=ms.float32)
    lattice = ms.Tensor(entry.structure.cell.array.copy(), dtype=ms.float32).expand_dims(0)

    # 计算边源和目标索引
    edge_src, edge_dst, edge_shift = neighbor_list(
        "ijS", a=entry.structure, cutoff=r_max, self_interaction=True
    )

    # 计算相对距离和周期性边界位移
    edge_batch = ms.ops.zeros(positions.shape[0], dtype=ms.int64)[
        ms.Tensor(edge_src)
    ]
    
    # 使用矩阵乘法替代einsum
    edge_shift_tensor = ms.Tensor(edge_shift, dtype=ms.float32)
    lattice_batch = lattice[edge_batch]  # shape: [n_edges, 3, 3]
    
    # 计算 edge_shift @ lattice 的结果
    # edge_shift_expanded = edge_shift_tensor.expand_dims(-1)  # [n_edges, 3, 1]
    # lattice_contribution0 = ms.ops.matmul(lattice_batch, edge_shift_expanded).squeeze(-1)  # [n_edges, 3]
    lattice_contribution = ms.mint.einsum('nj,nij->nj', edge_shift_tensor, lattice_batch)
    
    edge_vec = (
        positions[ms.Tensor(edge_dst)]
        - positions[ms.Tensor(edge_src)]
        + lattice_contribution
    )

    # 计算边长度
    edge_len = np.around(edge_vec.norm(dim=1).asnumpy(), decimals=2)

    # 构建节点特征
    x = am_onehot[[type_encoding[specie] for specie in symbols]].astype(ms.float32)
    z = type_onehot[[type_encoding[specie] for specie in symbols]].astype(ms.float32)

    if embed_ph_dos and embed_e_dos:
        p_ph_dos = process_phdos(entry, fine=fine)
        p_e_dos = process_edos(entry, fine=fine)

        x = ms.ops.concat((x, ms.ops.ones_like(p_ph_dos), ms.ops.ones_like(p_e_dos)), 1)
        z = ms.ops.concat((z, p_ph_dos, p_e_dos), 1)

    elif embed_ph_dos:
        p_ph_dos = process_phdos(entry, fine=fine)
        x = ms.ops.concat((x, ms.ops.ones_like(p_ph_dos)), 1)
        z = ms.ops.concat((z, p_ph_dos), 1)

    elif embed_e_dos:
        p_e_dos = process_edos(entry, fine=fine)
        x = ms.ops.concat((x, ms.ops.ones_like(p_e_dos)), 1)
        z = ms.ops.concat((z, p_e_dos), 1)

    # 创建Sharker数据对象
    data = Data(
        x=x,
        edge_index=ms.ops.stack(
            [ms.Tensor(edge_src, dtype=ms.int64), ms.Tensor(edge_dst, dtype=ms.int64)], axis=0
        ),
        edge_attr=None,  # 可以后续添加边属性
        y=ms.Tensor(np.asarray(entry.target), dtype=ms.float32).expand_dims(0),
        crd=positions,  # 使用crd参数存储坐标
        # 添加自定义属性
        pos=positions,
        lattice=lattice,
        symbol=symbols,
        z=z,
        edge_shift=ms.Tensor(edge_shift, dtype=ms.float32),
        edge_vec=edge_vec,
        edge_len=edge_len,
        target=ms.Tensor(np.asarray(entry.target), dtype=ms.float32).expand_dims(0),
    )
    return data

def get_target(df):
    """
    处理目标数据
    """
    x = df.Freq_meV
    y = df.a2F
    xl = np.arange(0.25, 101, 0.1)
    y = np.interp(xl, x, y)
    Y = savgol_filter(y, 101, 3, mode="interp")
    Y = np.interp(Freq_final, xl, Y)
    Y = np.asarray([y if y > 0.0 else 0.0 for y in Y])
    return Y

def get_neighbors(df, idx):
    """
    获取每个节点的邻居数量
    """
    n = []
    for entry in df.itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)

def get_phdos(df):
    """
    获取声子态密度
    """
    x = df.PhFreq_meV
    y = df.Tot_PhDOS
    xl = np.arange(0.25, 101, 0.1)
    y = np.interp(xl, x, y)
    Y = savgol_filter(y, 101, 3, mode="interp")
    Y = np.interp(Freq_final, xl, Y)
    return np.asarray([y if y > 0.0 else 0.0 for y in Y])

def process_phdos(entry, fine=False):
    """
    处理声子态密度数据
    """
    Y_proc = []
    if fine:
        x = entry.PhFreq_meV_dense
        ys = entry.Site_Proj_PhDOS_dense
    else:
        x = entry.Ph_2x2x2_interpolated_Freq_meV
        ys = entry.Ph_2x2x2_interpolated_Site_Proj_DOS

    for y in ys:
        xl = np.arange(0.25, 101, 0.1)
        y = np.interp(xl, x, y)
        # 调整窗口长度以适应数据大小
        window_length = min(101, len(y))
        if window_length % 2 == 0:
            window_length -= 1  # 确保窗口长度为奇数
        window_length = max(3, window_length)  # 至少为3
        Y = savgol_filter(y, 101, 3, mode="interp")
        Y = np.interp(Freq_final, xl, Y)
        Y = [y if y > 0.0 else 0.0 for y in Y]
        Y_proc.append(Y.copy())
    return ms.Tensor(Y_proc, dtype=ms.float32)

def process_edos(entry, fine=False):
    """
    处理电子态密度数据
    """
    ys = entry.Site_proj_eDOS
    x = entry.Site_proj_eDOS_eng_meV
    Y_proc = []

    for y in ys:
        # 调整窗口长度以适应数据大小
        window_length = min(101, len(y))
        if window_length % 2 == 0:
            window_length -= 1  # 确保窗口长度为奇数
        window_length = max(3, window_length)  # 至少为3
        Y = savgol_filter(y, 101, 3, mode="interp")
        Y = np.interp(Freq_final_E, x, Y)
        Y = [y if y > 0.0 else 0.0 for y in Y]
        Y_proc.append(Y.copy())
    return ms.Tensor(Y_proc, dtype=ms.float32)

def load_data_splits(idx=None):
    """
    加载原版论文的数据划分索引
    返回训练集和测试集的索引
    """
    if isinstance(idx, int):
        idx_train = np.loadtxt(f'indices/idx_train_V2_{idx}.txt').astype(int)
        idx_valid = np.loadtxt(f'indices/idx_valid_V2_{idx}.txt').astype(int)
        idx_test = np.loadtxt(f'indices/idx_test_full.txt').astype(int)

    else:
        # 加载测试集索引 (174个样本)
        idx_test = np.loadtxt('indices/idx_test_full.txt').astype(int)
        # 加载训练集索引 (652个样本) 
        idx_train = np.loadtxt('indices/idx_train_full.txt').astype(int)
        idx_valid = None
    
    return idx_train, idx_test, idx_valid

def get_original_data_split(df, idx=None):
    """
    根据原版论文的索引划分数据
    确保训练和测试数据完全分离
    
    注意：由于数据库版本差异，某些索引可能不存在，
    我们只使用存在的样本，但保持原有的划分比例和策略
    """
    idx_train, idx_test, idx_valid = load_data_splits(idx)
    
    # 过滤存在于DataFrame中的索引
    available_train_indices = [idx for idx in idx_train if idx in df.index]
    available_test_indices = [idx for idx in idx_test if idx in df.index]

    # 检查缺失的样本
    missing_train = len(idx_train) - len(available_train_indices)
    missing_test = len(idx_test) - len(available_test_indices)
    
    train_df = df.loc[available_train_indices].copy()
    test_df = df.loc[available_test_indices].copy()

    if idx_valid is not None:
        available_val_indices = [idx for idx in idx_valid if idx in df.index]
        missing_val = len(idx_valid) - len(available_val_indices)
        val_df = df.loc[available_val_indices].copy()
        print(f"   - 验证集: {len(val_df)} 样本 (原版: {len(idx_valid)}, 缺失: {missing_val})")
    else:
        val_df = None

    
    print(f"📊 原版数据划分 (已处理缺失样本):")
    print(f"   - 训练集: {len(train_df)} 样本 (原版: {len(idx_train)}, 缺失: {missing_train})")
    print(f"   - 测试集: {len(test_df)} 样本 (原版: {len(idx_test)}, 缺失: {missing_test})")
    if idx_valid is None:
        print(f"   - 数据无重叠(训练，测试): {len(set(train_df.index) & set(test_df.index)) == 0}")
    else:
        print(f"   - 数据无重叠(训练，验证，测试): {len(set(train_df.index) & set(test_df.index)) + (len(set(train_df.index) & set(val_df.index))) == 0}")
    
    
    if missing_train > 0 or missing_test > 0:
        print(f"   ⚠️  数据库版本差异: 总计缺失 {missing_train + missing_test} 个样本")
        print(f"   ✅ 使用可用样本，保持原有划分策略")
    
    return train_df, test_df, val_df

# 添加物理量计算函数
def cal_lamb(freq_w, alpha_F):
    """
    从a2F谱计算λ (电子-声子耦合常数)
    """
    lambdaF = 0
    try:
        for i in range(1, len(freq_w)):
            dw = freq_w[i] - freq_w[i-1]
            w = freq_w[i]
            alpha_F_w = alpha_F[i]
            lambdaF = lambdaF + ((alpha_F_w/w)*dw)
        return 2*lambdaF
    except:
        return np.nan

def cal_w_log(freq_w, alpha_F, lamb):
    """
    从a2F谱计算ω_log (对数平均频率)
    """
    w_logF = 0
    try:
        for i in range(1, len(freq_w)):
            dw = freq_w[i] - freq_w[i-1]
            w_logF = w_logF + (alpha_F[i]*np.log(freq_w[i])*dw/freq_w[i])
        return np.exp(2*w_logF/lamb)
    except: 
        return np.nan

def cal_w_sq(freq_w, alpha_F, lamb):
    """
    从a2F谱计算ω_2 (二阶矩频率)
    """
    w_sqF = 0
    try:
        for i in range(1, len(freq_w)):
            dw = freq_w[i] - freq_w[i-1]
            w_sqF = w_sqF + (alpha_F[i]*freq_w[i]*dw)
        return (2*w_sqF/lamb)**.5
    except:
        return np.nan

def compute_physical_properties(a2F_spectrum):
    """
    从a2F谱计算物理量λ, ω_log, ω_2
    
    参数:
    - a2F_spectrum: a2F谱 (51维向量)
    
    返回:
    - lambda, w_log (K), w_2 (K)
    """
    frequency = Freq_final  # 频率网格
    
    # 计算λ
    lamb = cal_lamb(frequency, a2F_spectrum)
    
    # 计算ω_log (转换为Kelvin)
    w_log = cal_w_log(frequency, a2F_spectrum, lamb) / 0.08617 if lamb > 0 else np.nan
    
    # 计算ω_2 (转换为Kelvin) 
    w_2 = cal_w_sq(frequency, a2F_spectrum, lamb) / 0.08617 if lamb > 0 else np.nan
    
    return lamb, w_log, w_2 