#!/usr/bin/env python3
"""
BETE-NET FPD模型推理脚本 - 支持权重加载和可视化
================================================

更新功能：
1. 支持加载预训练权重 (--weight_path参数)
2. 生成与原版相同的散点图
3. 计算并显示MAE、RMSE、R²指标
4. 保存详细结果和汇总报告
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ase.io

# MindSpore设置
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)

# 导入路径设置
sys.path.append('./notebooks_ms')
from setup_paths import setup_paths
setup_paths()

# 导入项目模块
from sharker.data import Batch
from mindspore import nn, ops, Tensor
from notebooks_ms import utils_data_ms as data_utils
from notebooks_ms import utils_model_ms as model_utils

def cal_mae_rmse_r2(dft_val, pred):
    """计算MAE, RMSE, R²指标，处理NaN值"""
    # 过滤掉NaN值
    valid_mask = ~(np.isnan(dft_val) | np.isnan(pred))
    dft_clean = dft_val[valid_mask]
    pred_clean = pred[valid_mask]
    
    if len(dft_clean) == 0:
        return np.nan, np.nan, np.nan
    
    mae = mean_absolute_error(dft_clean, pred_clean)
    rmse = mean_squared_error(dft_clean, pred_clean,) ** 0.5
    r2 = r2_score(dft_clean, pred_clean)
    return mae, rmse, r2

def add_metrics(title, mae, r2, ax, rmse, unit='', test=True, fontsize=10.5):
    """在图上添加评估指标文本，处理NaN值"""
    # 处理NaN值的显示
    mae_str = f'{mae:.3f}' if not np.isnan(mae) else 'NaN'
    rmse_str = f'{rmse:.3f}' if not np.isnan(rmse) else 'NaN'
    r2_str = f'{r2:.3f}' if not np.isnan(r2) else 'NaN'
    
    if test:
        text = f'{title}\nMAE = {mae_str} {unit}\nRMSE = {rmse_str} {unit}\nR² = {r2_str}'
    else:
        text = f'{title}\nMAE = {mae_str} {unit}\nR² = {r2_str}'
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BETE-NET FPD Model Inference')
    parser.add_argument('--weight_path', type=str, default=None,
                       help='Path to pretrained model weights (.ckpt file)')
    parser.add_argument('--output_dir', type=str, default='fpd_inference_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🚀 BETE-NET FPD模型推理")
    print("=" * 50)
    print(f"权重文件: {args.weight_path if args.weight_path else '随机权重'}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("📂 Loading database...")
    df = pd.read_json('database.json')
    df.dropna(inplace=True)
    
    print("📁 Loading structures...")
    structures = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading structures"):
        try:
            structure = ase.io.read(f'structures/{index}.cif')
            structures.append(structure)
        except Exception as e:
            structures.append(None)
    
    df['structure'] = structures
    df = df.dropna(subset=['structure'])
    print(f"✅ 成功加载 {len(df)} 个结构")
    
    # 处理数据 - FPD配置
    print("⚙️ 处理FPD模型数据...")
    df['target'] = df.apply(data_utils.get_target, axis=1)
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    
    # FPD配置参数
    r_max = 4
    embed_ph_dos = True   # FPD使用精细PhDOS
    embed_e_dos = False
    fine = True           # FPD使用精细PhDOS
    
    print(f"📈 构建图数据 (r_max={r_max}, embed_ph_dos={embed_ph_dos}, fine={fine})...")
    tqdm.pandas()
    df['data'] = df.progress_apply(
        data_utils.build_data, 
        embed_ph_dos=embed_ph_dos,
        embed_e_dos=embed_e_dos,
        fine=fine, 
        r_max=r_max, 
        axis=1
    )
    
    # 获取数据维度
    sample_data = df.iloc[0]['data']
    out_dim = len(df.iloc[0]['target'])
    in_dim = sample_data.x.shape[1]
    em_dim = 64    
    print(f"📊 数据维度:")
    print(f"   - 输入特征: {in_dim}")
    print(f"   - 输出目标: {out_dim}")
    print(f"   - 总样本数: {len(df)}")
    
    # 使用原版论文的数据划分 (重要!)
    print("📋 使用原版论文的数据划分...")
    train_df, test_df, _ = data_utils.get_original_data_split(df)

    def create_batches(df, batch_size, shuffle=True):
        """Create batches from dataframe using Sharker's Batch.from_data_list"""
        indices = df.index.tolist()
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            if len(batch_indices) == 1:
                # Single sample - no batching needed
                data = df.loc[batch_indices[0], 'data']
                target = ms.Tensor([df.loc[batch_indices[0], 'target']], dtype=ms.float32)
                batches.append((data, target))
            else:
                # Multiple samples - use Sharker batching
                data_list = [df.loc[idx, 'data'] for idx in batch_indices]
                targets = [df.loc[idx, 'target'] for idx in batch_indices]
                
                batch_data = Batch.from_data_list(data_list)
                batch_targets = ms.Tensor(targets, dtype=ms.float32)
                batches.append((batch_data, batch_targets))
        
        return batches

    test_batch = create_batches(test_df, batch_size=128, shuffle=False)
    
    # 从训练集中划分验证集 (用于模型创建时的参数估计)
    val_split_idx = int(len(train_df) * 0.8)
    val_df = train_df.iloc[val_split_idx:].copy()
    train_subset_df = train_df.iloc[:val_split_idx].copy()
    
    # 创建FPD模型
    print("🏗️ 创建FPD模型...")
    model_params = {
        'in_dim': 118+51,                           
        'em_dim': em_dim,
        'irreps_in': f'{em_dim}x0e',
        'irreps_out': f'{out_dim}x0e',
        'irreps_node_attr': f'{em_dim}x0e',
        'layers': 2,
        'mul': 32,
        'lmax': 1,
        'max_radius': r_max,
        'number_of_basis': 10,
        'radial_layers': 1,
        'radial_neurons': 128,
        'num_neighbors': data_utils.get_neighbors(train_df, train_df.index).mean(),
        'num_nodes': 8.0,
        'reduce_output': True,
        'dropout': False
    }
    
    model = model_utils.PeriodicNetwork(**{k: v for k, v in model_params.items() if k not in ['input_dim', 'output_dim']})
    param_count = sum(p.size for p in model.get_parameters())

    print(f"✅ FPD模型创建完成，参数量: {param_count:,}")
    
    # 加载权重
    if args.weight_path and os.path.exists(args.weight_path):
        print(f"📥 加载权重文件: {args.weight_path}")
        try:
            ms.load_checkpoint(args.weight_path, model)
            print("✅ 权重加载成功")
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            print("⚠️ 继续使用随机权重...")
    else:
        if args.weight_path:
            print(f"⚠️ 权重文件不存在: {args.weight_path}")
        print("⚠️ 使用随机权重 (模型未训练)")
    
    # 推理
    print("🧪 在测试集上运行推理...")
    model.set_train(False)
    
    predictions = []
    targets = []
    
    num_fold = 1
    start = 0

    folds =range(start, start + num_fold)
    for k in tqdm(folds):
        test_batch = create_batches(test_df, batch_size=128, shuffle=False)
        name = f"best_fpd_model_ms_{k}.ckpt"
        run_name = f'./fpd/{name}'
        ms.load_checkpoint(run_name, model)
        print("✅ 权重加载成功")
        # for idx in tqdm(test_df, desc="推理中"):
        #     data = test_df.loc[idx, 'data']
        #     target = test_df.loc[idx, 'target']
        for data, target in tqdm(test_batch, desc="推理中"):
            pred = model(data)
            pred_np = pred.asnumpy()#.flatten()
            
            predictions.append(pred_np)
            targets.append(target)
    
    # 转换为numpy数组
    predictions = np.concatenate(predictions)
    targets = ms.ops.concat(targets)
    # predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 从a2F谱计算物理量 (正确的方法!)
    print(f"\n🧮 从a2F谱计算物理量...")

    # 计算target的物理量
    target_properties = []
    for i, target_spectrum in enumerate(targets):
        lamb, w_log, w_2 = data_utils.compute_physical_properties(target_spectrum)
        target_properties.append([lamb, w_log, w_2])
        if i < 3:  # 显示前几个样本的计算结果
            print(f"Target {i+1}: λ={lamb:.4f}, ω_log={w_log:.1f}K, ω_2={w_2:.1f}K")

    # 计算prediction的物理量
    pred_properties = []
    for i, pred_spectrum in enumerate(predictions):
        lamb, w_log, w_2 = data_utils.compute_physical_properties(pred_spectrum)
        pred_properties.append([lamb, w_log, w_2])
        if i < 3:  # 显示前几个样本的计算结果
            print(f"Pred {i+1}:   λ={lamb:.4f}, ω_log={w_log:.1f}K, ω_2={w_2:.1f}K")

    target_properties = np.array(target_properties)
    pred_properties = np.array(pred_properties)

    # 分解预测结果
    target_names = ['lamb', 'wlog', 'w2']

    # 添加预测结果到DataFrame
    test_df = test_df.copy()
    for i, prop in enumerate(target_names):
        # test_df[f'{prop}_target'] = target_properties[:, i]
        # test_df[f'{prop}_pred'] = pred_properties[:, i]
        test_df[f'{prop}_target'] = target_properties[:, i].reshape((-1, num_fold)).mean(axis=1)
        test_df[f'{prop}_pred'] = pred_properties[:, i].reshape((-1, num_fold)).mean(axis=1)

    # 计算整体指标
    print(f"\n📊 FPD模型测试结果:")

    # 使用物理量计算整体指标
    all_targets = target_properties.flatten()
    all_preds = pred_properties.flatten()

    # 过滤掉NaN值
    valid_mask = ~(np.isnan(all_targets) | np.isnan(all_preds))
    all_targets_clean = all_targets[valid_mask]
    all_preds_clean = all_preds[valid_mask]

    if len(all_targets_clean) > 0:
        overall_mae = mean_absolute_error(all_targets_clean, all_preds_clean)
        overall_rmse = mean_squared_error(all_targets_clean, all_preds_clean,) ** 0.5
        overall_r2 = r2_score(all_targets_clean, all_preds_clean)
    else:
        overall_mae = overall_rmse = overall_r2 = np.nan

    print(f"   - 整体 MAE: {overall_mae:.6f}")
    print(f"   - 整体 RMSE: {overall_rmse:.6f}")
    print(f"   - 整体 R²: {overall_r2:.6f}")
    print(f"   - 有效样本数: {len(all_targets_clean)}/{len(all_targets)}")
    
    # 生成可视化图像 (与原版相同)
    print("📈 生成可视化图像...")
    
    # 设置matplotlib参数
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = 'DejaVu Sans'  # 使用系统默认字体
    
    # 创建图像
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.99, top=0.90)
    
    prop = ['lamb', 'wlog', 'w2']
    color = ['C0', 'C5', 'C8']
    units = ['', 'K', 'K']
    titles = [r'$\lambda$', r'$\omega_{\text{log}}$', r'$\omega_{2}$']
    
    # 为每个属性生成散点图
    for i in range(3):
        # 设置坐标轴范围 (与原版完全一致)
        if i == 0:  # lambda
            lim = (0, 2)
            ticks = [0, 0.5, 1.0, 1.5, 2.0]
        elif i == 1:  # wlog
            lim = (0, 550)
            ticks = np.arange(0, 600, 200)
        else:  # w2
            lim = (0, 700)
            ticks = np.arange(0, 800, 200)
        
        # 计算并添加指标
        mae, rmse, r2 = cal_mae_rmse_r2(
            dft_val=test_df[f'{prop[i]}_target'], 
            pred=test_df[f'{prop[i]}_pred']
        )
        
        # 计算有效样本数
        valid_mask = ~(np.isnan(test_df[f'{prop[i]}_target']) | np.isnan(test_df[f'{prop[i]}_pred']))
        n_valid = valid_mask.sum()
        n_total = len(test_df)
        
        # 绘制散点图 (只绘制有效数据点)
        valid_targets = test_df[f'{prop[i]}_target'][valid_mask]
        valid_preds = test_df[f'{prop[i]}_pred'][valid_mask]
        
        if len(valid_targets) > 0:
            axs[i].scatter(valid_targets, valid_preds, marker='.', color=color[i], alpha=0.5)
        
        add_metrics(title=f'Test (n={n_valid}/{n_total})', mae=mae, r2=r2, ax=axs[i], 
                   rmse=rmse, unit=units[i], test=True, fontsize=10.5)
        
        # 绘制对角线
        axs[i].plot(lim, lim, 'k--', zorder=0)
        
        # 设置坐标轴
        axs[i].set_xlim(lim)
        axs[i].set_ylim(lim)
        axs[i].set_xticks(ticks)
        axs[i].set_yticks(ticks)
        axs[i].set_aspect('equal')
        
        # 设置标签 (与原版一致)
        if i == 0:
            axs[i].set_ylabel(f'Predicted {titles[i]}')
        else:
            axs[i].set_ylabel(f'Predicted {titles[i]} (K)')
        
        if i == 2:
            axs[i].set_xlabel('Target')
        
        # 打印每个属性的指标
        print(f"   - {prop[i].upper()} - MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
    
    # 保存图像
    plot_path = os.path.join(args.output_dir, 'FPD_prediction_results.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 图像保存至: {plot_path}")
    
    # 保存详细结果
    results_path = os.path.join(args.output_dir, 'FPD_detailed_results.csv')
    test_df.to_csv(results_path, index=False)
    print(f"💾 详细结果保存至: {results_path}")
    
    # 保存汇总指标
    summary_path = os.path.join(args.output_dir, 'FPD_summary_metrics.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("BETE-NET FPD Model - Prediction Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Configuration: FPD (Fine PhDOS)\n")
        f.write(f"Weight Path: {args.weight_path if args.weight_path else 'Random weights'}\n")
        f.write(f"Test Samples: {len(test_df)}\n")
        f.write(f"Model Parameters: {param_count:,}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  MAE:  {overall_mae:.6f}\n")
        f.write(f"  RMSE: {overall_rmse:.6f}\n")
        f.write(f"  R2:   {overall_r2:.6f}\n\n")
        
        f.write("Individual Property Metrics:\n")
        for i, prop in enumerate(prop):
            mae, rmse, r2 = cal_mae_rmse_r2(
                dft_val=test_df[f'{prop}_target'], 
                pred=test_df[f'{prop}_pred']
                # dft_val=target_properties, 
                # pred=target_properties
            )
            f.write(f"  {prop.upper()}:\n")
            f.write(f"    MAE:  {mae:.6f} {units[i]}\n")
            f.write(f"    RMSE: {rmse:.6f} {units[i]}\n")
            f.write(f"    R2:   {r2:.6f}\n\n")
    
    print(f"📄 汇总指标保存至: {summary_path}")
    
    plt.show()
    
    print(f"\n✅ FPD模型推理完成!")
    print(f"📁 所有结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 