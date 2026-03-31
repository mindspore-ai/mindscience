# -*- coding: utf-8 -*-
"""
数据集长度分析脚本

功能:
1. 加载训练、验证和测试数据集。
2. 使用与训练脚本完全相同的模板和分词器进行处理。
3. 计算每条数据包含输入和标签（完整序列）的总token长度。
4. 统计并输出长度的最大值、平均值、中位数和百分位数。
5. 生成长度分布的直方图，以帮助选择合适的`max_length`。

使用方式:
python analyze_data_length.py --config config/forward_1.5b_training.yaml

"""
import os
import yaml
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# 从您的项目中导入必要的模块
# 假设此脚本与您的项目文件夹在同一层级
from mindnlp.transformers import AutoTokenizer
from utils.solvent_dataset_process import parse_solvent_data, SolventPredictionDataset
from model.tokenizer import get_default_tokenizer

def full_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def analyze_dataset_lengths(reactions, qwen_tokenizer, rxn_tokenizer, config):
    """
    分析给定数据集的token长度。
    
    Args:
        reactions (list): 从文本文件解析出的反应列表。
        qwen_tokenizer: Qwen的分词器。
        rxn_tokenizer: Reaction-BERT的分词器。
        config (dict): 包含了任务类型等信息的配置字典。
        
    Returns:
        list: 包含每条数据token长度的列表。
    """
    if not reactions:
        return []

    data_config = config['data']
    
    # 创建数据集实例，关键点：设置一个超大的max_len以防止截断，从而获取真实长度
    dataset = SolventPredictionDataset(
        data=reactions,
        qwen_tokenizer=qwen_tokenizer,
        rxn_tokenizer=rxn_tokenizer,
        use_cls_token=data_config['use_cls_token']
    )
    
    lengths = []
    print(f"正在分析 {len(dataset)} 条数据...")
    # 使用tqdm显示进度条
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        # 'input_ids' 包含了输入和输出的完整token序列,和labels长度相同
        # 因此其长度就是我们需要的总长度
        total_length = len(sample['rxn_input_ids'])
        lengths.append(total_length)
        
    return lengths

def main():
    parser = argparse.ArgumentParser(description="分析化学反应预测数据集的token长度")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径，与训练脚本使用的相同")
    
    args = parser.parse_args()
    
    # 1. 加载配置文件
    config = full_config(args.config)
    print("=== 配置加载成功 ===")
    
    # 2. 初始化分词器 (与训练脚本保持一致)
    print("\n=== 初始化分词器 ===")
    model_config = config['model']
    if model_config.get('use_extended_vocab', False) and os.path.exists(model_config['extended_vocab_path']):
        print(f"从扩展词表加载Qwen分词器: {model_config['extended_vocab_path']}")
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config['extended_vocab_path'])
    else:
        print(f"从预训练模型加载Qwen分词器: {model_config['qwen_model_name']}")
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config['qwen_model_name'])

    rxn_tokenizer = get_default_tokenizer()
    
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    print("分词器初始化完成。")
    
    # 3. 加载数据
    print("\n=== 加载数据文件 ===")
    data_path = config['data']['data_path']
    train_file = os.path.join(data_path, 'train.txt')
    valid_file = os.path.join(data_path, 'valid.txt')
    test_file = os.path.join(data_path, 'test.txt')

    train_reactions = parse_solvent_data(train_file) if os.path.exists(train_file) else []
    val_reactions = parse_solvent_data(valid_file) if os.path.exists(valid_file) else []
    test_reactions = parse_solvent_data(test_file) if os.path.exists(test_file) else []
    
    print(f"训练集样本数: {len(train_reactions)}")
    print(f"验证集样本数: {len(val_reactions)}")
    print(f"测试集样本数: {len(test_reactions)}")

    # 4. 分析各个数据集的长度
    print("\n=== 开始分析token长度 ===")
    test_lengths = analyze_dataset_lengths(test_reactions, qwen_tokenizer, rxn_tokenizer, config)
    
    all_lengths = test_lengths
    
    if not all_lengths:
        print("\n未找到任何数据，无法进行分析。请检查数据路径配置。")
        return

    # 5. 统计和报告结果
    print("\n=== 数据集rxn长度统计结果 ===")
    df = pd.DataFrame(all_lengths, columns=['length'])
    
    max_len = df['length'].max()
    mean_len = df['length'].mean()
    median_len = df['length'].median()
    p90 = df['length'].quantile(0.90)
    p95 = df['length'].quantile(0.95)
    p97 = df['length'].quantile(0.97)
    p99 = df['length'].quantile(0.99)
    
    print(f"最大长度 (Max): {max_len}")
    print(f"平均长度 (Mean): {mean_len:.2f}")
    print(f"中位长度 (Median): {median_len:.2f}")
    print(f"90百分位 (90th Percentile): {p90:.2f}")
    print(f"95百分位 (95th Percentile): {p95:.2f}")
    print(f"97百分位 (90th Percentile): {p97:.2f}")
    print(f"99百分位 (99th Percentile): {p99:.2f}")

    print("\n建议: 您可以根据95或99百分位来设置`max_length`，")
    print("例如，设置为一个略大于99百分位值的、方便计算的整数（如384, 512, 768等）。")
    print("这样可以在覆盖绝大部分数据的同时，最大化训练效率。")

    # 6. 可视化长度分布
    print("\n=== 生成长度分布直方图 ===")
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.histplot(df['length'], bins=50, kde=True)
    
    plt.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.2f}')
    plt.axvline(median_len, color='green', linestyle='-', label=f'Median: {median_len:.2f}')
    plt.axvline(p99, color='purple', linestyle=':', label=f'99th Percentile: {p99:.2f}')
    
    plt.title('Token Length Distribution (Input + Label)')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.legend()
    
    output_filename = "rxn_length_distribution.png"
    plt.savefig(output_filename)
    print(f"分布图已保存为: {output_filename}")
    # plt.show() # 如果在图形界面环境，可以取消这行注释来直接显示图片

if __name__ == "__main__":
    main()