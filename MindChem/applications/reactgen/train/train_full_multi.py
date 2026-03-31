# -*- coding: utf-8 -*-
"""
多卡全量微调脚本 - 化学反应预测
支持MindSpore分布式训练

使用方式:
python train/train_full_multi.py --config config/forward_1.5b_training.yaml

"""
import os
import yaml
import json
import argparse
import numpy as np
import random
from datetime import datetime
from zoneinfo import ZoneInfo
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import set_auto_parallel_context, ParallelMode
from typing import Dict, Any
from swanlab.integration.transformers import SwanLabCallback
from eval.evaluation_metrics import ChemicalReactionEvaluator

from mindnlp.engine import Trainer, TrainingArguments
from mindnlp.engine.callbacks import TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback
from mindnlp.transformers import AutoTokenizer

from model.reactionqwen import MultiModalQwen, MultiModalQwenConfig
from dataset.dataset import DualRepresentationDataset, parse_reactions
from model.tokenizer import get_default_tokenizer

class TrainLossRecorder(TrainerCallback):
    """记录训练过程中的损失"""
    def __init__(self, output_dir, rank_id=0):
        super().__init__()
        self.output_dir = output_dir
        self.rank_id = rank_id
        self.train_logs = []
        self.eval_logs = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        _ = args, kwargs  # 消除未使用参数警告
        if logs:
            if "loss" in logs:
                self.train_logs.append((state.global_step, logs["loss"]))
            if "eval_loss" in logs:
                self.eval_logs.append((state.global_step, logs["eval_loss"]))
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        _ = args, state, control, kwargs  # 消除未使用参数警告
        # 只在rank 0保存训练损失
        if self.rank_id == 0:
            # 保存训练损失
            with open(f"{self.output_dir}/train_loss.txt", "w") as f:
                for step, loss in self.train_logs:
                    f.write(f"{step}\t{loss:.6f}\n")
            
            # 保存验证损失
            with open(f"{self.output_dir}/eval_loss.txt", "w") as f:
                for step, loss in self.eval_logs:
                    f.write(f"{step}\t{loss:.6f}\n")
            
            print(f"Training logs saved to {self.output_dir}/")
        return control
    
class EnsureCheckpointDirCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

def evaluate_chemical_predictions(model, dataset, tokenizer, evaluator, generation_config, max_samples=100):
    """
    评估化学反应预测的准确性
    
    Args:
        model: 训练好的模型
        dataset: 评估数据集
        tokenizer: 分词器
        evaluator: 化学反应评估器
        max_samples: 最大评估样本数
        
    Returns:
        评估指标字典
    """
    print(f"\n=== 开始化学反应预测评估 (样本数: {min(max_samples, len(dataset))}) ===")
    
    model.set_train(False)
    evaluator.reset_metrics()
    
    # 限制评估样本数以节省时间
    eval_samples = min(max_samples, len(dataset))
    
    for i in range(eval_samples):
        try:
            sample = dataset[i]
            
            # 准备输入 (只使用prompt部分)
            input_text = sample['input_text']
            prompt_encoding = tokenizer(input_text, return_tensors="ms")
            
            # 准备多模态输入
            rxn_input_ids = ms.Tensor([sample['rxn_input_ids']], dtype=ms.int32)
            rxn_attention_mask = ms.Tensor([sample['rxn_attention_mask']], dtype=ms.int32)
            
            # 生成预测
            # MindSpore在推理模式下不需要显式的no_grad上下文管理器
            generated_ids = model.generate(
                input_ids=prompt_encoding.input_ids,
                attention_mask=prompt_encoding.attention_mask,
                rxn_input_ids=rxn_input_ids,
                rxn_attention_mask=rxn_attention_mask,
                max_new_tokens=generation_config['max_new_tokens'],
                do_sample=generation_config['do_sample'],
                pad_token_id=tokenizer.pad_token_id,
                num_beams=generation_config['num_beams']
            )
            
            # 解码生成结果
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # 移除输入部分，只保留生成的部分
            if input_text in generated_text:
                generated_text = generated_text.replace(input_text, "").strip()
            
            # 清理生成文本中的特殊标记
            generated_text = generated_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<|im_start|>", "").strip()
            
            # 从样本中提取真实产物SMILES
            target_text = sample['target_text']
            # 移除可能的前缀和后缀，包括各种特殊标记
            target_smiles = target_text.replace(tokenizer.eos_token, "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            
            # 添加到评估器
            evaluator.add_prediction(generated_text, target_smiles)
            
            # 每10个样本打印一次进度
            if (i + 1) % 10 == 0:
                print(f"已评估: {i+1}/{eval_samples}")
                
        except Exception as e:
            print(f"评估样本 {i} 时出错: {e}")
            # 添加一个失败的预测
            evaluator.add_prediction("", sample.get('target_text', ''))
    
    # 计算最终指标
    metrics = evaluator.compute_metrics()
    
    print(f"\n=== 评估完成 ===")
    print(f"总样本数: {metrics.get('total_predictions', 0)}")
    print(f"有效预测: {metrics.get('valid_predictions', 0)}")
    print(f"精确匹配: {metrics.get('exact_matches', 0)}")
    print(f"分子有效性: {metrics.get('validity_rate', 0):.4f}")
    print(f"Top-1准确率: {metrics.get('top1_accuracy', 0):.4f}")
    
    return metrics

def setup_parallel_environment(config: Dict[str, Any]) -> tuple:
    """配置并行环境 - 适配msrun动态组网"""
    if not config.get('parallel', {}).get('enabled', False):
        return 0, 1
    
    print("=== 配置并行环境 (msrun动态组网) ===")
    
    # 1. 首先从环境变量获取rank_id并设置设备（必须在init()之前）
    rank_id = int(os.getenv('RANK_ID', '0'))
    ms.set_device("Ascend", rank_id)
    
    # 2. 初始化通信
    init()
    
    # 3. 获取并行信息
    rank_id = get_rank()
    device_num = get_group_size()
    
    print(f"当前进程rank: {rank_id}, 总设备数: {device_num}")
    
    # 4. 配置自动并行上下文 - 适配动态组网
    set_auto_parallel_context(
        parallel_mode=ParallelMode.DATA_PARALLEL,  # 数据并行模式
        gradients_mean=True,                      # 梯度平均
        device_num=device_num,                    # 设备数量
        parameter_broadcast=True                   # 参数广播
    )
    
    print(f"并行配置完成: 数据并行模式, {device_num}卡 (msrun动态组网)")
    return rank_id, device_num

def create_distributed_dataset(dataset, batch_size: int, rank_id: int, device_num: int, config: Dict[str, Any]):
    """创建分布式数据集，使用DatasetWrapper支持索引访问"""
    print(f"=== 创建分布式数据集 (rank {rank_id}/{device_num}) ===")
    
    # 创建一个包装类，提供生成器接口但支持索引访问
    class DatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
            self.indices = list(range(len(dataset)))
            random.shuffle(self.indices)  # 打乱数据顺序
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # 使用打乱后的索引访问数据
            actual_idx = self.indices[idx]
            sample = self.dataset[actual_idx]
            return (
                sample["input_ids"],
                sample["attention_mask"],
                sample["labels"],
                sample["rxn_input_ids"],
                sample["rxn_attention_mask"]
            )
    
    # 创建包装后的数据集
    wrapped_dataset = DatasetWrapper(dataset)
    
    # 创建数据集，直接传递可索引的对象
    ms_dataset = ds.GeneratorDataset(
        wrapped_dataset,
        column_names=[
            "input_ids", "attention_mask", "labels",
            "rxn_input_ids", "rxn_attention_mask"
        ],
        shuffle=False,  # 已经在wrapper中shuffle
        num_parallel_workers=1,
        num_shards=device_num,  # 数据分片总数
        shard_id=rank_id        # 当前分片ID
    )
    
    # 批处理
    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True)
    
    print(f"分布式数据集创建完成: rank {rank_id}, 批次数: {ms_dataset.get_dataset_size()}")
    return ms_dataset

def create_single_dataset(dataset, batch_size: int, shuffle: bool = True):
    """创建单卡数据集，使用DatasetWrapper支持索引访问"""
    print(f"=== 创建单卡数据集 ===")
    
    # 创建一个包装类，提供生成器接口但支持索引访问
    class DatasetWrapper:
        def __init__(self, dataset, shuffle=True):
            self.dataset = dataset
            self.indices = list(range(len(dataset)))
            if shuffle:
                random.shuffle(self.indices)  # 打乱数据顺序
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # 使用索引访问数据
            actual_idx = self.indices[idx]
            sample = self.dataset[actual_idx]
            return (
                sample["input_ids"],
                sample["attention_mask"],
                sample["labels"],
                sample["rxn_input_ids"],
                sample["rxn_attention_mask"]
            )
    
    # 创建包装后的数据集
    wrapped_dataset = DatasetWrapper(dataset, shuffle)
    
    # 创建数据集，直接传递可索引的对象
    ms_dataset = ds.GeneratorDataset(
        wrapped_dataset,
        column_names=[
            "input_ids", "attention_mask", "labels",
            "rxn_input_ids", "rxn_attention_mask"
        ],
        shuffle=False,  # 已经在wrapper中处理shuffle
        num_parallel_workers=1
    )
    
    # 批处理
    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True)
    
    print(f"单卡数据集创建完成: 批次数: {ms_dataset.get_dataset_size()}")
    return ms_dataset

def setup_model_for_full(config: Dict[str, Any]):
    """设置模型用于全量微调"""
    model_config = config['model']
    model_path = model_config.get("model_path", None)

    # 检查是否有需要加载的权重
    checkpoint_path = config['training'].get('load_checkpoint', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n=== 从 {checkpoint_path} 加载预训练权重 ===")

        # 直接加载完整模型（包括配置）
        base_model = MultiModalQwen.from_pretrained(
            checkpoint_path,
            qwen_model_name=model_config['qwen_model_name'],
            rxn_model_name=model_config['rxn_model_name'],
            proj_dropout=float(model_config['proj_dropout']),
            freeze_backbones=False,  # 全量微调时不冻结
            use_cls_token=bool(model_config['use_cls_token'])
        )
    elif model_path:
        base_model = MultiModalQwen.from_pretrained(model_path)
    else:
        # 创建基础模型配置
        print(f"\n=== 直接加载基础模型 ===")
        mm_config = MultiModalQwenConfig(
            qwen_model_name=model_config['qwen_model_name'],
            rxn_model_name=model_config['rxn_model_name'],
            proj_dropout=float(model_config['proj_dropout']),  # 确保proj_dropout是float类型
            freeze_backbones=bool(model_config['freeze_backbones']),  # 确保freeze_backbones是bool类型
            use_extended_vocab=bool(model_config['use_extended_vocab']),  # 确保use_extended_vocab是bool类型
            extended_vocab_path=model_config['extended_vocab_path'],
            sentinel_token=model_config['sentinel_token'],
            freeze_qwen=bool(model_config['freeze_qwen']),
            freeze_reactionbert=bool(model_config['freeze_reactionbert']),
            use_cls_token=bool(model_config['use_cls_token'])
        )
        # 创建基础模型（freeze_backbones已在模型内部处理参数冻结）
        base_model = MultiModalQwen(mm_config)
    
    print("\n=== 基础模型加载完成 ===")

    if model_config['freeze_reactionbert']:
        for param in base_model.rxn.get_parameters():
            param.requires_grad = False

    base_model.print_trainable_parameters()

    return base_model


def full_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_experiment_name(config: Dict[str, Any]) -> str:
    """根据配置生成实验名称"""
    # 提取模型大小
    qwen_model_name = config['model']['qwen_model_name']
    if '0.5B' in qwen_model_name:
        model_size = '0.5B'
    elif '1.5B' in qwen_model_name:
        model_size = '1.5B'
    elif '3B' in qwen_model_name:
        model_size = '3B'
    elif '7B' in qwen_model_name:
        model_size = '7B'
    else:
        model_size = 'unknown'
    
    # 提取其他参数
    task_type = config['data']['task_type']
    batch_size = config['training']['batch_size']
    learning_rate = float(config['training']['learning_rate'])  # 确保转换为float
    
    # 生成时间戳（北京时间）
    beijing_tz = ZoneInfo('Asia/Shanghai')
    timestamp = datetime.now(beijing_tz).strftime('%Y%m%d_%H%M%S')
    
    # 格式化学习率（去掉科学计数法中的e）
    lr_str = f"{learning_rate:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    
    # 生成实验名称
    experiment_name = f"{model_size}_{task_type}_bs{batch_size}_lr{lr_str}_{timestamp}"
    
    return experiment_name


def main():
    parser = argparse.ArgumentParser(description="多卡全量微调训练化学反应预测模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = full_config(args.config)
    
    # 设置随机种子
    seed = config.get('random_seed', 42)
    ms.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # 配置并行环境
    rank_id, device_num = setup_parallel_environment(config)

    experiment_name = generate_experiment_name(config)
    output_dir = os.path.join(config['data']['output_dir'], experiment_name)
        
    # 打印配置信息（只在rank 0打印）
    if rank_id == 0:
        print("=== 多卡全量微调训练配置 ===")
        print(f"配置文件: {args.config}")
        print(f"并行环境: rank {rank_id}/{device_num}")
        print(f"数据路径: {config['data']['data_path']}")
        print(f"输出目录: {output_dir}")
        print(f"训练参数: epochs={config['training']['epochs']}, batch_size={config['training']['batch_size']}, lr={config['training']['learning_rate']}")
    
    # 初始化分词器
    if rank_id == 0:
        print("\n=== 初始化分词器 ===")
    
    model_config = config['model']
    if model_config['use_extended_vocab']:
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config['extended_vocab_path'])
    else:
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config['qwen_model_name'])

    rxn_tokenizer = get_default_tokenizer()
    
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    print("\n=== 分词器加载完成 ===")
    
    # 加载数据
    if rank_id == 0:
        print("\n=== 加载数据 ===")
    
    data_path = config['data']['data_path']
    
    # 从文件夹中分别读取三个数据集文件
    train_file = os.path.join(data_path, 'train.txt')
    valid_file = os.path.join(data_path, 'valid.txt')
    test_file = os.path.join(data_path, 'test.txt')
    
    # 检查文件是否存在
    for file_path, file_name in [(train_file, 'train.txt'), (valid_file, 'valid.txt'), (test_file, 'test.txt')]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 分别解析三个数据集
    train_reactions = parse_reactions(train_file)
    val_reactions = parse_reactions(valid_file)
    test_reactions = parse_reactions(test_file)
    
    if rank_id == 0:
        print(f"训练集: {len(train_reactions)} 个反应 (来自 {train_file})")
        print(f"验证集: {len(val_reactions)} 个反应 (来自 {valid_file})")
        print(f"测试集: {len(test_reactions)} 个反应 (来自 {test_file})")
    
    # 创建数据集
    if rank_id == 0:
        print("\n=== 创建数据集 ===")
    
    data_config = config['data']
    train_dataset = DualRepresentationDataset(
        reactions=train_reactions,
        qwen_tokenizer=qwen_tokenizer,
        rxn_tokenizer=rxn_tokenizer,
        task_type=data_config['task_type'],
        max_len=data_config['max_length'],
        rxn_max_len=data_config['rxn_max_length'],
        use_cls_token=data_config['use_cls_token']
    )
    
    val_dataset = DualRepresentationDataset(
        reactions=val_reactions,
        qwen_tokenizer=qwen_tokenizer,
        rxn_tokenizer=rxn_tokenizer,
        task_type=data_config['task_type'],
        max_len=data_config['max_length'],
        rxn_max_len=data_config['rxn_max_length'],
        use_cls_token=data_config['use_cls_token']
    )
    
    # 创建数据加载器
    train_config = config['training']
    if device_num > 1:
        train_loader = create_distributed_dataset(
            train_dataset, train_config['batch_size'], rank_id, device_num, config
        )
        val_loader = create_distributed_dataset(
            val_dataset, train_config['batch_size'], rank_id, device_num, config
        )
    else:
        train_loader = create_single_dataset(train_dataset, train_config['batch_size'], shuffle=True)
        val_loader = create_single_dataset(val_dataset, train_config['batch_size'], shuffle=False)
    
    if rank_id == 0:
        print(f"训练数据加载器: {train_loader.get_dataset_size()} 批次")
        print(f"验证数据加载器: {val_loader.get_dataset_size()} 批次")
    
    
    # 设置模型
    if rank_id == 0:
        print("\n=== 设置全量微调模型 ===")
    
    model = setup_model_for_full(config) 

    # 创建回调
    loss_recorder = TrainLossRecorder(output_dir, rank_id)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    ensure_checkpointdir = EnsureCheckpointDirCallback()
    print("\n=== 模型加载完成 ===")

    # 训练参数
    total_steps = train_loader.get_dataset_size() * train_config['epochs']
    warmup_steps = int(total_steps * train_config.get('warmup_ratio', 0.1))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config['epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        per_device_eval_batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=float(train_config['learning_rate']),  # 确保learning_rate是float类型
        lr_scheduler_type=train_config['lr_scheduler_type'],
        warmup_steps=warmup_steps,
        weight_decay=float(train_config['weight_decay']),  # 确保weight_decay是float类型
        evaluation_strategy=train_config['evaluation_strategy'],
        eval_steps=train_config['eval_steps'],
        save_strategy=train_config['save_strategy'],
        save_steps=train_config['save_steps'],
        save_total_limit=train_config['save_total_limit'], 
        logging_strategy=train_config['logging_strategy'],
        logging_steps=train_config['logging_steps'],
        load_best_model_at_end=train_config['load_best_model_at_end'],
        metric_for_best_model=train_config['metric_for_best_model'],
        greater_is_better=train_config['greater_is_better'],
        overwrite_output_dir=True,
        fp16=train_config['fp16'],
    )
    
    # 只在rank 0初始化SwanLab
    swanlab_callback = None
    if rank_id == 0:
        print(f"SwanLab实验名称: {experiment_name}")
        swanlab_callback = SwanLabCallback(
            project="Qwen2.5-full", 
            experiment_name=experiment_name
        )
    
    # 创建训练器
    if rank_id == 0:
        print("\n=== 创建训练器 ===")
    
    # 构建回调列表，只在rank 0添加SwanLab回调
    callbacks = [loss_recorder, early_stopping]
    if rank_id == 0 and swanlab_callback is not None:
        callbacks.append(swanlab_callback)
        callbacks.append(ensure_checkpointdir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        tokenizer=qwen_tokenizer,
        callbacks=callbacks,
    )
    
    # 开始训练
    if rank_id == 0:
        print("\n=== 开始多卡全量微调训练 ===")

    trainer.train()

    # 保存模型（只在rank 0保存）
    if rank_id == 0:
        print("\n=== 保存模型 ===")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(f"{output_dir}/final_full_model")
        qwen_tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    
        # 最终评估
        if len(test_reactions) > 0:
            print("\n=== 最终评估 ===")
            test_dataset = DualRepresentationDataset(
                reactions=test_reactions,
                qwen_tokenizer=qwen_tokenizer,
                rxn_tokenizer=rxn_tokenizer,
                task_type=config['data']['task_type'],
                max_len=config['data']['max_length'],
                rxn_max_len=config['data']['rxn_max_length']
            )
            
            # 创建化学反应评估器
            chemical_evaluator = ChemicalReactionEvaluator()
            
            # 进行化学反应预测评估
            final_metrics = evaluate_chemical_predictions(
                model=model,
                dataset=test_dataset,
                tokenizer=qwen_tokenizer,
                evaluator=chemical_evaluator,
                generation_config=config['generation'],
                max_samples=50  # 限制评估样本数以节省时间
            )
            
            # 打印详细评估摘要
            chemical_evaluator.print_evaluation_summary()
            
            # 计算传统的困惑度作为参考
            test_loader = create_single_dataset(test_dataset, config['training']['batch_size'], shuffle=False)
            traditional_results = trainer.evaluate(test_loader)
            traditional_loss = traditional_results.get("eval_loss", 0.0)
            perplexity = np.exp(traditional_loss)
            
            print(f"\n=== 综合评估结果 ===")
            print(f"化学反应预测指标:")
            print(f"  Top-1准确率: {final_metrics.get('top1_accuracy', 0):.4f} ({final_metrics.get('top1_accuracy', 0)*100:.2f}%)")
            print(f"  分子有效性: {final_metrics.get('validity_rate', 0):.4f} ({final_metrics.get('validity_rate', 0)*100:.2f}%)")
            print(f"  有效预测中的准确率: {final_metrics.get('valid_accuracy', 0):.4f} ({final_metrics.get('valid_accuracy', 0)*100:.2f}%)")
            print(f"传统语言模型指标:")
            print(f"  Loss: {traditional_loss:.4f}")
            print(f"  Perplexity: {perplexity:.4f}")
            
            # 保存最终结果
            with open(f"{output_dir}/final_results.json", "w") as f:
                json.dump({
                    # 化学反应预测指标
                    "top1_accuracy": float(final_metrics.get('top1_accuracy', 0)),
                    "validity_rate": float(final_metrics.get('validity_rate', 0)),
                    "valid_accuracy": float(final_metrics.get('valid_accuracy', 0)),
                    "total_predictions": final_metrics.get('total_predictions', 0),
                    "exact_matches": final_metrics.get('exact_matches', 0),
                    "valid_predictions": final_metrics.get('valid_predictions', 0),
                    
                    # 传统语言模型指标
                    "eval_loss": float(traditional_loss),
                    "perplexity": float(perplexity),
                    
                    "training_args": {
                        "epochs": config['training']['epochs'],
                        "batch_size": config['training']['batch_size'],
                        "learning_rate": config['training']['learning_rate'],
                        "max_length": config['data']['max_length'],
                        "rxn_max_length": config['data']['rxn_max_length']
                    }
                }, f, indent=2)
            
            # 保存详细的预测结果
            detailed_results = chemical_evaluator.get_detailed_results()
            with open(f"{output_dir}/detailed_predictions.json", "w") as f:
                json.dump(detailed_results, f, indent=2)
            
            print(f"\n详细结果已保存至: {output_dir}/detailed_predictions.json")
    
    
    if rank_id == 0:
        print("\n=== 多卡全量微调训练完成 ===")
        print(f"模型保存路径: {output_dir}/final_full_model")
        print(f"训练日志保存路径: {output_dir}/train_loss.txt")
    
    # 分布式训练结束，确保所有进程同步
    if device_num > 1:
        print(f"进程 {rank_id} 训练完成")


if __name__ == "__main__":
    main()