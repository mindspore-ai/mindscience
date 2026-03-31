# -*- coding: utf-8 -*-
"""
溶剂预测任务多卡全量微调训练脚本 (适配闭集多标签二分类版 - Wrapper nn.Cell 架构)
基于train_full_multi.py

使用方式:
python train/train_solvent_multi.py --config config/solvent_1.5b_training.yaml


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

from mindnlp.engine import Trainer, TrainingArguments
from mindnlp.engine.callbacks import TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback
from mindnlp.transformers import AutoTokenizer

# 确保这里的 ReactionQwenForSolventPrediction 是你修改过继承 nn.Cell 的版本
from model.reactionqwen_solvent import ReactionQwenForSolventPrediction, ReactionQwenForSolventPredictionConfig
from dataset.solvent_dataset import SolventPredictionDataset, parse_solvent_data
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
        _ = args, kwargs
        if logs:
            if "loss" in logs:
                self.train_logs.append((state.global_step, logs["loss"]))
            if "eval_loss" in logs:
                self.eval_logs.append((state.global_step, logs["eval_loss"]))
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        _ = args, state, control, kwargs
        if self.rank_id == 0:
            with open(f"{self.output_dir}/train_loss.txt", "w") as f:
                for step, loss in self.train_logs:
                    f.write(f"{step}\t{loss:.6f}\n")
            with open(f"{self.output_dir}/eval_loss.txt", "w") as f:
                for step, loss in self.eval_logs:
                    f.write(f"{step}\t{loss:.6f}\n")
            print(f"Training logs saved to {self.output_dir}/")
        return control
    
class EnsureCheckpointDirCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

def setup_parallel_environment(config: Dict[str, Any]) -> tuple:
    """配置并行环境 - 适配msrun动态组网"""
    if not config.get('parallel', {}).get('enabled', False):
        return 0, 1
    
    print("=== 配置并行环境 (msrun动态组网) ===")
    rank_id = int(os.getenv('RANK_ID', '0'))
    ms.set_device("Ascend", rank_id)
    init()
    rank_id = get_rank()
    device_num = get_group_size()
    
    set_auto_parallel_context(
        parallel_mode=ParallelMode.DATA_PARALLEL,
        gradients_mean=True,
        device_num=device_num,
        parameter_broadcast=True
    )
    print(f"并行配置完成: 数据并行模式, {device_num}卡 (msrun动态组网)")
    return rank_id, device_num

def create_distributed_dataset(dataset, batch_size: int, rank_id: int, device_num: int, config: Dict[str, Any]):
    """创建分布式数据集"""
    print(f"=== 创建分布式数据集 (rank {rank_id}/{device_num}) ===")
    class DatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
            self.indices = list(range(len(dataset)))
            random.shuffle(self.indices)
        def __len__(self): return len(self.dataset)
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            sample = self.dataset[actual_idx]
            return (
                sample["input_ids"], sample["attention_mask"], sample["solvent_labels"],
                sample["rxn_input_ids"], sample["rxn_attention_mask"]
            )
    
    wrapped_dataset = DatasetWrapper(dataset)
    ms_dataset = ds.GeneratorDataset(
        wrapped_dataset,
        column_names=["input_ids", "attention_mask", "solvent_labels", "rxn_input_ids", "rxn_attention_mask"],
        shuffle=False,
        num_parallel_workers=1,
        num_shards=device_num,
        shard_id=rank_id
    )
    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True)
    return ms_dataset

def create_single_dataset(dataset, batch_size: int, shuffle: bool = True):
    """创建单卡数据集"""
    print(f"=== 创建单卡数据集 ===")
    class DatasetWrapper:
        def __init__(self, dataset, shuffle=True):
            self.dataset = dataset
            self.indices = list(range(len(dataset)))
            if shuffle: random.shuffle(self.indices)
        def __len__(self): return len(self.dataset)
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            sample = self.dataset[actual_idx]
            return (
                sample["input_ids"], sample["attention_mask"], sample["solvent_labels"],
                sample["rxn_input_ids"], sample["rxn_attention_mask"]
            )
    
    wrapped_dataset = DatasetWrapper(dataset, shuffle)
    ms_dataset = ds.GeneratorDataset(
        wrapped_dataset,
        column_names=["input_ids", "attention_mask", "solvent_labels", "rxn_input_ids", "rxn_attention_mask"],
        shuffle=False,
        num_parallel_workers=1
    )
    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True)
    return ms_dataset

def setup_model_for_training(config):
    """
    标准训练初始化函数：
    1. 实例化新版 Wrapper 模型 (nn.Cell)
    2. 根据 config 自动加载预训练的 Backbone (Qwen/Bert)
    3. 根据 config 决定是否冻结 Backbone
    """
    print("\n=== [Full Training Mode] 初始化模型 ===")
    
    model_params = config['model']
    
    # 1. 构建 Backbone Config
    backbone_config_dict = {
        'qwen_model_name': model_params['qwen_model_name'],
        'rxn_model_name': model_params['rxn_model_name'],
        'proj_dropout': float(model_params.get('proj_dropout', 0.1)),
        # 注意：这里的 freeze 选项会传递给 MultiModalQwen 内部处理
        'freeze_backbones': bool(model_params.get('freeze_backbones', False)),
        'freeze_qwen': bool(model_params.get('freeze_qwen', False)),
        'freeze_reactionbert': bool(model_params.get('freeze_reactionbert', True)),
        
        'use_extended_vocab': bool(model_params.get('use_extended_vocab', True)),
        'extended_vocab_path': model_params.get('extended_vocab_path', None),
        'sentinel_token': model_params.get('sentinel_token', '<SMI_TOKEN>'),
        'use_cls_token': bool(model_params.get('use_cls_token', True))
    }

    # 2. 构建整体 Config
    solvent_config = ReactionQwenForSolventPredictionConfig(
        backbone_config=backbone_config_dict,
        num_solvent_labels=int(model_params.get('num_solvent_labels', 51)),
        pos_weight_value=float(model_params.get('pos_weight_value', 1.0)),
        pooling_type=model_params.get('pooling_type', 'mean'),
        loss_weight_multi_label=float(model_params.get('loss_weight_multi_label', 1.0))
    )
    
    print("正在实例化模型结构 (nn.Cell Wrapper Mode)...")
    # 这里会调用你修改后的 nn.Cell 类的 __init__
    # 它会自动下载/加载 Qwen 和 ReactionBERT 的预训练权重 (由 MindNLP 内部处理)
    model = ReactionQwenForSolventPrediction(solvent_config)
    
    # --- 打印参数量确认 ---
    # 这次应该能看到几百 M 的参数了
    total_params = len(list(model.get_parameters()))
    trainable_params = len(list(model.trainable_params()))
    print(f"  - 模型总参数量 (Tensor数): {total_params}") 
    print(f"  - 可训练参数量 (Tensor数): {trainable_params}")
    
    # 简单的逻辑检查：如果参数量 < 10，说明 Backbone 还是没注册上
    if total_params < 100:
        raise RuntimeError(f"❌ 严重错误：Backbone 参数未注册！当前只有 {total_params} 个参数。请检查 model定义里的 self._cells['backbone'] = ... 是否加上了。")
    
    print("✅ 参数量正常，准备开始全量训练...")
    return model

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_experiment_name(config: Dict[str, Any]) -> str:
    qwen_model_name = config['model']['qwen_model_name']
    if '0.5B' in qwen_model_name: model_size = '0.5B'
    elif '1.5B' in qwen_model_name: model_size = '1.5B'
    elif '3B' in qwen_model_name: model_size = '3B'
    elif '7B' in qwen_model_name: model_size = '7B'
    else: model_size = 'unknown'
    
    task_type = config['data']['task_type']
    batch_size = config['training']['batch_size']
    learning_rate = float(config['training']['learning_rate'])
    
    beijing_tz = ZoneInfo('Asia/Shanghai')
    timestamp = datetime.now(beijing_tz).strftime('%Y%m%d_%H%M%S')
    lr_str = f"{learning_rate:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    
    return f"{model_size}_{task_type}_bs{batch_size}_lr{lr_str}_{timestamp}"

def main():
    parser = argparse.ArgumentParser(description="溶剂预测任务多卡全量微调训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    config = load_config(args.config)
    seed = config.get('random_seed', 42)
    ms.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    rank_id, device_num = setup_parallel_environment(config)
    experiment_name = generate_experiment_name(config)
    output_dir = os.path.join(config['data']['output_dir'], experiment_name)
        
    if rank_id == 0:
        print("=== 溶剂预测任务多卡全量微调训练配置 ===")
        print(f"配置文件: {args.config}")
        print(f"数据路径: {config['data']['data_path']}")
        print(f"输出目录: {output_dir}")
    
    # 初始化分词器
    if rank_id == 0: print("\n=== 初始化分词器 ===")
    model_config = config['model']
    if model_config['use_extended_vocab']:
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config['extended_vocab_path'])
    else:
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config['qwen_model_name'])
    rxn_tokenizer = get_default_tokenizer()
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    if model_config['sentinel_token'] not in qwen_tokenizer.vocab:
        qwen_tokenizer.add_tokens([model_config['sentinel_token']])
    
    # 加载数据
    data_path = config['data']['data_path']
    train_file = os.path.join(data_path, 'train.txt')
    valid_file = os.path.join(data_path, 'valid.txt')
    test_file = os.path.join(data_path, 'test.txt')
    
    train_data = parse_solvent_data(train_file)
    val_data = parse_solvent_data(valid_file)
    test_data = parse_solvent_data(test_file)
    
    data_config = config['data']
    train_dataset = SolventPredictionDataset(train_data, qwen_tokenizer, rxn_tokenizer, data_config['language'], data_config['sentinel_token'], data_config['max_length'], data_config['rxn_max_length'], data_config['use_cls_token'])
    val_dataset = SolventPredictionDataset(val_data, qwen_tokenizer, rxn_tokenizer, data_config['language'], data_config['sentinel_token'],data_config['max_length'], data_config['rxn_max_length'], data_config['use_cls_token'])
    test_dataset = SolventPredictionDataset(test_data, qwen_tokenizer, rxn_tokenizer, data_config['language'], data_config['sentinel_token'],data_config['max_length'], data_config['rxn_max_length'], data_config['use_cls_token'])
    
    train_config = config['training']
    if device_num > 1:
        train_loader = create_distributed_dataset(train_dataset, train_config['batch_size'], rank_id, device_num, config)
        val_loader = create_distributed_dataset(val_dataset, train_config['batch_size'], rank_id, device_num, config)
    else:
        train_loader = create_single_dataset(train_dataset, train_config['batch_size'], shuffle=True)
        val_loader = create_single_dataset(val_dataset, train_config['batch_size'], shuffle=False)
    
    # --- 设置模型 (从头训练模式) ---
    if rank_id == 0: print("\n=== 设置溶剂预测模型 (Fresh Init) ===")
    
    # 使用标准的初始化函数
    model = setup_model_for_training(config) 

    # --- 关键检查：确保分类头存在且可训练 ---
    if rank_id == 0:
        print("Checking model parameters...")
        # 注意：现在是 nn.Cell，参数字典是扁平的
        params_dict = {p.name: p for p in model.get_parameters()}
        
        # 你的分类头名字现在应该是 solvent_classifier.weight
        if 'solvent_classifier.weight' in params_dict:
            print(f"✅ solvent_classifier.weight found! Shape: {params_dict['solvent_classifier.weight'].shape}")
            if params_dict['solvent_classifier.weight'].requires_grad:
                print("✅ solvent_classifier is TRAINABLE.")
            else:
                print("❌ WARNING: solvent_classifier is FROZEN!")
        else:
            print("❌ ERROR: solvent_classifier.weight NOT found in parameters!")
            # 打印一些存在的参数名帮助调试
            print(f"Available params sample: {list(params_dict.keys())[:5]}")

    loss_recorder = TrainLossRecorder(output_dir, rank_id)
    early_stopping = EarlyStoppingCallback(early_stopping_patience=config['training']['early_stopping_patience'])
    ensure_checkpointdir = EnsureCheckpointDirCallback()

    total_steps = train_loader.get_dataset_size() * train_config['epochs']
    warmup_steps = int(total_steps * train_config.get('warmup_ratio', 0.1))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config['epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        per_device_eval_batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=float(train_config['learning_rate']),
        lr_scheduler_type=train_config['lr_scheduler_type'],
        warmup_steps=warmup_steps,
        weight_decay=float(train_config['weight_decay']),
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
    
    swanlab_callback = None
    if rank_id == 0:
        print(f"SwanLab实验名称: {experiment_name}")
        swanlab_callback = SwanLabCallback(project="Qwen2.5-solvent", experiment_name=experiment_name)
    
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
    
    if rank_id == 0: 
        print("\n=== 开始溶剂预测任务多卡全量微调训练 ===")
    
    trainer.train()

    if rank_id == 0:
        print("\n=== 保存模型 ===")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存 Config 和 Tokenizer (这个保留，即使报错也不影响核心权重)
        try:
            model.save_pretrained(f"{output_dir}/final_solvent_model")
            qwen_tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        except Exception as e:
            print(f"⚠️ Warning: save_pretrained failed (non-critical): {e}")

        # 2. 核心：强制保存完整权重 (.ckpt)
        manual_ckpt_path = os.path.join(output_dir, "final_solvent_model", "full_model_manual.ckpt")
        print(f"正在强制保存参数列表到: {manual_ckpt_path} ...")

        # === 🚨 修复开始：显式构建参数列表 ===
        # 不要直接传 model，而是传 [{"name":..., "data":...}] 列表
        # 这样 MindSpore 就不需要检查 model 是否是 nn.Cell 了
        try:
            # 获取所有参数
            # 注意：使用 parameters_and_names 确保名字准确
            save_obj = []
            for name, param in model.parameters_and_names():
                save_obj.append({"name": name, "data": param})
            
            # 执行保存
            ms.save_checkpoint(save_obj, manual_ckpt_path)
            print(f"✅ 完整参数已保存！文件大小: {os.path.getsize(manual_ckpt_path) / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"❌ 严重错误：手动保存失败: {e}")
            # 最后的保底尝试：保存 parameters_dict
            try:
                ms.save_checkpoint(model.parameters_dict(), manual_ckpt_path)
                print("✅ 通过 parameters_dict 保存成功！")
            except:
                print("❌ 彻底保存失败。")
        # === 🚨 修复结束 ===

        # ... (后续评估代码保持不变) ...
        print("\n=== 最终评估 ===")
        test_loader = create_single_dataset(test_dataset, config['training']['batch_size'], shuffle=False)
        final_results = trainer.evaluate(test_loader)
        final_loss = final_results.get("eval_loss", 0.0)
        
        print(f"\n=== 溶剂预测任务评估结果 (Test Set) ===")
        print(f"  Final Loss: {final_loss:.4f}")
        
        with open(f"{output_dir}/final_results.json", "w") as f:
            json.dump({
                "eval_loss": float(final_loss),
                "training_args": {
                    "epochs": config['training']['epochs'],
                    "batch_size": config['training']['batch_size'],
                    "learning_rate": config['training']['learning_rate']
                }
            }, f, indent=2)
    
    if rank_id == 0:
        print(f"训练完成. 日志: {output_dir}/train_loss.txt")
    
    if device_num > 1:
        print(f"进程 {rank_id} 训练完成")

if __name__ == "__main__":
    main()