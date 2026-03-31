# -*- coding: utf-8 -*-
"""
溶剂预测任务推理脚本 - 支持多卡
基于 inference_full_multi.py 修改，适配溶剂预测的多任务分类模型

核心评估指标:
- 精确匹配率 (Exact Match Ratio, EMR): 
  模型预测的溶剂列表 (e.g., ['water', 'toluene'])
  是否与真实的溶剂列表 (e.g., ['water', 'toluene']) 完全一致（顺序无关）。
  这评估了 model.predict_solvents() 策略的最终准确性。

使用方式:
msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 --master_port=8120 \
  --log_dir=./logs/solvent/full_inference_logs --join=True inference/inference_solvent_multi.py \
  --config config/solvent_1.5b_inference.yaml
"""
import os
import json
import yaml
import argparse
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import set_auto_parallel_context, ParallelMode
from typing import Dict, Any, List

from mindnlp.transformers import AutoTokenizer

# --- 溶剂预测任务特定的导入 ---
from model.reactionqwen_solvent import ReactionQwenForSolventPrediction, ReactionQwenForSolventPredictionConfig
from dataset.solvent_dataset import SolventPredictionDataset, parse_solvent_data
from model.tokenizer import get_default_tokenizer

import mindspore.train.serialization as serialization

# --- 【关键修复】Monkey Patch: 手动添加缺失的类型映射 ---
# 这一步是为了解决 KeyError: 'mindspore.float32'
if hasattr(serialization, 'tensor_to_ms_type'):
    # 强制告诉 MindSpore: 看到 "mindspore.float32" 时，把它当做 ms.float32 处理
    serialization.tensor_to_ms_type['mindspore.float32'] = ms.float32
    print("✅ 已应用 MindSpore 类型映射补丁 (mindspore.float32 -> ms.float32)")
else:
    print("⚠️ 警告: 无法找到 tensor_to_ms_type，补丁可能未生效")
# ------------------------------------------------------

class SolventEvaluator:
    """
    用于溶剂预测任务的评估器
    计算精确匹配率 (Exact Match Ratio, EMR)
    """
    def __init__(self, id_to_solvent_map: Dict[int, str]):
        self.id_to_solvent_map = id_to_solvent_map
        self.reset_metrics()

    def reset_metrics(self):
        self.total_samples = 0
        self.exact_matches = 0

    def add_prediction(self, pred_solvents: List[str], true_solvents: List[str]) -> Dict:
        """
        添加一个预测结果并返回该样本的匹配状态
        
        Args:
            pred_solvents: 模型预测的溶剂名称列表
            true_solvents: 真实的溶剂名称列表
        """
        self.total_samples += 1
        
        # 使用集合(set)进行无序比较
        is_match = (set(pred_solvents) == set(true_solvents))
        
        if is_match:
            self.exact_matches += 1
        
        return {
            'exact_match': is_match,
            'predicted_solvents': pred_solvents,
            'true_solvents': true_solvents
        }

    def compute_metrics(self) -> Dict:
        """计算最终的EMR"""
        if self.total_samples == 0:
            return {
                'exact_match_ratio': 0.0,
                'total_samples': 0,
                'exact_matches': 0
            }
        
        emr = self.exact_matches / self.total_samples
        
        return {
            'exact_match_ratio': emr,
            'total_samples': self.total_samples,
            'exact_matches': self.exact_matches
        }

    def print_evaluation_summary(self):
        """打印评估结果摘要"""
        metrics = self.compute_metrics()
        print("\n=== 溶剂预测评估结果 (EMR) ===")
        print(f"  总样本数: {metrics['total_samples']}")
        print(f"  精确匹配 (EMR): {metrics['exact_match_ratio']:.4f} ({metrics['exact_matches']}/{metrics['total_samples']})")
        print("  (EMR: 预测的溶剂列表与真实列表完全一致)")


def setup_parallel_environment(config: Dict[str, Any]) -> tuple:
    """配置并行环境 (与模板一致)"""
    parallel_config = config.get('parallel', {})
    parallel_enabled = parallel_config.get('enabled', False)
    
    if not parallel_enabled:
        print("=== 单卡推理模式 ===")
        device_id = int(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(',')[0]) if os.getenv('CUDA_VISIBLE_DEVICES') else 0
        try:
            ms.set_device("Ascend", device_id)
        except Exception:
            try:
                ms.set_device("GPU", device_id)
            except Exception:
                ms.set_device("CPU")
        print(f"单卡设备配置完成: device_id={device_id}")
        return 0, 1
    
    print("=== 配置并行推理环境 (msrun) ===")
    if 'RANK_ID' not in os.environ:
        raise RuntimeError("多卡推理模式需要设置RANK_ID环境变量。请使用msrun启动。")
    
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
    
    print(f"并行配置完成: rank {rank_id}/{device_num}")
    return rank_id, device_num


def load_solvent_maps(json_path: str) -> tuple[Dict[str, int], Dict[int, str], int]:
    """加载溶剂ID映射文件"""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"溶剂映射文件未找到: {json_path}\n"
                            "此文件 (e.g., solvent_map.json) 是必需的，用于将模型输出索引映射回溶剂名称。")
                            
    with open(json_path, 'r', encoding='utf-8') as f:
        solvent_to_id = json.load(f)
    
    # 我们需要反向映射（ID -> 名称）来进行解码
    id_to_solvent = {int(idx): name for name, idx in solvent_to_id.items()}
    num_solvents = len(solvent_to_id)
    
    print(f"从 {json_path} 加载了 {num_solvents} 种溶剂")
    return solvent_to_id, id_to_solvent, num_solvents

def setup_model_for_solvent(config: Dict[str, Any]):
    """
    设置溶剂预测模型 (手动加载模式 - 绕过类型检查)
    """
    print("\n=== 初始化溶剂预测模型 ===")
    
    model_config = config['model']
    
    # 1. 正在根据 Config 初始化模型结构...
    print(f"1. 正在根据 Config 初始化模型结构...")
    
    backbone_config_dict = {
        'qwen_model_name': model_config['qwen_model_name'],
        'rxn_model_name': model_config['rxn_model_name'],
        'proj_dropout': float(model_config['proj_dropout']),
        'freeze_backbones': bool(model_config['freeze_backbones']),
        'use_extended_vocab': bool(model_config['use_extended_vocab']),
        'extended_vocab_path': model_config['extended_vocab_path'],
        'sentinel_token': model_config['sentinel_token'],
        'freeze_qwen': bool(model_config['freeze_qwen']),
        'freeze_reactionbert': bool(model_config['freeze_reactionbert']),
        'use_cls_token': bool(model_config['use_cls_token'])
    }
    
    solvent_config = ReactionQwenForSolventPredictionConfig(
        backbone_config=backbone_config_dict,
        num_solvent_labels=model_config['classifier_head']['num_solvent_labels'],
        pos_weight_value=model_config['classifier_head']['pos_weight_value'],
        pooling_type=model_config['classifier_head']['pooling_type']
    )
    
    # 实例化模型
    model = ReactionQwenForSolventPrediction(solvent_config)
    
    # 2. 手动加载 Checkpoint
    manual_ckpt = model_config.get('load_checkpoint')
    
    if manual_ckpt and os.path.exists(manual_ckpt):
        print(f"2. 正在加载 Checkpoint: {manual_ckpt}")
        
        # 读取 ckpt 文件为字典
        param_dict = ms.load_checkpoint(manual_ckpt)
        
        # === 🚨 修复点在这里 ===
        # 获取模型当前的所有参数 (转为字典方便查找)
        # 这里的 k 是参数名(str), v 是参数对象(Parameter)
        model_params = {k: v for k, v in model.parameters_and_names()}
        
        print(f"   - Checkpoint 参数量: {len(param_dict)}")
        print(f"   - 模型参数量: {len(model_params)}")
        
        loaded_count = 0
        
        # === 手动循环赋值 ===
        for name, value in param_dict.items():
            if name in model_params:
                # 直接设置 Tensor 数据
                if isinstance(value, ms.Parameter):
                    model_params[name].assign_value(value.data)
                else:
                    model_params[name].assign_value(value)
                loaded_count += 1
            else:
                pass

        print(f"✅ 成功加载了 {loaded_count} 个参数！")
        
        if loaded_count == 0:
            raise RuntimeError("❌ 加载失败：没有匹配到任何参数名！请检查 ckpt 是否对应。")
            
        return model

    else:
        raise RuntimeError(f"❌ 无法找到 Checkpoint 文件: {manual_ckpt}")
    
def run_inference_and_evaluate(
    model: ReactionQwenForSolventPrediction,
    dataset: SolventPredictionDataset,
    raw_data: List[Dict],
    evaluator: SolventEvaluator,
    config: Dict[str, Any],
    id_to_solvent_map: Dict[int, str],
    rank_id: int = 0,
    device_num: int = 1,
    max_samples: int = None
) -> List[Dict]:
    """
    运行推理并评估
    """
    print(f"\n=== Rank {rank_id} 开始推理 ===")
    model.set_train(False)
    
    # 只在rank 0重置主评估器
    if rank_id == 0:
        evaluator.reset_metrics()
    
    total_samples = len(dataset)
    if max_samples is not None:
        total_samples = min(max_samples, total_samples)

    inference_config = config['inference']
    threshold = inference_config.get('threshold', 0.9) # 默认 0.4
    max_count = inference_config.get('max_count', 3)   # 默认最多 3 个
    
    # 数据分片
    samples_per_rank = total_samples // device_num
    start_idx = rank_id * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank_id < device_num - 1 else total_samples
    
    print(f"Rank {rank_id} 处理样本范围: {start_idx}-{end_idx-1} (共{end_idx-start_idx}个样本)")
    
    results = []
    for i in range(start_idx, end_idx):
        try:
            # 1. 从Dataset获取模型输入 (Tensors)
            sample = dataset[i]
            
            # 2. 从Raw Data获取真实标签 (文本)
            raw_sample = raw_data[i]
            true_solvents = raw_sample['solvents']
            
            # 3. 准备模型输入
            batch_inputs = {
                'input_ids': ms.Tensor([sample['input_ids']], dtype=ms.int32),
                'attention_mask': ms.Tensor([sample['attention_mask']], dtype=ms.int32),
                'rxn_input_ids': ms.Tensor([sample['rxn_input_ids']], dtype=ms.int32),
                'rxn_attention_mask': ms.Tensor([sample['rxn_attention_mask']], dtype=ms.int32),
            }
            
            # 4. 执行推理
            # 调用模型内置的 "先预测数量，再取Top-n" 策略
            prediction_result = model.predict_solvents(
                **batch_inputs,
                threshold=threshold,       # 稍微降低阈值，依靠排序来筛选
                max_count=max_count,
                solvent_vocab=id_to_solvent_map,  # 传递map，使输出为溶剂名称
                return_probabilities=True
            )
            
            predicted_solvents = prediction_result['predicted_solvents']
            
            # 5. 评估
            if rank_id == 0:
                result = evaluator.add_prediction(predicted_solvents, true_solvents)
            else:
                # 其他进程使用本地评估器
                local_evaluator = SolventEvaluator(id_to_solvent_map)
                result = local_evaluator.add_prediction(predicted_solvents, true_solvents)
            
            # 6. 保存详细结果
            result.update({
                'reaction': raw_sample['reaction'],
                'predicted_probabilities': prediction_result.get('solvent_probabilities', []),
                'rank_id': rank_id
            })
            results.append(result)
            
            if (i + 1 - start_idx) % 20 == 0:
                print(f"Rank {rank_id} 已处理: {i+1-start_idx}/{end_idx-start_idx}")
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            results.append({
                'reaction': raw_sample.get('reaction', ''),
                'predicted_solvents': [],
                'true_solvents': raw_sample.get('solvents', []),
                'exact_match': False,
                'error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="溶剂预测任务推理脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置并行环境
    rank_id, device_num = setup_parallel_environment(config)
    
    if rank_id == 0:
        print("=== 溶剂预测模型推理 ===")
        print(f"配置文件: {args.config}")
        print(f"并行环境: rank {rank_id}/{device_num}")
    
    # 初始化分词器
    model_config = config['model']
    if model_config['use_extended_vocab']:
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config['extended_vocab_path'])
    else:
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_config['qwen_model_name'])
    # --- 新增：强制检查哨兵 Token ---
    sentinel = config['model']['sentinel_token']
    if sentinel in qwen_tokenizer.get_vocab():
        print(f"✅ Tokenizer 已包含哨兵: {sentinel} (ID: {qwen_tokenizer.convert_tokens_to_ids(sentinel)})")
    else:
        print(f"⚠️ 警告: Tokenizer 未包含哨兵，正在添加 (这可能会导致 ID 错位!)")
        qwen_tokenizer.add_tokens([sentinel])

    qwen_tokenizer.padding_side = "left"
    rxn_tokenizer = get_default_tokenizer()
    if rank_id == 0:
        print("Qwen 和 RXN 分词器加载完成")

    # 加载关键的溶剂映射文件
    data_config = config['data']
    try:
        _, id_to_solvent_map, _ = load_solvent_maps(data_config['solvent_map_path'])
    except Exception as e:
        if rank_id == 0:
            print(f"加载 solvent_map_path 失败: {e}")
        return
    
    model = setup_model_for_solvent(config)
    model.set_train(False)
    
    if rank_id == 0:
        print("溶剂预测模型加载完成")
    
    # 加载原始数据
    test_data = parse_solvent_data(data_config['test_data_path'])
    
    # 创建数据集 (用于模型输入)
    test_dataset = SolventPredictionDataset(
        data=test_data,
        qwen_tokenizer=qwen_tokenizer,
        rxn_tokenizer=rxn_tokenizer,
        max_len=data_config['max_length'],
        rxn_max_len=data_config['rxn_max_length'],
        use_cls_token=model_config['use_cls_token']
    )
    # --- 新增：金标准 ID 检查 (Golden Check) ---
    # 在开始推理前，检查模型 embedding 层的词表大小和 tokenizer 是否一致
    print(f"Model Vocab Size: {model.backbone.qwen.model.embed_tokens.weight.shape[0]}")
    print(f"Tokenizer Vocab Size: {len(qwen_tokenizer)}")
    
    if len(qwen_tokenizer) != model.backbone.qwen.model.embed_tokens.weight.shape[0]:
        print("❌ 致命错误: 模型词表大小与 Tokenizer 不一致！推理必错！")
        # 这通常意味着 Tokenizer 加载错了，或者多加了 Token
    
    if rank_id == 0:
        print(f"测试集样本数: {len(test_dataset)}")
        if len(test_dataset) != len(test_data):
            print(f"警告: Dataset ({len(test_dataset)}) 与 Raw Data ({len(test_data)}) 长度不匹配!")
    
    # 初始化评估器
    evaluator = SolventEvaluator(id_to_solvent_map)
    
    # 运行推理
    results = run_inference_and_evaluate(
        model=model,
        dataset=test_dataset,
        raw_data=test_data,
        evaluator=evaluator,
        config=config,
        id_to_solvent_map=id_to_solvent_map,
        rank_id=rank_id,
        device_num=device_num,
        max_samples=config['inference'].get('max_samples')
    )
    
    # === 结果汇总 (与模板一致) ===
    
    output_dir = config['inference']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    if device_num > 1:
        # 多卡：先写出分片
        per_rank_path = f"{output_dir}/predictions_rank_{rank_id}.json"
        try:
            with open(per_rank_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if rank_id == 0:
                print(f"已写出分片结果: {per_rank_path}")
        except Exception as e:
            print(f"写出分片结果失败: {e}")

        # 进程同步
        from mindspore import ops
        print(f"进程 {rank_id} 推理完成，等待其他进程...")
        try:
            all_gather = ops.AllGather()
            sync_tensor = ms.Tensor([rank_id], dtype=ms.int32)
            all_gather(sync_tensor)
            if rank_id == 0:
                print("所有进程已完成推理")
        except Exception as e:
            print(f"进程同步失败: {e}，继续执行...")

    # 保存与汇总
    # 在多卡场景下，先各自写出分片结果，供rank 0汇总
    try:
        output_dir = config['inference']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        if device_num > 1:
            per_rank_path = f"{output_dir}/predictions_rank_{rank_id}.json"
            with open(per_rank_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if rank_id == 0:
                print(f"已写出分片结果: {per_rank_path}")
    except Exception as e:
        print(f"写出分片结果失败: {e}")

    # 进程同步：等待所有进程完成（确保各rank分片结果已写出）
    if device_num > 1:
        from mindspore import ops
        print(f"进程 {rank_id} 推理完成，等待其他进程...")
        try:
            # 使用AllGather进行同步，确保所有进程都到达这里
            all_gather = ops.AllGather()
            sync_tensor = ms.Tensor([rank_id], dtype=ms.int32)
            gathered_ranks = all_gather(sync_tensor)
            if rank_id == 0:
                print(f"所有进程已完成推理: {gathered_ranks.asnumpy()}")
        except Exception as e:
            print(f"进程同步失败: {e}，继续执行...")

    # 保存与汇总结果
    output_dir = config['inference']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    if device_num == 1:
        # 单卡：保持原有逻辑
        if results:
            metrics = evaluator.compute_metrics()
            with open(f"{output_dir}/eval_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            with open(f"{output_dir}/predictions.json", 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            evaluator.print_evaluation_summary()
            print(f"\n结果已保存至: {output_dir}")
        else:
            print("警告: 没有生成任何预测结果")
    else:
        # 多卡：仅在rank 0汇总所有分片，再统一评估与保存
        if rank_id == 0:
            print("开始汇总各rank分片结果...")
            all_results = []
            missing_shards = []
            for rid in range(device_num):
                shard_path = f"{output_dir}/predictions_rank_{rid}.json"
                if not os.path.exists(shard_path):
                    missing_shards.append(rid)
                    continue
                try:
                    with open(shard_path, 'r') as f:
                        shard = json.load(f)
                        if isinstance(shard, list):
                            all_results.extend(shard)
                        else:
                            print(f"分片文件内容异常（非list）: {shard_path}")
                except Exception as e:
                    print(f"读取分片失败 {shard_path}: {e}")
            if missing_shards:
                print(f"警告: 缺失以下rank的分片文件: {missing_shards}")

            # 统一评估（以防各rank评估逻辑差异，重新计算一次）
            merged_evaluator = SolventEvaluator(id_to_solvent_map=id_to_solvent_map)
            for item in all_results:
                pred = item.get('predicted_solvents', '')
                tgt = item.get('true_solvents', '')
                merged_evaluator.add_prediction(pred_solvents=pred, true_solvents=tgt)

            merged_metrics = merged_evaluator.compute_metrics()

            # 写出合并后的结果与指标
            with open(f"{output_dir}/eval_metrics.json", 'w') as f:
                json.dump(merged_metrics, f, indent=2)
            with open(f"{output_dir}/predictions.json", 'w') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            merged_evaluator.print_evaluation_summary()
            print(f"\n已汇总 {len(all_results)} 条预测，结果已保存至: {output_dir}")
        else:
            # 非rank 0无需再次写整体结果
            pass
    
    print(f"进程 {rank_id} 完成所有任务")

if __name__ == "__main__":
    main()