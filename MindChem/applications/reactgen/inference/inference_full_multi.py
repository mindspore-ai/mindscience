# -*- coding: utf-8 -*-
"""
全量微调模型推理脚本 - 化学反应预测
支持多卡推理和评估

使用方式:
msrun --worker_num=6 --local_worker_num=6 --master_addr=127.0.0.1 --master_port=8119 \
  --log_dir=./logs/forward/full_inference_results --join=True inference/inference_full_multi.py --config config/forward_1.5b_inference.yaml
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
from model.reactionqwen import MultiModalQwen
from eval.evaluation_metrics import ChemicalReactionEvaluator
from dataset.dataset import parse_reactions, DualRepresentationDataset
from model.tokenizer import get_default_tokenizer


def extract_assistant_response(full_output: str, input_text: str) -> str:
    """
    从完整的ChatML输出中正确提取assistant的回答
    
    Args:
        full_output: 包含ChatML格式的完整输出
        input_text: 输入文本（包含ChatML格式）
        
    Returns:
        清理后的assistant回答内容
    """
    try:
        # 方法1: 查找ChatML格式的assistant标记
        assistant_start = "<|im_start|>assistant\n"
        assistant_end = "<|im_end|>"
        
        if assistant_start in full_output:
            # 找到assistant开始位置
            start_idx = full_output.find(assistant_start) + len(assistant_start)
            
            # 查找第一个结束标记位置
            end_idx = full_output.find(assistant_end, start_idx)
            if end_idx == -1:
                # 如果没有找到结束标记，取到字符串末尾
                assistant_response = full_output[start_idx:].strip()
            else:
                assistant_response = full_output[start_idx:end_idx].strip()
            
            # 特殊处理：如果包含换行符，只取第一行（通常是SMILES）
            lines = assistant_response.split('\n')
            if lines:
                assistant_response = lines[0].strip()
            
            # 清理可能残留的特殊标记和重复内容
            cleanup_tokens = ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
            for token in cleanup_tokens:
                assistant_response = assistant_response.replace(token, "")
            
            # 移除可能的角色标记
            role_patterns = ["user", "assistant", "system"]
            for pattern in role_patterns:
                if assistant_response.startswith(pattern):
                    assistant_response = assistant_response[len(pattern):].strip()
            
            return assistant_response.strip()
        
        # 方法2: 如果没有找到ChatML格式，尝试简单的字符串替换
        else:
            # 移除输入部分
            generated_text = full_output.replace(input_text, "").strip()
            
            # 清理各种特殊标记
            cleanup_tokens = [
                "<|im_start|>", "<|im_end|>", "<|endoftext|>", 
                "system\n", "user\n", "assistant\n"
            ]
            for token in cleanup_tokens:
                generated_text = generated_text.replace(token, "")
            
            # 如果有多行，取第一行
            lines = generated_text.split('\n')
            if lines:
                generated_text = lines[0].strip()
            
            return generated_text.strip()
            
    except Exception as e:
        print(f"提取assistant回答时出错: {e}")
        # 降级到简单的字符串处理
        cleaned = full_output.replace(input_text, "").strip()
        # 至少清理明显的重复标记
        import re
        cleaned = re.sub(r'<\|im_end\|>+', '', cleaned)
        return cleaned.split('\n')[0].strip() if cleaned else ""
    

def setup_parallel_environment(config: Dict[str, Any]) -> tuple:
    """配置并行环境"""
    parallel_config = config.get('parallel', {})
    parallel_enabled = parallel_config.get('enabled', False)
    
    if not parallel_enabled:
        print("=== 单卡推理模式 ===")
        # 单卡模式：设置设备但不初始化并行环境
        device_id = int(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(',')[0]) if os.getenv('CUDA_VISIBLE_DEVICES') else 0
        try:
            ms.set_device("Ascend", device_id)
        except:
            # 如果Ascend设备不可用，尝试GPU
            try:
                ms.set_device("GPU", device_id)
            except:
                # 最后尝试CPU
                ms.set_device("CPU")
        print(f"单卡设备配置完成: device_id={device_id}")
        return 0, 1
    
    print("=== 配置并行推理环境 ===")
    # 检查必要的环境变量
    if 'RANK_ID' not in os.environ:
        raise RuntimeError("多卡推理模式需要设置RANK_ID环境变量。请使用msrun启动多卡推理。")
    
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


def generate_predictions(
    model: MultiModalQwen,
    dataset: DualRepresentationDataset,
    tokenizer: AutoTokenizer,
    evaluator: ChemicalReactionEvaluator,
    config: Dict[str, Any],
    rank_id: int = 0,
    device_num: int = 1,
    max_samples: int = None
) -> List[Dict]:
    """生成预测结果并评估 - 支持数据并行"""
    print(f"\n=== Rank {rank_id} 开始生成预测 ===")
    model.set_train(False)
    
    # 只在rank 0初始化evaluator
    if rank_id == 0:
        evaluator.reset_metrics()
    
    gen_config = config['generation']
    max_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    # 数据分片：每个进程处理不同的数据子集
    samples_per_rank = max_samples // device_num
    start_idx = rank_id * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank_id < device_num - 1 else max_samples
    
    print(f"Rank {rank_id} 处理样本范围: {start_idx}-{end_idx-1} (共{end_idx-start_idx}个样本)")

    results = []    
    for i in range(start_idx, end_idx):
        try:
            sample = dataset[i]
            original_input_text = sample['input_text']
            input_text = original_input_text
            # -----------------------------
            
            # 编码输入
            prompt_encoding = tokenizer(
                input_text,
                return_tensors="ms",
                max_length=config['data']['max_length'],
                truncation=True
            )
            
            # 准备反应物输入
            rxn_input_ids = ms.Tensor([sample['rxn_input_ids']], dtype=ms.int32)
            rxn_attention_mask = ms.Tensor([sample['rxn_attention_mask']], dtype=ms.int32)
            
            # 设置停止词 - 优先使用ChatML结束标记
            eos_token_id = tokenizer.eos_token_id  # 默认值
            
            # 检查并设置ChatML结束标记作为主要停止词
            if "<|im_end|>" in tokenizer.vocab:
                eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                print(f"使用ChatML结束标记作为停止词: {eos_token_id}")
            elif tokenizer.eos_token_id is not None:
                print(f"使用默认EOS标记作为停止词: {eos_token_id}")
            else:
                print("警告: 没有找到合适的停止词")
            
            # 生成预测
            generated_ids = model.generate(
                input_ids=prompt_encoding.input_ids,
                attention_mask=prompt_encoding.attention_mask,
                rxn_input_ids=rxn_input_ids,
                rxn_attention_mask=rxn_attention_mask,
                max_new_tokens=gen_config['max_new_tokens'],
                do_sample=gen_config['do_sample'],
                temperature=gen_config.get('temperature', 1.0),
                top_p=gen_config.get('top_p', 1.0),
                num_beams=gen_config['num_beams'],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=gen_config.get('repetition_penalty', 1.0)
            )
            
            # 解码生成结果 - 保留特殊标记以正确解析ChatML格式
            full_output = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=False
            )
            
            # 正确提取ChatML格式中的assistant回答
            generated_text = extract_assistant_response(full_output, input_text)
            
            # 提取目标SMILES
            target_text = sample['target_text']
            # 移除可能的特殊标记
            target_smiles = target_text.replace(tokenizer.eos_token, "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            
            # 评估预测
            if rank_id == 0:
                result = evaluator.add_prediction(generated_text, target_smiles)
            else:
                # 其他进程进行本地评估
                from eval.evaluation_metrics import ChemicalReactionEvaluator as LocalEvaluator
                local_evaluator = LocalEvaluator()
                result = local_evaluator.add_prediction(generated_text, target_smiles)
            
            result.update({
                'input_text': input_text,
                'generated_text': generated_text,
                'full_output': full_output,  # 保存完整输出用于调试
                'target_text': target_text,
                'target_smiles': target_smiles,
                'rank_id': rank_id
            })
            results.append(result)
            
            if (i + 1 - start_idx) % 10 == 0:
                print(f"Rank {rank_id} 已处理: {i+1-start_idx}/{end_idx-start_idx}")
                
        except Exception as e:
            print(f"生成样本 {i} 时出错: {e}")
            results.append({
                'input_text': sample.get('input_text', ''),
                'generated_text': '',
                'target_text': sample.get('target_text', ''),
                'target_smiles': '',
                'is_valid': False,
                'exact_match': False
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="全量微调模型推理脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置并行环境
    rank_id, device_num = setup_parallel_environment(config)
    
    if rank_id == 0:
        print("=== 全量微调模型推理 ===")
        print(f"配置文件: {args.config}")
        print(f"并行环境: rank {rank_id}/{device_num}")
    
    # 初始化分词器
    model_config = config['model']
    if model_config['use_extended_vocab']:
        tokenizer = AutoTokenizer.from_pretrained(model_config['extended_vocab_path'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config['qwen_model_name'])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if rank_id == 0:
        print("分词器加载完成")
    
    model_path = config['inference']['model_path']
    if rank_id == 0:
        print(f"从 {model_path} 加载全量微调模型...")
    # 加载模型
    model = MultiModalQwen.from_pretrained(model_path)
    
    model.eval()
    if rank_id == 0:
        print("全量微调模型加载完成")
    # 加载数据
    data_config = config['data']
    test_reactions = parse_reactions(data_config['test_data_path'])
    
    # 创建数据集（使用修复后的数据集构建逻辑）
    test_dataset = DualRepresentationDataset(
        reactions=test_reactions,
        qwen_tokenizer=tokenizer,
        rxn_tokenizer=get_default_tokenizer(),
        task_type=data_config['task_type'],
        max_len=data_config['max_length'],
        rxn_max_len=data_config['rxn_max_length'],
        use_cls_token=True,
    )
    
    if rank_id == 0:
        print(f"测试集样本数: {len(test_dataset)}")
    
    # 生成预测并评估
    evaluator = ChemicalReactionEvaluator()
    results = generate_predictions(
        model=model,
        dataset=test_dataset,
        tokenizer=tokenizer,
        evaluator=evaluator,
        config=config,
        rank_id=rank_id,
        device_num=device_num,
        max_samples=config['inference'].get('max_samples')
    )
    
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
            merged_evaluator = ChemicalReactionEvaluator()
            for item in all_results:
                pred = item.get('generated_text', '')
                tgt = item.get('target_smiles', '')
                merged_evaluator.add_prediction(prediction=pred, target=tgt)

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