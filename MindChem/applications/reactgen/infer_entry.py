#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化学反应预测推理统一入口程序
自动读取配置文件中的并行参数并启动相应的推理模式

功能:
- 自动检测配置文件中的并行设置
- 如果启用并行推理，使用msrun启动多卡推理
- 如果未启用并行推理，直接启动单卡推理
- 支持LoRA和全量微调模型的推理
"""

import os
import sys
import yaml
import argparse
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any


def generate_experiment_name(config: Dict[str, Any], mode: str = "lora") -> str:
    """根据配置生成实验名称"""
    # 提取模型大小
    qwen_model_name = config['model']['qwen_model_name']
    if '0.5B' in qwen_model_name:
        model_size = '0.5B'
    elif '1.5B' in qwen_model_name:
        model_size = '1.5B'
    elif '7B' in qwen_model_name:
        model_size = '7B'
    else:
        model_size = 'unknown'
    
    # 提取其他参数
    task_type = config['data']['task_type']
    
    # 生成时间戳（北京时间）
    beijing_tz = ZoneInfo('Asia/Shanghai')
    timestamp = datetime.now(beijing_tz).strftime('%Y%m%d_%H%M%S')
    
    # 根据推理模式生成不同的实验名称
    if mode == "lora":
        experiment_name = f"{model_size}_{task_type}_lora_inference_{timestamp}"
    elif mode == "solvent":
        experiment_name = f"{model_size}_{task_type}_solvent_inference_{timestamp}"
    elif mode == "seq2seq":
        experiment_name = f"{model_size}_{task_type}_seq2seq_inference_{timestamp}"
    else:
        # 全量微调模式
        experiment_name = f"{model_size}_{task_type}_full_inference_{timestamp}"
    
    return experiment_name


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 配置文件格式错误: {e}")
        sys.exit(1)


def check_file_exists(file_path: str, description: str) -> None:
    """检查文件是否存在"""
    if not os.path.isfile(file_path):
        print(f"错误: {description}不存在: {file_path}")
        sys.exit(1)


def detect_inference_mode(config: Dict[str, Any]) -> str:
    """检测推理模式（LoRA或全量微调）"""
    inference_config = config.get('inference', {})
    
    # 检查是否有LoRA相关配置
    if 'lora_model_path' in inference_config:
        return "lora"
    elif 'solvent_model_path' in inference_config:
        return "solvent"
    elif 'model_path' in inference_config:
        return "full"
    else:
        print("错误: 配置文件中未找到有效的模型路径配置")
        print("请确保配置文件中包含 'lora_model_path' 或 'model_path'")
        sys.exit(1)


def run_single_card_inference(config_path: str, mode: str, cuda_devices: str = None) -> int:
    """启动单卡推理"""
    print("\n" + "="*50)
    print(f"启动单卡{mode.upper()}推理")
    print("="*50)
    
    # 选择推理脚本
    if mode == "lora":
        inference_script = "inference/inference_lora.py"
    elif mode == "solvent":
        inference_script = "inference/inference_solvent_multi.py"
    elif mode == "seq2seq":
        inference_script = "inference/inference_full_multi_seq2seq.py"
    else:
        # 检查是否有单卡全量推理脚本，如果没有则使用多卡脚本
        single_full_script = "inference/inference_full.py"
        if os.path.isfile(single_full_script):
            inference_script = single_full_script
        else:
            inference_script = "inference/inference_full_multi.py"
    
    check_file_exists(inference_script, "推理脚本")
    
    cmd = [
        sys.executable,  # 使用当前Python解释器
        inference_script,
        "--config", os.path.abspath(config_path)
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    if cuda_devices:
        env['CUDA_VISIBLE_DEVICES'] = cuda_devices
        print(f"设置CUDA_VISIBLE_DEVICES={cuda_devices}")
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\n推理被用户中断")
        return 1
    except Exception as e:
        print(f"执行推理时出错: {e}")
        return 1


def run_multi_card_inference(config_path: str, config: dict, parallel_config: dict, mode: str, cuda_devices: str = None) -> int:
    """启动多卡推理"""
    print("\n" + "="*50)
    print(f"启动多卡{mode.upper()}推理")
    print("="*50)
    
    # 选择推理脚本
    if mode == "lora":
        multi_script = "inference/inference_lora_multi.py"
    elif mode == "solvent":
        multi_script = "inference/inference_solvent_multi.py"
    elif mode == "seq2seq":
        multi_script = "inference/inference_full_multi_seq2seq.py"
    else:
        multi_script = "inference/inference_full_multi.py"
    
    check_file_exists(multi_script, "多卡推理脚本")
    
    # 输出并行配置信息
    print("并行配置:")
    worker_num = parallel_config.get('worker_num', 2)
    local_worker_num = parallel_config.get('local_worker_num', 2)
    master_addr = parallel_config.get('master_addr', '127.0.0.1')
    master_port = parallel_config.get('master_port', 8119)
    log_dir = parallel_config.get('log_dir', './logs/forward/inference_logs')

    # 生成实验名称并附加到log_dir
    experiment_name = generate_experiment_name(config, mode)
    log_dir = os.path.join(log_dir, experiment_name)
    
    print(f"  worker_num: {worker_num}")
    print(f"  local_worker_num: {local_worker_num}")
    print(f"  master_addr: {master_addr}")
    print(f"  master_port: {master_port}")
    print(f"  log_dir: {log_dir}")
    print(f"  experiment_name: {experiment_name}")
    
    # 构建msrun命令
    cmd = [
        'msrun',
        f'--worker_num={worker_num}',
        f'--local_worker_num={local_worker_num}',
        f'--master_addr={master_addr}',
        f'--master_port={master_port}',
        f'--log_dir={log_dir}',
        '--join=True',
        multi_script,
        f'--config={os.path.abspath(config_path)}'
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    if cuda_devices:
        env['CUDA_VISIBLE_DEVICES'] = cuda_devices
        print(f"设置CUDA_VISIBLE_DEVICES={cuda_devices}")
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except FileNotFoundError:
        print("错误: 找不到msrun命令，请确保MindSpore分布式训练环境已正确安装")
        return 1
    except KeyboardInterrupt:
        print("\n推理被用户中断")
        return 1
    except Exception as e:
        print(f"执行多卡推理时出错: {e}")
        return 1


def main():
    """主函数"""
    # 添加项目根目录到Python路径，确保import始终以项目路径为基础
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    parser = argparse.ArgumentParser(
        description="化学反应预测推理统一入口程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例用法:
  python infer_entry.py --config config/forward_1.5b_inference.yaml --cuda-visible-devices 0,1
  
注意:
  - 程序会自动检测配置文件中的parallel.enabled设置
  - 如果enabled=true，启动多卡推理（需要msrun）
  - 如果enabled=false或未设置，启动单卡推理
  - 程序会自动检测推理模式（LoRA或全量微调）
  - 使用--cuda-visible-devices可以指定要使用的GPU设备
  - 通过修改配置文件中的parallel.enabled来控制单卡/多卡推理
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="配置文件路径（YAML格式）"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full", "auto","solvent","seq2seq"],
        default="auto",
        help="选择推理模式: lora、full（全量微调）或 auto（自动检测）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要执行的命令，不实际执行"
    )
    
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        help="指定可见的CUDA设备，例如: 0,1,2,3 或 0,2,4,6。如果不指定，则使用系统默认设置"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 加载配置文件
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 检测推理模式
    if args.mode == "auto":
        mode = detect_inference_mode(config)
        print(f"自动检测推理模式: {mode.upper()}")
    else:
        mode = args.mode
        print(f"指定推理模式: {mode.upper()}")
    
    # 检查并行配置
    parallel_config = config.get('parallel', {})
    parallel_enabled = parallel_config.get('enabled', False)
    
    # 获取CUDA设备配置
    cuda_devices = getattr(args, 'cuda_visible_devices', None)
    if cuda_devices:
        print(f"指定CUDA设备: {cuda_devices}")
    
    # 显示推理模式
    if parallel_enabled:
        print("检测到并行推理配置，将启动多卡推理")
        if args.dry_run:
            print("[DRY RUN] 多卡推理命令预览:")
            run_multi_card_inference(config_path, config, parallel_config, mode, cuda_devices)
            return 0
        else:
            return run_multi_card_inference(config_path, config, parallel_config, mode, cuda_devices)
    else:
        print("并行推理未启用，将启动单卡推理")
        if args.dry_run:
            print("[DRY RUN] 单卡推理命令预览:")
            run_single_card_inference(config_path, mode, cuda_devices)
            return 0
        else:
            return run_single_card_inference(config_path, mode, cuda_devices)


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)