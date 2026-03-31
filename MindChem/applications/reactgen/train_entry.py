#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化学反应预测LoRA训练统一入口程序
自动读取配置文件中的并行参数并启动相应的训练模式

功能:
- 自动检测配置文件中的并行设置
- 如果启用并行训练，使用msrun启动多卡训练
- 如果未启用并行训练，直接启动单卡训练
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
    elif '3B' in qwen_model_name:
        model_size = '3B'
    elif '7B' in qwen_model_name:
        model_size = '7B'
    else:
        model_size = 'unknown'
    
    # 提取其他参数
    task_type = config['data'].get('task_type', 'unknown')
    batch_size = config['training']['batch_size']
    learning_rate = float(config['training']['learning_rate'])  # 确保转换为float
    
    # 生成时间戳（北京时间）
    beijing_tz = ZoneInfo('Asia/Shanghai')
    timestamp = datetime.now(beijing_tz).strftime('%Y%m%d_%H%M%S')
    
    # 格式化学习率（去掉科学计数法中的e）
    lr_str = f"{learning_rate:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    
    # 根据训练模式生成不同的实验名称
    if mode == "lora":
        # LoRA模式需要包含rank信息
        lora_rank = config['lora']['rank']
        experiment_name = f"{model_size}_{task_type}_bs{batch_size}_lr{lr_str}_r{lora_rank}_{timestamp}"
    elif mode == "seq2seq":
        experiment_name = f"{model_size}_{task_type}_bs{batch_size}_lr{lr_str}_seq2seq_{timestamp}"
    else:
        # 全量微调模式
        experiment_name = f"{model_size}_{task_type}_bs{batch_size}_lr{lr_str}_full_{timestamp}"
    
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


def run_single_card_training(config_path: str, cuda_devices: str = None, mode: str = "full") -> int:
    """启动单卡训练"""
    print("\n" + "="*50)
    print(f"启动单卡{mode}训练")
    print("="*50)
    
    # 根据模式选择训练脚本
    if mode == "solvent":
        single_script = "train/train_solvent_multi.py"  # 溶剂预测使用专门的脚本
    elif mode == "lora":
        single_script = "train/train_lora_single_new.py"
        if not os.path.isfile(single_script):
            # 如果没有专门的单卡脚本，使用多卡脚本（会自动检测设备数量）
            single_script = "train/train_lora_multi.py"
    elif mode == "seq2seq":
        single_script = "train/train_full_single_seq2seq.py"
    else:  # full mode
        single_script = "train/train_full_multi.py"
    
    check_file_exists(single_script, "训练脚本")
    
    cmd = [
        sys.executable,  # 使用当前Python解释器
        single_script,
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
        print("\n训练被用户中断")
        return 1
    except Exception as e:
        print(f"执行训练时出错: {e}")
        return 1


def run_multi_card_training(config_path: str, config: dict, parallel_config: dict, cuda_devices: str = None, mode: str = "lora") -> int:
    """启动多卡训练"""
    print("\n" + "="*50)
    print(f"启动多卡训练")
    print("="*50)
    
    # 检查多卡训练脚本是否存在
    if mode == "solvent":
        multi_script = "train/train_solvent_multi.py"
    elif mode == "lora":
        multi_script = "train/train_lora_multi.py"
    elif mode == "seq2seq":
        multi_script = "train/train_full_multi_seq2seq.py"
    else:
        multi_script = "train/train_full_multi.py"
    
    check_file_exists(multi_script, "多卡训练脚本")
    
    # 输出并行配置信息
    print("并行配置:")
    worker_num = parallel_config.get('worker_num', 2)
    local_worker_num = parallel_config.get('local_worker_num', 2)
    master_addr = parallel_config.get('master_addr', '127.0.0.1')
    master_port = parallel_config.get('master_port', 8118)
    log_dir = parallel_config.get('log_dir', './logs/forward/lora_multi_logs')

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
        print("\n训练被用户中断")
        return 1
    except Exception as e:
        print(f"执行多卡训练时出错: {e}")
        return 1


def main():
    """主函数"""
    # 添加项目根目录到Python路径，确保import始终以项目路径为基础
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    parser = argparse.ArgumentParser(
        description="化学反应预测LoRA训练统一入口程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例用法:
  python train_entry.py --config config/forward_1.5b_training.yaml --cuda-visible-devices 0,2,4,6 --run-suffix gpu_0246
  
注意:
  - 程序会自动检测配置文件中的parallel.enabled设置
  - 如果enabled=true，启动多卡训练（需要msrun）
  - 如果enabled=false或未设置，启动单卡训练
  - 使用--run-suffix可以为输出路径添加后缀，避免多次运行时覆盖
  - 使用--cuda-visible-devices可以指定要使用的GPU设备，支持GPU环境下的设备选择
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
        choices=["lora","full","solvent","seq2seq"],
        default=None,
        help="选择训练模式: lora、full（全量微调）或 solvent（溶剂预测）。如果不指定，将自动从配置文件的data.task_type中提取"
    )
    
    parser.add_argument(
        "--force-single",
        action="store_true",
        help="强制使用单卡训练模式，忽略配置文件中的并行设置"
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
    
    # 自动提取训练模式
    if args.mode is None:
        # 从配置文件中提取task_type作为mode
        task_type = config.get('data', {}).get('task_type', 'full')
        mode = task_type
        print(f"自动从配置文件提取训练模式: {mode}")
    else:
        mode = args.mode
        print(f"使用指定的训练模式: {mode}")
    
    # 验证mode是否有效
    valid_modes = ["lora", "full", "solvent", "seq2seq"]
    if mode not in valid_modes:
        print(f"错误: 无效的训练模式 '{mode}'，支持的模式: {valid_modes}")
        print(f"请检查配置文件中的data.task_type设置或使用--mode参数指定有效模式")
        sys.exit(1)
    
    # 检查并行配置
    parallel_config = config.get('parallel', {})
    parallel_enabled = parallel_config.get('enabled', False)
    
    if args.force_single:
        print("强制使用单卡训练模式")
        parallel_enabled = False
    
    # 获取CUDA设备配置
    cuda_devices = getattr(args, 'cuda_visible_devices', None)
    if cuda_devices:
        print(f"指定CUDA设备: {cuda_devices}")
    
    # 显示训练模式
    if parallel_enabled:
        print("检测到并行训练配置，将启动多卡训练")
        if args.dry_run:
            print("[DRY RUN] 多卡训练命令预览:")
            run_multi_card_training(config_path, config, parallel_config, cuda_devices, mode)
            return 0
        else:
            return run_multi_card_training(config_path, config, parallel_config, cuda_devices, mode)
    else:
        print("并行训练未启用，将启动单卡训练")
        if args.dry_run:
            print("[DRY RUN] 单卡训练命令预览:")
            run_single_card_training(config_path, cuda_devices, mode)
            return 0
        else:
            return run_single_card_training(config_path, cuda_devices, mode)


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