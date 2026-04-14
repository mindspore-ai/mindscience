#!/usr/bin/env python3
"""
BETE-NET 可配置训练启动脚本
支持自定义batch_size和其他训练参数
"""

import argparse
import sys
import os
sys.path.append('./notebooks_ms')
from run_full_training import full_training

def main():
    parser = argparse.ArgumentParser(description='BETE-NET Training with Configurable Batch Size')
    
    # 模型配置
    parser.add_argument('--config', type=str, default='FPD', 
                       choices=['CSO', 'CPD', 'FPD'],
                       help='Model configuration (CSO/CPD/FPD)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs (default: 100)')
    parser.add_argument('--display_interval', type=int, default=5,
                       help='Display progress every N epochs (default: 5)')
    parser.add_argument('--plot_interval', type=int, default=5,
                       help='Generate plots every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    print(f"🚀 BETE-NET Training Configuration")
    print(f"=" * 50)
    print(f"Model Configuration: {args.config}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Display Interval: {args.display_interval}")
    print(f"Plot Interval: {args.plot_interval}")
    print(f"=" * 50)
    
    # 根据模型配置推荐batch_size
    recommendations = {
        'CSO': {'batch_size': 32, 'reason': '轻量模型，可用较大batch'},
        'CPD': {'batch_size': 32, 'reason': '中等模型，平衡性能和内存'},
        'FPD': {'batch_size': 16, 'reason': '大模型，建议较小batch避免内存溢出'}
    }
    
    rec = recommendations[args.config]
    if args.batch_size != rec['batch_size']:
        print(f"💡 建议: {args.config}模型推荐batch_size={rec['batch_size']} ({rec['reason']})")
        print(f"   当前设置: {args.batch_size}")
        
        confirm = input("是否继续使用当前设置? (y/n): ")
        if confirm.lower() != 'y':
            print("训练已取消")
            return
    
    print(f"\n🎯 开始训练...")
    
    try:
        for i in range(1,10):
            print(f"============ {i} ==============")
            results = full_training(
                config_name=args.config,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                display_interval=args.display_interval,
                plot_interval=args.plot_interval,
                idx = i
            )
        print(f"\n✅ 训练完成!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 