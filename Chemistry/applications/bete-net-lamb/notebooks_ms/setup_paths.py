"""
设置MindChemistry和Sharker的导入路径
"""

import sys
import os

def setup_paths():
    """
    将mindchemistry和sharker添加到Python路径中
    """
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 添加MindChemistry路径
    mindchemistry_path = os.path.join(parent_dir, 'mindscience', 'MindChemistry')
    if os.path.exists(mindchemistry_path) and mindchemistry_path not in sys.path:
        sys.path.insert(0, mindchemistry_path)
        print(f"Added MindChemistry path: {mindchemistry_path}")
    
    # 添加Sharker路径
    sharker_path = os.path.join(parent_dir, 'sharker')
    if os.path.exists(sharker_path) and sharker_path not in sys.path:
        sys.path.insert(0, sharker_path)
        print(f"Added Sharker path: {sharker_path}")

# 自动执行路径设置
setup_paths() 