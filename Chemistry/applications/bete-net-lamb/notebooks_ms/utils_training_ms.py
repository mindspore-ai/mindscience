# MindSpore & Sharker 版本的训练工具

# 设置导入路径
import setup_paths

import mindspore as ms
import mindspore.nn as nn
from sharker.loader import Dataloader as DataLoader
from mindchemistry.e3 import o3
import numpy as np
from tqdm import tqdm
# 其他必要的mindspore和sharker导入

# 这里实现get_model, train等函数，接口与原training.py一致
# 具体实现需参考原training.py和mindspore API

def get_model(init_dict):
    """
    使用MindSpore和MindChemistry.e3构建模型
    
    参数:
    - init_dict: 包含模型初始化参数的字典
    
    返回:
    - Network: 初始化好的模型实例
    """
    import utils_model_ms
    
    model = utils_model_ms.Network(
        irreps_in=init_dict['irreps_in'],
        irreps_out=init_dict['irreps_out'],
        irreps_node_attr=init_dict['irreps_node_attr'],
        layers=init_dict['layers'],
        mul=init_dict['mul'],
        lmax=init_dict['lmax'],
        max_radius=init_dict['max_radius'],
        num_neighbors=init_dict['num_neighbors'],
        reduce_output=init_dict['reduce_output'],
        p=init_dict['p']
    )
    return model

def train(model, dataloader, loss_fn, optimizer, max_iter, device=None):
    """
    使用MindSpore训练模型
    
    参数:
    - model: 要训练的模型
    - dataloader: 数据加载器
    - loss_fn: 损失函数
    - optimizer: 优化器
    - max_iter: 最大迭代次数
    - device: 训练设备
    
    返回:
    - 训练后的模型
    """
    model.set_train()
    
    for epoch in range(max_iter):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{max_iter}'):
            # 前向传播
            pred = model(batch)
            target = batch.target
            
            # 计算损失
            loss = loss_fn(pred, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.asnumpy()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{max_iter}, Average Loss: {avg_loss:.4f}')
    
    return model

def evaluate(model, dataloader, loss_fn, device=None):
    """
    评估模型性能
    
    参数:
    - model: 要评估的模型
    - dataloader: 数据加载器
    - loss_fn: 损失函数
    - device: 评估设备
    
    返回:
    - 平均损失值
    """
    model.set_eval()
    total_loss = 0
    
    for batch in dataloader:
        pred = model(batch)
        target = batch.target
        loss = loss_fn(pred, target)
        total_loss += loss.asnumpy()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def predict(model, dataloader, device=None):
    """
    使用模型进行预测
    
    参数:
    - model: 训练好的模型
    - dataloader: 数据加载器
    - device: 预测设备
    
    返回:
    - 预测结果列表
    """
    model.set_eval()
    predictions = []
    
    for batch in dataloader:
        pred = model(batch)
        predictions.append(pred.asnumpy())
    
    return np.concatenate(predictions, axis=0) 