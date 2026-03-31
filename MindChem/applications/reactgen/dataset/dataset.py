# -*- coding: utf-8 -*-
"""
双重表示输入数据集
支持正向反应预测、逆合成预测下游任务
基于双重表示输入策略：同时使用原始SMILES和哨兵令牌

更新说明：
- 移除了对sentinel_positions的依赖，与reactionqwen.py的新逻辑同步
- 新的模型基于哨兵token ID进行逐元素替换，不再需要位置信息
- 简化了数据集接口，提高了梯度回传的兼容性
"""

import numpy as np
from mindnlp.transformers import AutoTokenizer
from typing import List, Tuple


class DualRepresentationDataset:
    """
    双重表示输入数据集，支持化学反应预测任务
    
    支持的任务类型：
    - forward: 正向反应预测 (reactants -> products)
    - retrosynthesis: 逆合成预测 (products -> reactants)
    """
    
    # 任务模板定义 - 使用ChatML格式，支持中英文
    TASK_TEMPLATES = {
        'en': {
            'forward': {
                'system_message': "You are a helpful chemical reaction prediction assistant. You can understand chemical structures represented by sentinel tokens and predict reaction products.",
                'user_template': "The reactants {input_smiles} can be represented as {sentinel_tokens}. Predict the product.",
                'description': "正向反应预测：根据反应物预测产物"
            },
            'retrosynthesis': {
                'system_message': "You are a helpful chemical reaction prediction assistant. You can understand chemical structures represented by sentinel tokens and predict reactants for retrosynthesis.",
                'user_template': "The product {input_smiles} can be represented as {sentinel_tokens}. Predict the reactants.",
                'description': "逆合成预测：根据产物预测反应物"
            }
        },
        'zh': {
            'forward': {
                'system_message': "你是化学助手，擅长理解化学结构并预测产物。",
                'user_template': "反应物{input_smiles}可表示为{sentinel_tokens}，请预测产物。",
                'description': "正向反应预测：根据反应物预测产物"
            },
            'retrosynthesis': {
                'system_message': "你是化学助手，擅长理解化学结构并预测逆合成反应物。",
                'user_template': "产物{input_smiles}可表示为{sentinel_tokens}，请预测反应物。",
                'description': "逆合成预测：根据产物预测反应物"
            }
        }
    }

    
    def __init__(self, 
                 reactions: List[Tuple[str, str]], 
                 qwen_tokenizer: AutoTokenizer,
                 rxn_tokenizer,
                 task_type: str = 'forward',
                 language: str = 'zh',
                 sentinel_token: str = '<SMI_TOKEN>',
                 max_len: int = 512,
                 rxn_max_len: int = 256,
                 use_cls_token: bool = True):
        """
        初始化数据集
        
        Args:
            reactions: 反应数据列表 [(reactants, products), ...]
            qwen_tokenizer: Qwen分词器
            rxn_tokenizer: ReactionBert分词器
            task_type: 任务类型 ('forward', 'retrosynthesis')
            language: 模板语言 ('en', 'zh')
            sentinel_token: 哨兵令牌
            max_len: 最大序列长度
            rxn_max_len: ReactionBert最大序列长度
        """
        self.reactions = reactions
        self.qwen_tokenizer = qwen_tokenizer
        self.rxn_tokenizer = rxn_tokenizer
        self.task_type = task_type
        self.language = language
        self.sentinel_token = sentinel_token
        self.max_len = max_len
        self.rxn_max_len = rxn_max_len
        self.use_cls_token = use_cls_token
        
        # 验证语言和任务类型
        if language not in self.TASK_TEMPLATES:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.TASK_TEMPLATES.keys())}")
        if task_type not in self.TASK_TEMPLATES[language]:
            raise ValueError(f"Unsupported task_type: {task_type}. Supported: {list(self.TASK_TEMPLATES[language].keys())}")
        
        # 确保哨兵令牌在词表中
        if sentinel_token not in qwen_tokenizer.vocab:
            qwen_tokenizer.add_tokens([sentinel_token])
            print(f"Added sentinel token '{sentinel_token}' to vocabulary")
        
        # 设置pad_token
        if qwen_tokenizer.pad_token is None:
            qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
            
        self.template_info = self.TASK_TEMPLATES[language][task_type]
        print(f"Initialized dataset for task: {self.template_info['description']} (Language: {language})")
        print(f"System message: {self.template_info['system_message']}")
        print(f"User template: {self.template_info['user_template']}")
    
    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, idx):
        """获取单个数据样本 - 使用ChatML格式"""
        reaction_data = self.reactions[idx]
        
        # 根据任务类型解析反应数据
        input_smiles, target_smiles = self._parse_reaction_data(reaction_data)
        
        # 创建双重表示输入（包含system和user消息）
        dual_input = self._create_dual_representation_input(input_smiles)
        
        # 准备assistant响应（目标输出）
        # 不手动添加eos_token，让apply_chat_template自动处理
        target_output = target_smiles
        
        # 创建完整的消息，包括assistant响应
        full_messages = dual_input['messages'] + [
            {"role": "assistant", "content": target_output}
        ]
        
        # 使用apply_chat_template生成完整的训练文本
        full_text = self.qwen_tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False  # 不添加生成提示，因为我们已有assistant响应
        )
        
        # 对完整文本进行分词
        full_encoding = self.qwen_tokenizer(full_text, return_tensors="np")
        full_input_ids = full_encoding['input_ids'][0]
        
        # 计算prompt长度（直接使用input_text的长度）
        input_encoding = self.qwen_tokenizer(dual_input['input_text'], return_tensors="np")
        prompt_len = len(input_encoding['input_ids'][0])
        
        # 创建标签：prompt部分为-100，目标部分为对应的token id
        labels = [-100] * prompt_len + full_input_ids[prompt_len:].tolist()
        
        # 组合输入和标签用于训练
        combined_ids = full_input_ids.tolist()
        
        # 截断到最大长度
        combined_ids = combined_ids[:self.max_len]
        labels = labels[:self.max_len]
        attention_mask = [1] * len(combined_ids)
        
        # 填充到固定长度
        padding_len = self.max_len - len(combined_ids)
        input_ids = combined_ids + [self.qwen_tokenizer.pad_token_id] * padding_len
        attention_mask = attention_mask + [0] * padding_len
        labels = labels + [-100] * padding_len
        
        # 处理rxn数据 - 现在都是numpy格式
        rxn_input_ids = dual_input['rxn_input_ids'][0].astype(np.int32)
        rxn_attention_mask = dual_input['rxn_attention_mask'][0].astype(np.int32)
        
        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "rxn_input_ids": rxn_input_ids,
            "rxn_attention_mask": rxn_attention_mask,
            # 额外信息，用于调试和分析
            "input_text": dual_input['input_text'],
            "target_text": target_output,
            "full_text": full_text,  # 完整的ChatML格式文本
            "task_type": self.task_type
        }
    
    def _parse_reaction_data(self, reaction_data):
        """根据任务类型解析反应数据"""
        if self.task_type == 'forward':
            reactants, products = reaction_data
            return reactants, products
        elif self.task_type == 'retrosynthesis':
            reactants, products = reaction_data
            return products, reactants  # 输入产物，预测反应物
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
    
    def _create_dual_representation_input(self, input_smiles: str):
        """
        创建双重表示输入：使用ChatML格式同时包含原始SMILES和哨兵令牌
        """
        # 1. 使用ReactionBert分词器处理SMILES - 使用numpy格式避免tensor运算
        rxn_encoding = self.rxn_tokenizer(
            input_smiles, 
            padding="max_length", 
            truncation=True, 
            max_length=self.rxn_max_len, 
            return_tensors="np"
        )
        
        if self.use_cls_token:
            num_sentinel_tokens = 1
        else:
            # 2. 计算需要的哨兵令牌数量（排除特殊令牌）
            actual_length = int(rxn_encoding['attention_mask'].sum())
            num_sentinel_tokens = max(1, actual_length - 2)  # 减去[CLS]和[SEP]，至少保留1个
        
        # 3. 构建哨兵令牌序列
        sentinel_sequence = self.sentinel_token * num_sentinel_tokens
        
        # 4. 构建用户消息内容
        user_content = self.template_info['user_template'].format(
            input_smiles=input_smiles,
            sentinel_tokens=sentinel_sequence
        )
        
        # 5. 创建ChatML格式的消息
        messages = [
            {"role": "system", "content": self.template_info['system_message']},
            {"role": "user", "content": user_content}
        ]
        
        # 6. 使用apply_chat_template生成格式化文本 - 使用numpy格式
        input_text = self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 7. 分词和编码
        input_encoding = self.qwen_tokenizer(input_text, return_tensors="np")
        
        return {
            'input_text': input_text,
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'rxn_input_ids': rxn_encoding['input_ids'],
            'rxn_attention_mask': rxn_encoding['attention_mask'],
            'messages': messages  # 保存消息结构用于调试
        }
    
    
    def get_task_info(self):
        """获取当前任务信息"""
        return {
            'task_type': self.task_type,
            'language': self.language,
            'description': self.template_info['description'],
            'system_message': self.template_info['system_message'],
            'user_template': self.template_info['user_template'],
            'dataset_size': len(self.reactions)
        }


def parse_reactions(file_path: str) -> List[Tuple[str, str]]:
    """
    解析反应文件
    
    Args:
        file_path: 反应数据文件路径，格式为 'reactants>>products'
        
    Returns:
        反应数据列表 [(reactants, products), ...]
    """
    reactions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if '>>' in line:
                parts = line.split('>>', 1)
                if len(parts) == 2:
                    reactants, products = parts
                    if reactants.strip() and products.strip():
                        reactions.append((reactants.strip(), products.strip()))
                    else:
                        print(f"Warning: Empty reactants or products at line {line_num}")
                else:
                    print(f"Warning: Invalid format at line {line_num}: {line}")
            elif line:  # 非空行但格式不对
                print(f"Warning: No '>>' separator found at line {line_num}: {line}")
    
    print(f"Parsed {len(reactions)} valid reactions from {file_path}")
    return reactions


def parse_solvent_reactions(file_path: str) -> List[Tuple[str, str, str]]:
    """
    解析包含溶剂信息的反应文件
    注意：溶剂预测任务已移除，此函数保留用于兼容性
    
    Args:
        file_path: 反应数据文件路径，格式为 'reactants>>products.solvents'
        
    Returns:
        反应数据列表 [(reactants, products, solvents), ...]
    """
    print("Warning: 溶剂预测任务已移除，建议使用parse_reactions()函数")
    reactions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if '>>' in line:
                reactants, products_solvents = line.split('>>', 1)
                
                # 尝试分离产物和溶剂
                if '.' in products_solvents:
                    # 假设最后一个分子是溶剂
                    parts = products_solvents.split('.')
                    products = '.'.join(parts[:-1])
                    solvents = parts[-1]
                else:
                    products = products_solvents
                    solvents = ""  # 无溶剂信息
                
                if reactants.strip() and products.strip():
                    reactions.append((reactants.strip(), products.strip(), solvents.strip()))
                else:
                    print(f"Warning: Empty reactants or products at line {line_num}")
            elif line:
                print(f"Warning: No '>>' separator found at line {line_num}: {line}")
    
    print(f"Parsed {len(reactions)} valid reactions with solvent info from {file_path}")
    return reactions


# 使用示例和测试函数
def create_sample_data():
    """创建示例数据用于测试"""
    # 正向反应预测示例
    forward_reactions = [
        ("C1=CC=C1.C=C", "C1C=CC2C1C2"),  # Diels-Alder反应
        ("CCO.CC(=O)Cl", "CC(=O)OCC"),    # 酯化反应
        ("c1ccccc1.Cl2", "c1ccc(Cl)cc1"), # 氯化反应
    ]
    
    return forward_reactions


def test_chatml_format():
    """测试ChatML格式数据集"""
    print("=== 测试ChatML格式数据集 ===\n")
    
    # 创建示例数据
    reactions = create_sample_data()
    print(f"测试数据: {reactions}\n")
    
    # 创建简单的分词器用于测试（不依赖ReactionBERT）
    class MockRxnTokenizer:
        def __call__(self, text, padding=None, truncation=None, max_length=None, return_tensors=None):
            # 简单的mock分词器，返回固定长度
            import numpy as np
            seq_len = max_length if max_length else 256
            # 模拟有效长度为10（包括CLS和SEP）
            input_ids = [1] + [100 + i for i in range(8)] + [2] + [0] * (seq_len - 10)
            attention_mask = [1] * 10 + [0] * (seq_len - 10)
            
            if return_tensors == "np":
                return {
                    'input_ids': np.array([input_ids]),
                    'attention_mask': np.array([attention_mask])
                }
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask]
            }
    
    # 尝试使用Qwen分词器
    try:
        qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        print("✓ 使用官方Qwen2.5-7B-Instruct分词器")
    except:
        try:
            qwen_tokenizer = AutoTokenizer.from_pretrained("./extended_qwen_tokenizer")
            print("✓ 使用本地扩展Qwen分词器")
        except:
            print("⚠ 无法加载Qwen分词器，跳过测试")
            return
    
    # 确保有pad_token
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    
    # 添加哨兵令牌
    sentinel_token = '<SMI_TOKEN>'
    if sentinel_token not in qwen_tokenizer.vocab:
        qwen_tokenizer.add_tokens([sentinel_token])
        print(f"✓ 添加哨兵令牌: {sentinel_token}")
    
    rxn_tokenizer = MockRxnTokenizer()
    
    # 测试中英文模板和两种任务类型
    for language in ['en', 'zh']:
        for task_type in ['forward', 'retrosynthesis']:
            print(f"\n--- 测试任务: {task_type} ({language}) ---")
            
            try:
                # 创建数据集
                dataset = DualRepresentationDataset(
                    reactions=reactions,
                    qwen_tokenizer=qwen_tokenizer,
                    rxn_tokenizer=rxn_tokenizer,
                    task_type=task_type,
                    language=language,
                    max_len=512,
                    rxn_max_len=256
                )
                
                # 获取任务信息
                task_info = dataset.get_task_info()
                print(f"任务描述: {task_info['description']}")
                print(f"系统消息: {task_info['system_message']}")
                print(f"用户模板: {task_info['user_template']}")
                
                # 测试第一个样本
                if len(dataset) > 0:
                    sample = dataset[0]
                    
                    print(f"\n=== 样本详细信息 ===")
                    print(f"输入文本: {sample['input_text'][:200]}...")
                    print(f"目标文本: {sample['target_text']}")
                    print(f"完整文本预览:")
                    print("-" * 50)
                    print(sample['full_text'][:500] + "..." if len(sample['full_text']) > 500 else sample['full_text'])
                    print("-" * 50)
                    
                    # 检查ChatML标记
                    full_text = sample['full_text']
                    has_im_start = '<|im_start|>' in full_text
                    has_im_end = '<|im_end|>' in full_text
                    has_system = 'system\n' in full_text
                    has_user = 'user\n' in full_text
                    has_assistant = 'assistant\n' in full_text
                    
                    print(f"\nChatML格式检查:")
                    print(f"✓ 包含<|im_start|>: {has_im_start}")
                    print(f"✓ 包含<|im_end|>: {has_im_end}")
                    print(f"✓ 包含system角色: {has_system}")
                    print(f"✓ 包含user角色: {has_user}")
                    print(f"✓ 包含assistant角色: {has_assistant}")
                    
                    # 检查数据维度
                    print(f"\n数据维度:")
                    print(f"input_ids形状: {sample['input_ids'].shape}")
                    print(f"labels形状: {sample['labels'].shape}")
                    print(f"rxn_input_ids形状: {sample['rxn_input_ids'].shape}")
                    
                    # 检查哨兵令牌
                    sentinel_token_id = qwen_tokenizer.convert_tokens_to_ids(sentinel_token)
                    sentinel_count = (sample['input_ids'] == sentinel_token_id).sum()
                    print(f"哨兵令牌数量: {sentinel_count}")
                    
                    if has_im_start and has_im_end and has_system and has_user and has_assistant:
                        print("✅ ChatML格式正确!")
                    else:
                        print("❌ ChatML格式不完整!")
                        
            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 运行ChatML格式测试
    test_chatml_format()