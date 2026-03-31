# -*- coding: utf-8 -*-
"""
溶剂预测任务数据集

核心功能:
1.  **解析多标签数据**: 读取包含反应、溶剂和溶剂标签的数据集。
2.  **双重表示输入**: 沿用`DualRepresentationDataset`的核心思想，为反应物创建双重表示输入。
3.  **多任务标签**:
    -   **溶剂种类**: 生成多热编码（multi-hot）的溶剂标签向量。
    -   **溶剂数量**: 生成预测溶剂数量的标签。
4.  **ChatML格式**: 输入文本遵循ChatML格式，与Qwen模型兼容。
"""

import numpy as np
from mindnlp.transformers import AutoTokenizer
from typing import List, Tuple, Dict

NUM_SOLVENTS = 52 # 0-51, 0 for no solvent
MAX_SOLVENT_COUNT = 3 # 0, 1, 2, 3

class SolventPredictionDataset:
    """
    用于溶剂预测任务的数据集，支持多标签分类和数量预测。
    """
    TASK_TEMPLATES = {
        'en': {
            'system_message': "You are a helpful chemical reaction assistant. You can understand chemical structures and predict the solvents used in a reaction.",
            'user_template': "The reactants {input_smiles} can be represented as {sentinel_tokens}. Predict the solvents and their count.",
            'description': "Solvent Prediction: Predict solvents and their count based on reactants."
        },
        'zh': {
            'system_message': "你是化学助手，擅长根据反应物预测反应中使用的溶剂。",
            'user_template': "反应物{input_smiles}可表示为{sentinel_tokens}，请预测所使用的溶剂及其数量。",
            'description': "溶剂预测：根据反应物预测溶剂及其数量。"
        }
    }

    def __init__(self,
                 data: List[Dict],
                 qwen_tokenizer: AutoTokenizer,
                 rxn_tokenizer,
                 language: str = 'zh',
                 sentinel_token: str = '<SMI_TOKEN>',
                 max_len: int = 512,
                 rxn_max_len: int = 256,
                 use_cls_token: bool = True):
        """
        初始化数据集

        Args:
            data: 数据列表，每个元素是一个字典，包含'reaction', 'solvents', 'solvent_labels'
            qwen_tokenizer: Qwen分词器
            rxn_tokenizer: ReactionBert分词器
            language: 模板语言 ('en', 'zh')
            sentinel_token: 哨兵令牌
            max_len: Qwen输入最大长度
            rxn_max_len: ReactionBert输入最大长度
            use_cls_token: 是否使用[CLS] token的表示
        """
        self.data = data
        self.qwen_tokenizer = qwen_tokenizer
        self.rxn_tokenizer = rxn_tokenizer
        self.language = language
        self.sentinel_token = sentinel_token
        self.max_len = max_len
        self.rxn_max_len = rxn_max_len
        self.use_cls_token = use_cls_token

        if language not in self.TASK_TEMPLATES:
            raise ValueError(f"Unsupported language: {language}")

        if sentinel_token not in qwen_tokenizer.vocab:
            qwen_tokenizer.add_tokens([sentinel_token])

        if qwen_tokenizer.pad_token is None:
            qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

        self.template_info = self.TASK_TEMPLATES[language]
        print(f"Initialized dataset for task: {self.template_info['description']} (Language: {language})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个数据样本"""
        item = self.data[idx]
        reactants = item['reaction'].split('>>')[0]

        # 1. 创建双重表示输入
        dual_input = self._create_dual_representation_input(reactants)

        # 2. 创建标签
        # 溶剂多标签
        solvent_labels_multi_hot = np.zeros(NUM_SOLVENTS, dtype=np.float32)
        solvent_indices = [int(label) for label in item['solvent_labels']]
        if solvent_indices:
            solvent_labels_multi_hot[solvent_indices] = 1.0

        # 溶剂数量标签
        solvent_count = len(item['solvents'])
        solvent_count_label = min(solvent_count, MAX_SOLVENT_COUNT) # 将数量限制在范围内

        # 3. 填充和截断
        input_ids = dual_input['input_ids'][0]
        attention_mask = dual_input['attention_mask'][0]
        
        return {
            "input_ids": input_ids.astype(np.int32),
            "attention_mask": attention_mask.astype(np.int32),
            "rxn_input_ids": dual_input['rxn_input_ids'][0].astype(np.int32),
            "rxn_attention_mask": dual_input['rxn_attention_mask'][0].astype(np.int32),
            "solvent_labels": solvent_labels_multi_hot,
            "solvent_count": np.int32(solvent_count_label),
            # 调试信息
            "reactants": reactants,
        }

    def _create_dual_representation_input(self, input_smiles: str) -> Dict:
        """创建双重表示输入"""
        rxn_encoding = self.rxn_tokenizer(
            input_smiles,
            padding="max_length",
            truncation=True,
            # max_length=self.rxn_max_len,
            return_tensors="np"
        )

        if self.use_cls_token:
            num_sentinel_tokens = 1
        else:
            actual_length = int(rxn_encoding['attention_mask'].sum())
            num_sentinel_tokens = max(1, actual_length - 2)

        sentinel_sequence = self.sentinel_token * num_sentinel_tokens

        user_content = self.template_info['user_template'].format(
            input_smiles=input_smiles,
            sentinel_tokens=sentinel_sequence
        )

        messages = [
            {"role": "system", "content": self.template_info['system_message']},
            {"role": "user", "content": user_content}
        ]

        input_text = self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_encoding = self.qwen_tokenizer(input_text, return_tensors="np", truncation=True)

        return {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'rxn_input_ids': rxn_encoding['input_ids'],
            'rxn_attention_mask': rxn_encoding['attention_mask'],
        }

def parse_solvent_data(file_path: str) -> List[Dict]:
    """
    解析包含溶剂信息的数据文件。
    文件格式: reaction\tsolvents\tsolvent_labels
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f) # 跳过表头
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            reaction, solvents_str, solvent_labels_str = parts
            
            # 解析solvents
            try:
                solvents = eval(solvents_str)
            except:
                solvents = []

            # 解析solvent_labels
            try:
                solvent_labels = eval(solvent_labels_str)
            except:
                solvent_labels = []

            data.append({
                "reaction": reaction,
                "solvents": solvents,
                "solvent_labels": solvent_labels
            })
    print(f"Parsed {len(data)} valid solvent prediction entries from {file_path}")
    return data

