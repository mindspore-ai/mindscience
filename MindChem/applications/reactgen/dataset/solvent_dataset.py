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
from typing import List, Dict

NUM_SOLVENTS = 51 # 0-51, 0 for no solvent
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

        valid_indices = []
        for label in item['solvent_labels']:
            idx = int(label)
            
            # 核心逻辑：只处理 0-50 的 ID
            if idx < NUM_SOLVENTS:
                solvent_labels_multi_hot[idx] = 1.0
                valid_indices.append(idx)

        if solvent_labels_multi_hot.sum() == 0:
             solvent_labels_multi_hot[0] = 1.0
    
        # 3. 填充和截断
        input_ids = dual_input['input_ids'][0]
        attention_mask = dual_input['attention_mask'][0]

        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids = np.pad(input_ids, (0, padding_len), 'constant', constant_values=self.qwen_tokenizer.pad_token_id)
            attention_mask = np.pad(attention_mask, (0, padding_len), 'constant', constant_values=0)
        else:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]

        return {
            "input_ids": input_ids.astype(np.int32),
            "attention_mask": attention_mask.astype(np.int32),
            "rxn_input_ids": dual_input['rxn_input_ids'][0].astype(np.int32),
            "rxn_attention_mask": dual_input['rxn_attention_mask'][0].astype(np.int32),
            "solvent_labels": solvent_labels_multi_hot,
            # 调试信息
            "reactants": reactants,
        }

    def _create_dual_representation_input(self, input_smiles: str) -> Dict:
        """创建双重表示输入"""
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

        input_encoding = self.qwen_tokenizer(input_text, return_tensors="np", max_length=self.max_len, truncation=True)

        return {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'rxn_input_ids': rxn_encoding['input_ids'],
            'rxn_attention_mask': rxn_encoding['attention_mask'],
        }

def parse_solvent_data(file_path: str) -> List[Dict]:
    """
    解析你的数据格式: Reaction \t Solvents_List_Str \t Label_IDs_List_Str
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f) 
        for line in f:
            parts = line.strip().split('\t')
            # 你的数据有时候可能只有 Reaction，没有溶剂（预测时），做个兼容
            if len(parts) < 3:
                continue
            
            reaction = parts[0]
            solvents_str = parts[1] # 这一列其实训练用不到，主要用 ID
            solvent_labels_str = parts[2]
            
            try:
                solvents = eval(solvents_str)
            except:
                solvents = []
            try:
                # 使用 eval 解析字符串列表 "[9, 2]" -> [9, 2]
                solvent_labels = eval(solvent_labels_str)
            except:
                solvent_labels = []

            data.append({
                "reaction": reaction,
                "solvents": solvents,
                "solvent_labels": solvent_labels
            })
            
    print(f"Parsed {len(data)} entries from {file_path}")
    return data


if __name__ == '__main__':
    import os
    
    print("--- Running Test for SolventPredictionDataset ---")

    # 1. 创建一个虚拟数据文件
    dummy_data = [
        "reaction\tsolvents\tsolvent_labels",
        "CCO>>CCN\t['c1ccccc1', 'O']\t[1, 12]",
        "CC(=O)O.CCN>>CC(=O)OCCN\t['CCO']\t[0]",
        "BrC1=CC=C(Br)C=C1.CC#N>>NC1=CC=C(N)C=C1\t['CS(C)=O', 'c1ccncc1']\t[9, 8]"
    ]
    dummy_file_path = "dummy_solvent_data.txt"
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        for line in dummy_data:
            f.write(line + "\n")

    # 2. Mock分词器
    class MockTokenizer:
        def __init__(self, vocab, pad_token_id=0):
            self.vocab = vocab
            self.pad_token_id = pad_token_id
            self.pad_token = "[PAD]"
            self.eos_token = "[EOS]"
        
        def __call__(self, text, return_tensors=None, max_length=None, truncation=None, padding=None):
            tokens = list(text[:max_length-2])
            input_ids = [self.vocab.get(t, 0) for t in tokens]
            input_ids = [1] + input_ids + [2] # 模拟CLS和SEP
            
            attention_mask = [1] * len(input_ids)

            if padding == "max_length":
                pad_len = max_length - len(input_ids)
                input_ids += [self.pad_token_id] * pad_len
                attention_mask += [0] * pad_len

            if return_tensors == "np":
                return {
                    "input_ids": np.array([input_ids]),
                    "attention_mask": np.array([attention_mask])
                }
            return {"input_ids": [input_ids], "attention_mask": [attention_mask]}

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            # 简化版template，仅用于测试
            text = ""
            for msg in messages:
                text += f"{msg['role']}: {msg['content']}\n"
            return text
        
        def add_tokens(self, tokens):
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    # 3. 初始化分词器和数据集
    qwen_vocab = {f"t{i}": i for i in range(100)}
    qwen_vocab.update({"<SMI_TOKEN>": 100, "[PAD]": 0, "[CLS]": 1, "[SEP]": 2})
    qwen_tokenizer = MockTokenizer(qwen_vocab)
    
    rxn_vocab = {f"r{i}": i for i in range(50)}
    rxn_tokenizer = MockTokenizer(rxn_vocab)

    # 解析数据
    parsed_data = parse_solvent_data(dummy_file_path)
    
    # 创建数据集
    dataset = SolventPredictionDataset(
        data=parsed_data,
        qwen_tokenizer=qwen_tokenizer,
        rxn_tokenizer=rxn_tokenizer,
        max_len=128,
        rxn_max_len=64
    )

    # 4. 检查一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n--- Sample 0 ---")
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {value}")
        
        # 验证标签是否正确
        expected_labels = np.zeros(NUM_SOLVENTS, dtype=np.float32)
        expected_labels[[1, 12]] = 1.0
        assert np.array_equal(sample['solvent_labels'], expected_labels), "Solvent labels mismatch!"
        assert sample['solvent_count'] == 2, "Solvent count mismatch!"
        print("\n✅ Sample 0 labels are correct.")

        # 验证维度
        assert sample['input_ids'].shape == (128,), "input_ids shape mismatch!"
        assert sample['rxn_input_ids'].shape == (64,), "rxn_input_ids shape mismatch!"
        print("✅ Shapes are correct.")

    print("\n--- Test Completed Successfully ---")

    # 5. 清理虚拟文件
    os.remove(dummy_file_path)