# -*- coding: utf-8 -*-
"""
词表扩展工具：将化学词表添加到Qwen分词器中
处理重合词汇并调整embedding大小
"""
import os
import argparse
from typing import Set, Tuple
from mindnlp.transformers import AutoTokenizer
from model.tokenizer import get_default_tokenizer


class VocabExtender:
    """词表扩展器，用于将化学词表合并到Qwen分词器中"""
    
    def __init__(self, qwen_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.qwen_model_name = qwen_model_name
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        self.chem_tokenizer = get_default_tokenizer()
        
        # 确保pad_token设置
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
    
    def analyze_vocab_overlap(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        分析Qwen和化学词表的重合情况
        
        Returns:
            overlap_tokens: 重合的词汇
            chem_only_tokens: 仅在化学词表中的词汇
            qwen_only_tokens: 仅在Qwen词表中的词汇（示例）
        """
        # 获取词表
        qwen_vocab = set(self.qwen_tokenizer.get_vocab().keys())
        chem_vocab = set(self.chem_tokenizer.vocab_list)
        
        # 计算重合和差异
        overlap_tokens = qwen_vocab.intersection(chem_vocab)
        chem_only_tokens = chem_vocab - qwen_vocab
        qwen_only_tokens = qwen_vocab - chem_vocab  # 仅用于统计，不实际使用
        
        print(f"=== 词表分析结果 ===")
        print(f"Qwen词表大小: {len(qwen_vocab)}")
        print(f"化学词表大小: {len(chem_vocab)}")
        print(f"重合词汇数量: {len(overlap_tokens)}")
        print(f"仅在化学词表中的词汇: {len(chem_only_tokens)}")
        print(f"仅在Qwen词表中的词汇: {len(qwen_only_tokens)}")
        
        # 打印一些重合的词汇示例
        if overlap_tokens:
            print(f"\n重合词汇示例: {list(overlap_tokens)[:20]}")
        
        # 打印一些仅在化学词表中的词汇示例
        if chem_only_tokens:
            print(f"仅在化学词表中的词汇示例: {list(chem_only_tokens)[:20]}")
        
        return overlap_tokens, chem_only_tokens, qwen_only_tokens
    
    def extend_tokenizer_vocab(self, add_sentinel_token: bool = True, sentinel_token: str = "<SMI_TOKEN>") -> AutoTokenizer:
        """
        扩展Qwen分词器的词表，添加不重合的化学词汇和哨兵令牌
        
        Args:
            add_sentinel_token: 是否添加哨兵令牌
            sentinel_token: 哨兵令牌字符串
        
        Returns:
            扩展后的分词器
        """
        _1, chem_only_tokens, _2 = self.analyze_vocab_overlap()
        
        # 将不重合的化学词汇转换为列表并排序，确保一致性
        new_tokens = sorted(list(chem_only_tokens))
        
        # 添加哨兵令牌
        if add_sentinel_token and sentinel_token not in self.qwen_tokenizer.get_vocab():
            new_tokens.append(sentinel_token)
            print(f"添加哨兵令牌: {sentinel_token}")
        
        if not new_tokens:
            print("没有需要添加的新词汇")
            return self.qwen_tokenizer
        
        print(f"\n=== 开始扩展词表 ===")
        print(f"原始词表大小: {len(self.qwen_tokenizer.get_vocab())}")
        print(f"将添加 {len(new_tokens)} 个新词汇（包含哨兵令牌）")
        
        # 添加新词汇到分词器
        num_added = self.qwen_tokenizer.add_tokens(new_tokens)
        print(f"成功添加 {num_added} 个新词汇")
        print(f"扩展后词表大小: {len(self.qwen_tokenizer.get_vocab())}")
        
        return self.qwen_tokenizer
    
    def save_extended_tokenizer(self, save_path: str, add_sentinel_token: bool = True, sentinel_token: str = "<SMI_TOKEN>"):
        """保存扩展后的分词器"""
        extended_tokenizer = self.extend_tokenizer_vocab(add_sentinel_token, sentinel_token)
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存分词器
        extended_tokenizer.save_pretrained(save_path)
        print(f"扩展后的分词器已保存到: {save_path}")
        
        # 保存词表信息
        info_file = os.path.join(save_path, "vocab_extension_info.txt")
        overlap_tokens, chem_only_tokens, _ = self.analyze_vocab_overlap()
        
        with open(info_file, "w", encoding="utf-8") as f:
            f.write(f"词表扩展信息\n")
            f.write(f"================\n")
            f.write(f"原始Qwen词表大小: {len(self.qwen_tokenizer.get_vocab()) - len(chem_only_tokens)}\n")
            f.write(f"化学词表大小: {len(self.chem_tokenizer.vocab_list)}\n")
            f.write(f"重合词汇数量: {len(overlap_tokens)}\n")
            f.write(f"新增词汇数量: {len(chem_only_tokens)}\n")
            f.write(f"哨兵令牌: {sentinel_token}\n")
            f.write(f"扩展后词表大小: {len(extended_tokenizer.get_vocab())}\n\n")
            
            f.write("重合词汇:\n")
            for token in sorted(overlap_tokens):
                f.write(f"  {token}\n")
            
            f.write("\n新增词汇:\n")
            for token in sorted(chem_only_tokens):
                f.write(f"  {token}\n")
        
        return extended_tokenizer
    
    def create_token_mapping(self, add_sentinel_token: bool = True, sentinel_token: str = "<SMI_TOKEN>") -> dict:
        """
        创建化学分词器到扩展Qwen分词器的token映射
        
        Returns:
            映射字典 {chem_token_id: qwen_token_id}
        """
        extended_tokenizer = self.extend_tokenizer_vocab(add_sentinel_token, sentinel_token)
        
        # 创建映射
        token_mapping = {}
        chem_vocab = self.chem_tokenizer.get_vocab()
        qwen_vocab = extended_tokenizer.get_vocab()
        
        for token, chem_id in chem_vocab.items():
            if token in qwen_vocab:
                qwen_id = qwen_vocab[token]
                token_mapping[chem_id] = qwen_id
        
        print(f"创建了 {len(token_mapping)} 个token映射")
        return token_mapping


def resize_model_embeddings(model, new_vocab_size: int):
    """
    调整模型的embedding大小以匹配新词表
    
    Args:
        model: 要调整的模型
        new_vocab_size: 新的词表大小
    """
    if hasattr(model, 'resize_token_embeddings'):
        model.resize_token_embeddings(new_vocab_size)
        print(f"模型embedding已调整为词表大小: {new_vocab_size}")
    elif hasattr(model, 'qwen') and hasattr(model.qwen, 'resize_token_embeddings'):
        model.qwen.resize_token_embeddings(new_vocab_size)
        print(f"Qwen子模型embedding已调整为词表大小: {new_vocab_size}")
    else:
        print("警告: 无法自动调整embedding大小，请手动处理")


def test_vocab_extension(qwen_model_name: str, save_path: str):
    """测试词表扩展功能"""
    print("=== 测试词表扩展功能 ===")
    
    # 创建扩展器
    extender = VocabExtender(qwen_model_name)
    
    # 分析词表重合情况
    extender.analyze_vocab_overlap()
    
    # 保存扩展后的分词器
    extended_tokenizer = extender.save_extended_tokenizer(save_path)
    
    # 创建token映射
    token_mapping = extender.create_token_mapping()
    
    # 测试分词效果
    test_smiles = "C1=CC=C1.C=C"
    test_text = f"The reactants {test_smiles} can be represented as <SMI_TOKEN> <SMI_TOKEN>. Predict the product."
    
    print(f"\n=== 测试分词效果 ===")
    print(f"测试文本: {test_text}")
    
    # 原始分词器分词
    original_tokens = AutoTokenizer.from_pretrained(qwen_model_name).tokenize(test_text)
    print(f"原始分词结果: {original_tokens}")
    
    # 扩展分词器分词
    extended_tokens = extended_tokenizer.tokenize(test_text)
    print(f"扩展分词结果: {extended_tokens}")
    
    # 化学分词器分词SMILES
    chem_tokens = extender.chem_tokenizer.tokenize(test_smiles)
    print(f"化学分词结果: {chem_tokens}")
    
    return extended_tokenizer, token_mapping


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="词表扩展工具：将化学词表添加到Qwen分词器中",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python utils/vocab_extension.py --model Qwen/Qwen2.5-0.5B-Instruct --save_path ./extended_tokenizers
  
注意: 实际保存路径会自动在指定路径下创建对应的模型文件夹 (qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b 或 qwen2.5-7b)
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        choices=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="原始Qwen分词器模型路径 (默认: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        default="./model/extended_qwen_tokenizer",
        help="扩展后分词器的保存路径 (默认: ../model/extended_qwen_tokenizer)"
    )
    
    parser.add_argument(
        "--sentinel_token",
        type=str,
        default="<SMI_TOKEN>",
        help="哨兵令牌 (默认: <SMI_TOKEN>)"
    )
    
    parser.add_argument(
        "--no_sentinel",
        action="store_true",
        help="不添加哨兵令牌"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行测试模式"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 根据模型参数生成对应的文件夹名称
    if "0.5B" in args.model:
        model_suffix = "qwen2.5-0.5b"
    elif "1.5B" in args.model:
        model_suffix = "qwen2.5-1.5b"
    elif "3B" in args.model:
        model_suffix = "qwen2.5-3b"
    elif "7B" in args.model:
        model_suffix = "qwen2.5-7b"
    else:
        model_suffix = "qwen2.5-unknown"
    
    # 在保存路径后面附加模型参数文件夹
    final_save_path = os.path.join(args.save_path, model_suffix)
    
    print(f"=== 词表扩展工具 ===")
    print(f"原始模型: {args.model}")
    print(f"保存路径: {final_save_path}")
    print(f"哨兵令牌: {args.sentinel_token if not args.no_sentinel else '无'}")
    print()
    
    if args.test:
        # 测试模式
        test_vocab_extension(args.model, final_save_path)
    else:
        # 正常扩展模式
        print("=== 开始词表扩展 ===")
        
        # 创建扩展器
        extender = VocabExtender(args.model)
        
        # 分析词表重合情况
        extender.analyze_vocab_overlap()
        
        # 保存扩展后的分词器
        extender.save_extended_tokenizer(
            final_save_path, 
            add_sentinel_token=not args.no_sentinel,
            sentinel_token=args.sentinel_token
        )
        
        # 创建token映射
        extender.create_token_mapping(
            add_sentinel_token=not args.no_sentinel,
            sentinel_token=args.sentinel_token
        )
    
    print("\n=== 词表扩展完成 ===")
    print("使用方法:")
    print(f"1. 加载扩展后的分词器: AutoTokenizer.from_pretrained('{final_save_path}')")
    print("2. 在模型中调用: resize_model_embeddings(model, len(extended_tokenizer.get_vocab()))")
    print("3. 使用token_mapping进行化学token映射")


if __name__ == "__main__":
    main()