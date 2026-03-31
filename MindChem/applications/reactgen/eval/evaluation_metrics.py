# -*- coding: utf-8 -*-
"""
化学反应预测评估指标
实现top-1准确率、SMILES标准化和分子有效性检查
"""
from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class ChemicalReactionEvaluator:
    """化学反应预测评估器"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """重置评估指标"""
        self.predictions = []
        self.targets = []
        self.valid_predictions = []
        self.exact_matches = []
        
    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        使用RDKit标准化SMILES
        
        Args:
            smiles: 输入的SMILES字符串
            
        Returns:
            标准化的SMILES字符串，如果无效则返回None
        """
        try:
            # 去除空格和特殊字符
            smiles = smiles.strip()
            if not smiles:
                return None
            
            # 使用RDKit解析分子
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 生成标准化SMILES
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            return canonical_smiles
            
        except Exception as e:
            print(f"SMILES标准化失败 '{smiles}': {e}")
            return None
    
    def extract_product_smiles(self, generated_text: str) -> Optional[str]:
        """
        从生成的文本中提取产物SMILES
        
        Args:
            generated_text: 模型生成的文本
            
        Returns:
            提取的产物SMILES字符串
        """
        # 去除前后空格和特殊标记
        generated_text = generated_text.strip()
        
        # 清理各种可能的特殊标记
        cleaned_text = generated_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("<|im_start|>", "").strip()
        
        # 简化的提取策略：
        # 1. 对于化学反应预测，生成的文本通常就是纯SMILES
        # 2. 直接使用清理后的文本进行RDKit验证
        
        if cleaned_text:
            # 尝试直接解析为SMILES，让RDKit来判断有效性
            if self.is_valid_molecule(cleaned_text):
                return cleaned_text
            
            # 如果包含多行，取第一行（可能包含解释文本的情况）
            first_line = cleaned_text.split('\n')[0].strip()
            if first_line and self.is_valid_molecule(first_line):
                return first_line
        
        # 如果RDKit无法解析，返回原始清理后的文本（交给后续标准化判断）
        return cleaned_text if cleaned_text else generated_text
    
    def is_likely_smiles(self, text: str) -> bool:
        """
        判断文本是否可能是SMILES字符串
        注意：现在主要使用is_valid_molecule进行验证，这个函数作为辅助
        
        Args:
            text: 待判断的文本
            
        Returns:
            是否可能是SMILES
        """
        if not text or len(text) < 2:
            return False
        
        # 简化判断：直接使用RDKit验证更可靠
        return self.is_valid_molecule(text)
    
    def is_valid_molecule(self, smiles: str) -> bool:
        """
        检查SMILES是否表示有效的分子
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            是否为有效分子
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def calculate_molecular_properties(self, smiles: str) -> Dict:
        """
        计算分子的基本性质
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            分子性质字典
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            return {
                'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'logp': rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
            }
        except Exception as e:
            print(f"计算分子性质失败: {e}")
            return {}
    
    def add_prediction(self, prediction: str, target: str, debug=False) -> Dict:
        """
        添加一个预测结果进行评估
        
        Args:
            prediction: 模型预测的文本
            target: 真实的产物SMILES
            debug: 是否打印调试信息
            
        Returns:
            当前预测的评估结果
        """
        # 提取预测的SMILES
        pred_smiles = self.extract_product_smiles(prediction)
        
        # 标准化SMILES
        canonical_pred = self.canonicalize_smiles(pred_smiles) if pred_smiles else None
        canonical_target = self.canonicalize_smiles(target)
        
        # 检查有效性
        is_valid = canonical_pred is not None
        
        # 检查精确匹配
        exact_match = (canonical_pred == canonical_target) if (canonical_pred and canonical_target) else False
        
        # 调试信息
        if debug:
            print(f"调试信息:")
            print(f"  原始预测: {prediction}")
            print(f"  提取SMILES: {pred_smiles}")
            print(f"  标准化预测: {canonical_pred}")
            print(f"  标准化目标: {canonical_target}")
            print(f"  有效性: {is_valid}")
            print(f"  精确匹配: {exact_match}")
            print("-" * 40)
        
        # 存储结果
        self.predictions.append(pred_smiles)
        self.targets.append(target)
        self.valid_predictions.append(is_valid)
        self.exact_matches.append(exact_match)
        
        # 返回当前预测的详细信息
        result = {
            'original_prediction': prediction,
            'extracted_smiles': pred_smiles,
            'canonical_prediction': canonical_pred,
            'canonical_target': canonical_target,
            'is_valid': is_valid,
            'exact_match': exact_match
        }
        
        return result
    
    def compute_metrics(self) -> Dict:
        """
        计算所有评估指标
        
        Returns:
            评估指标字典
        """
        if not self.predictions:
            return {}
        
        total_predictions = len(self.predictions)
        valid_count = sum(self.valid_predictions)
        exact_match_count = sum(self.exact_matches)
        
        # 计算指标
        validity_rate = valid_count / total_predictions if total_predictions > 0 else 0.0
        top1_accuracy = exact_match_count / total_predictions if total_predictions > 0 else 0.0
        
        # 计算在有效预测中的准确率
        valid_accuracy = (exact_match_count / valid_count) if valid_count > 0 else 0.0
        
        metrics = {
            'total_predictions': total_predictions,
            'valid_predictions': valid_count,
            'exact_matches': exact_match_count,
            'validity_rate': validity_rate,
            'top1_accuracy': top1_accuracy,
            'valid_accuracy': valid_accuracy,
            'invalid_predictions': total_predictions - valid_count
        }
        
        return metrics
    
    def get_detailed_results(self) -> List[Dict]:
        """
        获取详细的预测结果
        
        Returns:
            详细结果列表
        """
        results = []
        for i, (pred, target, valid, match) in enumerate(zip(
            self.predictions, self.targets, self.valid_predictions, self.exact_matches
        )):
            canonical_pred = self.canonicalize_smiles(pred) if pred else None
            canonical_target = self.canonicalize_smiles(target)
            
            result = {
                'index': i,
                'prediction': pred,
                'target': target,
                'canonical_prediction': canonical_pred,
                'canonical_target': canonical_target,
                'is_valid': valid,
                'exact_match': match,
                'prediction_properties': self.calculate_molecular_properties(canonical_pred) if canonical_pred else {},
                'target_properties': self.calculate_molecular_properties(canonical_target) if canonical_target else {}
            }
            results.append(result)
        
        return results
    
    def print_evaluation_summary(self, show_examples=True, max_examples=10):
        """打印评估摘要
        
        Args:
            show_examples: 是否显示样例
            max_examples: 最大显示样例数
        """
        metrics = self.compute_metrics()
        
        if not metrics:
            print("No predictions to evaluate")
            return
        
        print("\n" + "="*60)
        print("化学反应预测评估结果")
        print("="*60)
        print(f"总预测数: {metrics['total_predictions']}")
        print(f"有效预测数: {metrics['valid_predictions']}")
        print(f"精确匹配数: {metrics['exact_matches']}")
        print(f"无效预测数: {metrics['invalid_predictions']}")
        print("-" * 40)
        print(f"分子有效性: {metrics['validity_rate']:.4f} ({metrics['validity_rate']*100:.2f}%)")
        print(f"Top-1准确率: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
        print(f"有效预测中的准确率: {metrics['valid_accuracy']:.4f} ({metrics['valid_accuracy']*100:.2f}%)")
        print("="*60)
        
        if show_examples and metrics['total_predictions'] > 0:
            results = self.get_detailed_results()
            
            # 分类显示样例：正确预测、错误预测、无效预测
            correct_examples = [r for r in results if r['exact_match']]
            incorrect_valid_examples = [r for r in results if r['is_valid'] and not r['exact_match']]
            invalid_examples = [r for r in results if not r['is_valid']]
            
            examples_per_category = min(max_examples // 3, 3)  # 每类最多显示3个
            
            print("\n样例预测结果:")
            
            if correct_examples:
                print(f"\n✓ 正确预测样例 (共{len(correct_examples)}个):")
                for i, result in enumerate(correct_examples[:examples_per_category]):
                    print(f"  {i+1}. 预测: {result['prediction']}")
                    print(f"     真实: {result['target']}")
                    print()
            
            if incorrect_valid_examples:
                print(f"\n✗ 错误但有效的预测样例 (共{len(incorrect_valid_examples)}个):")
                for i, result in enumerate(incorrect_valid_examples[:examples_per_category]):
                    print(f"  {i+1}. 预测: {result['prediction']}")
                    print(f"     真实: {result['target']}")
                    if result['canonical_prediction'] != result['prediction']:
                        print(f"     标准化预测: {result['canonical_prediction']}")
                    print()
            
            if invalid_examples:
                print(f"\n⚠ 无效预测样例 (共{len(invalid_examples)}个):")
                for i, result in enumerate(invalid_examples[:examples_per_category]):
                    print(f"  {i+1}. 预测: {result['prediction']}")
                    print(f"     真实: {result['target']}")
                    print()
            
            if metrics['total_predictions'] > max_examples:
                print(f"\n注: 仅显示部分样例，完整结果请查看详细预测文件")


def create_evaluation_callback(evaluator: ChemicalReactionEvaluator):
    """
    创建用于训练过程中评估的回调函数
    
    Args:
        evaluator: 化学反应评估器
        
    Returns:
        评估回调函数
    """
    def evaluate_predictions(predictions: List[str], targets: List[str]) -> Dict:
        """
        评估预测结果
        
        Args:
            predictions: 预测结果列表
            targets: 真实目标列表
            
        Returns:
            评估指标
        """
        evaluator.reset_metrics()
        
        for pred, target in zip(predictions, targets):
            evaluator.add_prediction(pred, target)
        
        return evaluator.compute_metrics()
    
    return evaluate_predictions


if __name__ == "__main__":
    # 测试评估器
    evaluator = ChemicalReactionEvaluator()
    
    # 测试数据
    test_cases = [
        ("C1C=CC2C1C2", "C1C=CC2C1C2"),  # 完全匹配
        ("CC(=O)OCC", "CC(=O)OCC"),      # 完全匹配
        ("c1ccc(Cl)cc1", "c1ccc(Cl)cc1"), # 完全匹配
        ("INVALID", "CC(=O)O"),           # 无效SMILES
        ("C1C=CC2C1C2", "CC(=O)O"),      # 不匹配
    ]
    
    print("测试化学反应预测评估器")
    print("-" * 30)
    
    for i, (pred, target) in enumerate(test_cases):
        print(f"测试 {i+1}: 预测='{pred}', 真实='{target}'")
        result = evaluator.add_prediction(pred, target)
        print(f"  结果: 有效={result['is_valid']}, 匹配={result['exact_match']}")
        print()
    
    # 打印最终评估结果
    evaluator.print_evaluation_summary()