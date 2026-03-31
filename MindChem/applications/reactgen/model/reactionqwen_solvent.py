import mindspore as ms
from mindnlp.core import nn as mnn 
from mindnlp.core import ops as mops
from mindnlp.transformers import PretrainedConfig, PreTrainedModel
from model.reactionqwen import MultiModalQwen, MultiModalQwenConfig
import numpy as np

# --- Config 类  ---
class ReactionQwenForSolventPredictionConfig(PretrainedConfig):
    model_type = "reactionqwen_solvent_prediction"
    def __init__(
        self,
        backbone_config: dict = None,
        num_solvent_labels: int = 51,
        loss_weight_multi_label: float = 1.0,
        pooling_type: str = 'mean',
        pos_weight_value: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_config = backbone_config if backbone_config else {}
        self.num_solvent_labels = num_solvent_labels
        self.loss_weight_multi_label = loss_weight_multi_label
        self.pooling_type = pooling_type
        self.pos_weight_value = pos_weight_value

# 继承 PreTrainedModel
class ReactionQwenForSolventPrediction(PreTrainedModel):
    config_class = ReactionQwenForSolventPredictionConfig

    def __init__(self, config: ReactionQwenForSolventPredictionConfig):
        super().__init__(config)
        
        # 1. 加载骨干 (MindNLP 对象兼容 MindNLP 父类)
        print("正在初始化 Backbone (MindNLP Native)...")
        backbone_qwen_config = MultiModalQwenConfig(**config.backbone_config)
        self.backbone = MultiModalQwen(backbone_qwen_config)
        
        # 2. 定义分类头
        hidden_size = self.backbone.qwen.config.hidden_size
        self.solvent_classifier = mnn.Linear(hidden_size, config.num_solvent_labels, bias=True)

        # 3. 损失函数 (MindSpore 原生 Loss 通常兼容 Tensor)
        # 为了保险，我们在 construct 里手动算，或者用 ms.nn 也行，这里保留 ms.nn 
        # 因为 Loss 不涉及参数注册，混用通常没问题
        from mindspore import nn as ms_nn
        pos_weight = ms.Tensor([config.pos_weight_value] * config.num_solvent_labels, dtype=ms.float32)
        self.loss_fct = ms_nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # 初始化权重 (MindNLP 标准流程)
        self.post_init()

    # 使用 forward (MindNLP 标准入口)
    def forward(
        self,
        input_ids,
        attention_mask,
        rxn_input_ids,
        rxn_attention_mask,
        solvent_labels=None,
        **kwargs
    ):
        # 1. 骨干前向
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            rxn_input_ids=rxn_input_ids,
            rxn_attention_mask=rxn_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden_state = backbone_outputs.hidden_states[-1]

        # 2. 池化
        if self.config.pooling_type == 'mean':
            # 注意：使用 mindnlp.core.ops 或 mindspore.ops 均可，这里用 mops 保持风格一致
            masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            # 避免除以0
            sum_mask = mops.maximum(sum_mask, ms.Tensor(1e-9, dtype=sum_mask.dtype))
            pooled_output = masked_hidden_state.sum(dim=1) / sum_mask
        elif self.config.pooling_type == 'cls':
            pooled_output = last_hidden_state[:, 0]
        else:
            raise ValueError(f"Unsupported pooling type: {self.config.pooling_type}")

        # 3. 分类头 (调用 mindnlp.core.nn.Linear)
        solvent_logits = self.solvent_classifier(pooled_output)

        # 4. 计算损失
        total_loss = None
        loss_dict = {}

        if solvent_labels is not None:
            # Loss 计算依然可以使用 MindSpore 原生算子
            loss = self.loss_fct(solvent_logits, solvent_labels)
            total_loss = loss * self.config.loss_weight_multi_label
            loss_dict['loss_multi_label'] = loss

        return {
            "loss": total_loss,
            "loss_dict": loss_dict,
            "solvent_logits": solvent_logits,
            "hidden_states": backbone_outputs.hidden_states,
        }
    
    # 预测方法 (完整解码逻辑)
    def predict_solvents(
        self,
        input_ids,
        attention_mask,
        rxn_input_ids,
        rxn_attention_mask,
        solvent_vocab=None,
        threshold=0.9,
        max_count=3,
        return_probabilities=False,
        **kwargs
    ):
        """
        执行推理并解析结果
        """
        self.set_train(False)
        
        # 1. 获取模型输出 (Logits)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            rxn_input_ids=rxn_input_ids,
            rxn_attention_mask=rxn_attention_mask,
            return_dict=True
        )
        
        # 2. 计算概率 (Sigmoid)
        # 注意：这里使用 mindnlp.core.ops 或 mindspore.ops 均可
        solvent_logits = outputs['solvent_logits']
        solvent_probs = mops.sigmoid(solvent_logits) # 使用 mops 保持风格一致
        
        batch_results = []
        solvent_probs_np = solvent_probs.asnumpy() # 转为 numpy 处理逻辑
        
        for i in range(solvent_probs_np.shape[0]):
            sample_probs = solvent_probs_np[i]
            predicted_indices = []
            
            # --- 步骤 A: No Solvent (Index 0) 优先判定 ---
            # 如果 No Solvent 的概率非常高 (>0.9) 且 是所有里面最高的
            if sample_probs[0] > 0.9 and sample_probs[0] > sample_probs[1:].max():
                predicted_indices = [0]
            
            else:
                # --- 步骤 B: 筛选候选溶剂 (Index 1-50) ---
                # 1. 找到所有概率 > threshold 的溶剂
                candidate_indices = np.where(sample_probs[1:] > threshold)[0] + 1
                
                if len(candidate_indices) > 0:
                    # 2. 获取这些候选溶剂的概率
                    candidate_probs = sample_probs[candidate_indices]
                    
                    # 3. 按概率从高到低排序
                    sorted_args = np.argsort(candidate_probs)[::-1]
                    sorted_indices = candidate_indices[sorted_args]
                    
                    # 4. --- 核心限制 --- : 截断，最多只取 max_count 个
                    final_indices = sorted_indices[:max_count]
                    
                    predicted_indices = final_indices.tolist()
                else:
                    # --- 步骤 C: 保底策略 ---
                    # 如果没有一个超过阈值，取概率最大的那个 (Top-1)
                    # 排除掉 Index 0 (No Solvent)，只在 1-50 里找
                    # 除非 0 真的特别大
                    max_idx = np.argmax(sample_probs)
                    if max_idx == 0 and sample_probs[0] < 0.5:
                         # 如果最大是0但概率也不高，强制找第二大的
                         sample_probs_no_zero = sample_probs.copy()
                         sample_probs_no_zero[0] = -1
                         max_idx = np.argmax(sample_probs_no_zero)
                    
                    predicted_indices = [int(max_idx)]
            
            # 转换为溶剂名称
            if solvent_vocab is not None:
                # 处理可能出现的键错误 (比如预测了不存在的ID)
                predicted_solvents = [solvent_vocab.get(idx, f"Unknown_{idx}") for idx in predicted_indices]
            else:
                predicted_solvents = predicted_indices
            
            result = {
                'predicted_solvents': predicted_solvents,
                'predicted_count': len(predicted_indices)
            }
            
            if return_probabilities:
                selected_probs = sample_probs[predicted_indices]
                if not isinstance(selected_probs, np.ndarray):
                    selected_probs = np.array(selected_probs)
                result['solvent_probabilities'] = selected_probs.tolist()
                            
            batch_results.append(result)
        
        # 兼容单样本和Batch
        if len(batch_results) == 1 and input_ids.shape[0] == 1:
            return batch_results[0]
        
        # ⚠️ 注意：这里必须返回 List[Dict]，而不是单个 Dict
        # 之前的代码如果 Batch > 1 可能返回列表，推理脚本里是按 Batch 处理的
        return batch_results