# -*- coding: utf-8 -*-
"""
Multimodal Qwen that inherits PreTrainedModel
基于双重表示输入的化学多模态模型，同时使用原始SMILES和哨兵令牌
"""
import mindspore as ms
import mindspore.ops as ops
from mindnlp.core import nn
from mindnlp.transformers import (
    PretrainedConfig, PreTrainedModel,
    AutoModelForCausalLM, AutoTokenizer,
    BertModel,
    GenerationMixin
)


# ----------------------- 1. Config ----------------------------------------
class MultiModalQwenConfig(PretrainedConfig):
    model_type = "mm_qwen"

    def __init__(
        self,
        qwen_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        rxn_model_name="./model/reactionbert_mlm_16_1e-04_10_rsmiles/final_model",
        proj_dropout=0.1,
        freeze_backbones=True,
        use_extended_vocab: bool = False,
        extended_vocab_path: str = "./model/extended_qwen_tokenizer",
        sentinel_token: str = "<SMI_TOKEN>",
        sentinel_token_id: int = 152228,
        freeze_qwen: bool = False,
        freeze_reactionbert: bool = True,
        use_cls_token: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.qwen_model_name     = qwen_model_name
        self.rxn_model_name      = rxn_model_name
        self.proj_dropout        = proj_dropout
        self.freeze_backbones    = freeze_backbones
        self.use_extended_vocab  = use_extended_vocab
        self.extended_vocab_path = extended_vocab_path
        self.sentinel_token      = sentinel_token
        self.sentinel_token_id   = sentinel_token_id
        self.freeze_qwen         = freeze_qwen
        self.freeze_reactionbert = freeze_reactionbert
        self.use_cls_token       = use_cls_token 


# ----------------------- 2. Model -----------------------------------------
class MultiModalQwen(PreTrainedModel,GenerationMixin):
    config_class = MultiModalQwenConfig            # 必须绑定
    base_model_prefix = "mm_qwen"                  # 用于 save_pretrained()

    def __init__(
        self,
        config: MultiModalQwenConfig,
        qwen_model = None,
        reaction_model = None
    ):
        super().__init__(config)                   # 重要：注册 config 给基类

        # ------ 子模型 ------------------------------------------------------
        self.qwen = qwen_model or AutoModelForCausalLM.from_pretrained(
            config.qwen_model_name
        )
        self.rxn  = reaction_model or BertModel.from_pretrained(
            config.rxn_model_name
        )

        # ------ 词表扩展与Embedding调整 -----------------------------------
        if config.use_extended_vocab:
            # 1. 加载扩展后的分词器以获取新词表大小
            #    注意：这里只加载分词器用于获取大小，实际在训练脚本中使用的分词器需要保持一致
            extended_tokenizer = AutoTokenizer.from_pretrained(config.extended_vocab_path)
            new_vocab_size = len(extended_tokenizer)

            # 2. 调整Qwen模型的embedding大小
            current_vocab_size = self.qwen.config.vocab_size
            if new_vocab_size != current_vocab_size:
                print(f"Resizing Qwen token embeddings from {current_vocab_size} to {new_vocab_size}")
                self.qwen.resize_token_embeddings(new_vocab_size)
                # 更新模型配置中的vocab_size，以便保存后能正确加载
                self.qwen.config.vocab_size = new_vocab_size

        # ------ 投影层 ------------------------------------------------------
        rxn_dim  = self.rxn.config.hidden_size
        qwen_dim = self.qwen.config.hidden_size
        self.proj = nn.Linear(rxn_dim, qwen_dim, bias=False)
        self.drop = nn.Dropout(p=config.proj_dropout)

        # ------ 冻结主干（可选） --------------------------------------------
        if config.freeze_backbones:
            if config.freeze_qwen:
                for p in self.qwen.get_parameters(): p.requires_grad = False
            if config.freeze_qwen:
                for p in self.rxn.get_parameters(): p.requires_grad = False

        # ------ 让 PreTrainedModel 做权重初始化 / tie_weights ---------------
        self.post_init()

    # ---------------- required API hooks -----------------------------------
    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.qwen.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.qwen.get_output_embeddings()

    def get_decoder(self):
        return self.qwen.model

    def _init_weights(self, module):
        """初始化投影层；子模型已带预训练权重不必重新初始化"""
        if isinstance(module, nn.Linear):
            std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """为 `generate` 方法准备输入"""
        # 1. 从kwargs中提取我们自己的多模态输入
        rxn_input_ids = kwargs.pop("rxn_input_ids", None)
        rxn_attention_mask = kwargs.pop("rxn_attention_mask", None)

        # 2. 调用底层qwen的prepare_inputs_for_generation获取标准输入
        model_inputs = self.qwen.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )

        # 3. 只在第一步（past_key_values为None）时添加多模态输入
        # 在后续生成步骤中，不再需要重复处理反应物输入
        if past_key_values is None:
            model_inputs["rxn_input_ids"] = rxn_input_ids
            model_inputs["rxn_attention_mask"] = rxn_attention_mask
        else:
            # 后续步骤不传递反应物输入，避免重复处理
            model_inputs["rxn_input_ids"] = None
            model_inputs["rxn_attention_mask"] = None
        
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.qwen._reorder_cache(past_key_values, beam_idx)

    # ---------------- forward / construct ----------------------------------
    def forward(
        self,
        input_ids=None, # `generate`方法会传入此参数，我们将其视为 text_input_ids
        attention_mask=None, # `generate`方法会传入此参数，我们将其视为 text_attention_mask
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # --- 自定义的多模态输入 ---
        rxn_input_ids=None,
        rxn_attention_mask=None,
        **kwargs
    ):
        # 如果 `inputs_embeds` 存在，则直接使用它
        if inputs_embeds is not None:
            fused_emb = inputs_embeds
        # 否则，根据输入ID进行融合
        else:
            # ---- Reaction branch (if applicable) ---------------------------
            if rxn_input_ids is not None:
                rxn_outputs = self.rxn(
                    input_ids=rxn_input_ids,
                    attention_mask=rxn_attention_mask
                )
                rxn_token_embs = rxn_outputs.last_hidden_state
                projected_tokens = self.drop(self.proj(rxn_token_embs))
                
                tok_emb = self.qwen.get_input_embeddings()(input_ids)
                fused_emb = self._replace_tokens_elementwise(tok_emb, projected_tokens, input_ids)
            else:
                # 纯文本模式
                fused_emb = self.qwen.get_input_embeddings()(input_ids)

        # ---- Qwen branch --------------------------------------------------
        return self.qwen(
            inputs_embeds=fused_emb,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def _replace_tokens_elementwise(self,
                                    text_embeddings,       # (B, L, D)  来自 Qwen
                                    rxn_token_embeddings,  # (B, n, D)  ReactionBERT→proj 之后
                                    input_ids):            # (B, L)
        """
        将 <SMI_TOKEN> 处的 embedding 替换为 rxn_token_embeddings，对计算图友好。

        返回:
            fused_embeddings (Tensor): 形状仍为 (B, L, D)
        """
        # 1. 常量与形状
        sentinel_id = self.config.sentinel_token_id               # <SMI_TOKEN>
        B, L, D = text_embeddings.shape

        # 2. 找到哨兵位置 (B,L) → (N,2)  [batch_idx , seq_pos]
        sentinel_mask = (input_ids == sentinel_id)
        pos = ops.nonzero(sentinel_mask)                # (N, 2)
        if self.config.use_cls_token:
            first_sentinel_pos = sentinel_mask.astype(ms.int32).argmax(axis=1) # (B,)

            # 3. 构造展平索引
            batch_indices = ms.numpy.arange(B)  # (B,)
            flat_idx = (batch_indices * L + first_sentinel_pos).astype(ms.int32)  # (B,)
            
            # 4. 获取 ReactionBERT 的 [CLS] 表示
            # 通常位于 rxn_token_embeddings[:, 0, :]
            cls_embedding = rxn_token_embeddings[:, 0, :]  # (B, D)

            # 5. 构造 indices 和 updates
            indices = ops.tile(flat_idx.expand_dims(1), (1, D))  # (B, D)
            updates = cls_embedding  # (B, D)
            
            # 6. scatter 到原 embedding 中
            flat_text = text_embeddings.reshape(-1, D)  # (B*L, D)
            fused_flat = ops.tensor_scatter_elements(
            flat_text, indices, updates, axis=0)  # 替换对应位置
        else:
            # 3. 计算一维行索引  flat_idx ∈ [0, B*L)
            flat_idx = (pos[:, 0] * L + pos[:, 1]).astype(ms.int32)   # (N,)

            # 4. 整理待写入的更新向量  updates  =  (N, D)
            updates = rxn_token_embeddings.reshape(-1, D)[:flat_idx.shape[0]]

            # 5. 根据文档要求把 indices 复制到 (N, D) 以保持 updates.shape == indices.shape
            indices = ops.tile(flat_idx.expand_dims(1), (1, D))       # (N, D)

            # 6. 在展平的原 embedding 上执行 scatter
            flat_text = text_embeddings.reshape(-1, D)                # (B*L, D)
            fused_flat = ops.tensor_scatter_elements(
                flat_text, indices, updates, axis=0)                  # 默认为 reduction="none"

        # 7. 恢复原形状
        fused_embeddings = fused_flat.reshape(B, L, D)
        return fused_embeddings

    def print_trainable_parameters(self):
        """
        打印可训练参数的统计信息
        """
        def count_parameters(model, name):
            trainable_params = sum(p.numel() for p in model.get_parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.get_parameters())
            return trainable_params, total_params
        
        print("\n=== 可训练参数统计 ===")
        
        # Qwen模型参数
        qwen_trainable, qwen_total = count_parameters(self.qwen, "Qwen")
        for name, p in self.qwen.named_parameters():
            if p.requires_grad:
                print(name, p.shape)
        print(f"Qwen模型:")
        print(f"  可训练参数: {qwen_trainable:,}")
        print(f"  总参数: {qwen_total:,}")
        print(f"  可训练比例: {qwen_trainable/qwen_total*100:.2f}%")

        # 单独统计embedding层
        embedding_trainable = 0
        embedding_total = 0
        for name, p in self.qwen.named_parameters():
            if 'embed_tokens' in name or 'lm_head' in name:
                if p.requires_grad:
                    embedding_trainable += p.numel()
                embedding_total += p.numel()
        print(f"  Embedding层可训练参数: {embedding_trainable:,}")
        print(f"  Embedding层总参数: {embedding_total:,}")
        
            
        # ReactionBert模型参数
        rxn_trainable, rxn_total = count_parameters(self.rxn, "ReactionBert")
        print(f"\nReactionBert模型:")
        print(f"  可训练参数: {rxn_trainable:,}")
        print(f"  总参数: {rxn_total:,}")
        print(f"  可训练比例: {rxn_trainable/rxn_total*100:.2f}%")
        
        # 投射层参数
        proj_trainable = sum(p.numel() for p in self.proj.get_parameters() if p.requires_grad)
        proj_total = sum(p.numel() for p in self.proj.get_parameters())
        print(f"\n投射层:")
        print(f"  可训练参数: {proj_trainable:,}")
        print(f"  总参数: {proj_total:,}")
        print(f"  可训练比例: {proj_trainable/proj_total*100:.2f}%")
        
        # 整体统计
        total_trainable = qwen_trainable + rxn_trainable + proj_trainable
        total_params = qwen_total + rxn_total + proj_total
        print(f"\n整体统计:")
        print(f"  可训练参数: {total_trainable:,}")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练比例: {total_trainable/total_params*100:.2f}%")
        
        # 冻结状态信息
        print(f"\n冻结状态:")
        print(f"  Qwen冻结: {'Yes' if not any(p.requires_grad for p in self.qwen.get_parameters()) else 'No'}")
        print(f"  ReactionBert冻结: {'Yes' if not any(p.requires_grad for p in self.rxn.get_parameters()) else 'No'}")
        print(f"  投射层冻结: {'Yes' if not any(p.requires_grad for p in self.proj.get_parameters()) else 'No'}")
        print("="*40)


if __name__ == '__main__':
    def test_model_with_dataset():
        """
        使用dataset.py测试双重表示输入的多模态Qwen模型
        展示模型与数据集的正确分离
        """
        from ..dataset.dataset import DualRepresentationDataset, create_sample_data
        from tokenizer import get_default_tokenizer
        
        print("=== 使用Dataset测试模型 ===\n")
        
        # ---- 1. 初始化模型和分词器 ---------------------------------------
        config = MultiModalQwenConfig(
            use_extended_vocab=True,
            sentinel_token="<SMI_TOKEN>"
        )
        model = MultiModalQwen(config)

        # 加载分词器
        if config.use_extended_vocab:
            qwen_tokenizer = AutoTokenizer.from_pretrained(config.extended_vocab_path)
        else:
            qwen_tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_name)
        
        rxn_tokenizer = get_default_tokenizer()
        
        # 打印可训练参数统计
        model.print_trainable_parameters()
        
        # ---- 2. 创建数据集 ----------------------------------------------
        forward_reactions = create_sample_data()
        
        # 测试两种任务类型
        for task_type, reactions in [
            ('forward', forward_reactions),
            ('retrosynthesis', forward_reactions[:2])  # 减少数量用于演示
        ]:
            print(f"--- 测试任务: {task_type} ---")
            
            # 创建数据集
            dataset = DualRepresentationDataset(
                reactions=reactions,
                qwen_tokenizer=qwen_tokenizer,
                rxn_tokenizer=rxn_tokenizer,
                task_type=task_type,
                max_len=256
            )
            
            # 获取第一个样本
            if len(dataset) > 0:
                sample = dataset[0]
                
                print(f"输入文本: {sample['input_text']}")
                print(f"目标文本: {sample['target_text']}")
                
                # ---- 3. 测试模型推理 --------------------------------
                model.set_train(False)
                
                # 准备模型输入（只用prompt部分进行生成）
                # 从sample中提取输入部分（排除target）
                input_text = sample['input_text']
                prompt_encoding = qwen_tokenizer(input_text, return_tensors="ms")
                
                # 准备反应输入（用于推理）
                rxn_input_ids = ms.Tensor([sample['rxn_input_ids']], dtype=ms.int32)
                rxn_attention_mask = ms.Tensor([sample['rxn_attention_mask']], dtype=ms.int32)
                
                try:
                    # 模型生成（新实现不再需要sentinel_positions）
                    generated_ids = model.generate(
                        input_ids=prompt_encoding.input_ids,
                        attention_mask=prompt_encoding.attention_mask,
                        rxn_input_ids=rxn_input_ids,
                        rxn_attention_mask=rxn_attention_mask,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=qwen_tokenizer.pad_token_id
                    )
                    
                    # 解码生成结果
                    generated_text = qwen_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    print(f"模型生成: {generated_text}")
                    
                except Exception as e:
                    print(f"生成时出错: {e}")
                
                # ---- 4. 测试模型训练 --------------------------------
                print("\n训练模式测试:")
                model.set_train(True)
                
                try:
                    # 使用完整的训练样本（新实现不再需要sentinel_positions）
                    training_inputs = {
                        'input_ids': ms.Tensor([sample['input_ids']], dtype=ms.int32),
                        'attention_mask': ms.Tensor([sample['attention_mask']], dtype=ms.int32),
                        'labels': ms.Tensor([sample['labels']], dtype=ms.int32),
                        'rxn_input_ids': rxn_input_ids,
                        'rxn_attention_mask': rxn_attention_mask
                    }
                    
                    outputs = model(**training_inputs, return_dict=True)
                    print(f"训练Loss: {outputs.loss.item():.4f}")
                    print(f"Logits形状: {outputs.logits.shape}")
                    
                except Exception as e:
                    print(f"训练时出错: {e}")
                
            print("\n" + "="*50 + "\n")
        
        print("=== 架构优势总结 ===")
        print("1. 职责分离: 模型专注于forward()，数据处理在Dataset中")
        print("2. 任务支持: 支持正向反应预测和逆合成预测")
        print("3. 代码复用: Dataset可被不同训练脚本复用")
        print("4. 维护性: 数据处理逻辑集中管理，便于调试")
        print("5. 梯度友好: 使用逐元素乘加替换，支持梯度正常回传")
        print("6. 简化接口: 基于哨兵token ID自动识别，无需手动位置信息")
        print("\n测试完成！")
        
    # 运行测试
    test_model_with_dataset()
