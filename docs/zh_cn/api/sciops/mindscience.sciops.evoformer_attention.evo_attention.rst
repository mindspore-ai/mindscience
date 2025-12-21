mindscience.sciops.evoformer_attention.evo_attention
=====================================================

.. py:function:: mindscience.sciops.evoformer_attention.evo_attention(query, key, value, head_num, bias, attn_mask, scale_value, input_layout)

    使用自定义NPU算子执行evoformer注意力计算。

    此函数实现了Evoformer中的注意力机制，这是AlphaFold2等蛋白质结构预测模型中的关键组件。它计算注意力分数并将它们应用于值张量以产生输出。

    参数：
        - **query** (Tensor) - 用于注意力计算的查询张量。
        - **key** (Tensor) - 用于注意力计算的键张量。
        - **value** (Tensor) - 用于注意力计算的值张量。
        - **head_num** (int) - 注意力头的数量。
        - **bias** (Tensor) - 要添加到注意力分数的偏置张量。
        - **attn_mask** (Tensor) - 用于屏蔽某些位置的注意力掩码。
        - **scale_value** (float) - 应用于注意力分数的缩放因子。
        - **input_layout** (str) - 输入张量的布局（例如，'BHMK'或'BMKH'）。

    返回：
        Tensor。应用注意力机制后的输出张量。

    异常：
        - **RuntimeError** - 如果自定义算子加载或执行失败。
