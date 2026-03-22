import mindspore.mint as mint
from mindspore import Tensor



# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L231
def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done in-place."""
    indices_to_remove = logits < mint.topk(logits, top_k)[0][..., -1, None]
    logits=logits.masked_fill(indices_to_remove, float("-Inf"))
    return logits


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = mint.sort(logits, descending=False)
    cumulative_probs = mint.nn.functional.softmax(sorted_logits,dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = mint.scatter(mint.zeros_like(sorted_indices_to_remove),1, sorted_indices, sorted_indices_to_remove)
    logits=logits.masked_fill_(indices_to_remove, float("-inf"))
    return logits


# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py
def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    logits = mint.nan_to_num(logits)
    logits = mint.where(logits == float("-inf"), 0, logits)
    logits = mint.where(logits == float("inf"), 0, logits)

    if top_k == 1:  # Short-circuit for greedy decoding
        return mint.argmax(logits,dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])  # Safety check
            logits_top, indices = mint.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top = logits_top / temperature
            logits_top = modify_logits_for_top_p_filtering(logits_top, top_p)

            return indices[
                mint.arange(indices.shape[0]).to(indices.device),
                mint.multinomial(mint.nn.functional.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
            ]
        else:
            # Clone so that when we modify for top_p we don't change the original logits
            if temperature != 1.0:
                logits_top = logits / temperature
            else:
                logits_top = mint.clone(logits)
            logits_top = modify_logits_for_top_p_filtering(logits_top, top_p)
            return mint.multinomial(mint.nn.functional.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)
