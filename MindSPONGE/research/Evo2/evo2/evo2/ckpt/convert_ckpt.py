'''
Convert evo2_7b_base huggingface checkpoint to mindspore readable version.
'''

import torch
import mindspore as ms

state_dict = torch.load('./ckpt/evo2_7b_base.pt', map_location='cpu', weights_only=False)
key_to_del = set()

# Exclude layers beyond include_layers.
# To contain the front 4 layers, set as 4. Or all layers as 0.
include_layers = 0

for k, v in state_dict.items():
    if not isinstance(v, torch.Tensor):
        key_to_del.add(k)
    for i in range(include_layers, 32):
        if f"blocks.{i}" in k:
            key_to_del.add(k)

# delete keys
for key in key_to_del:
    state_dict.pop(key, None)

def update_ms_dict(ms_dict, k, v):
    if v.dtype is ms.bfloat16:
        v = v.float()
        ms_dict[k] = ms.Tensor(v.detach().cpu().numpy()).astype(ms.bfloat16)
    else:
        ms_dict[k] = ms.Tensor(v.detach().cpu().numpy())

# Convert the remained items to mindspore instances.
ms_dict = {}
for k, v in state_dict.items():
    if k == "embedding_layer.weight":
        update_ms_dict(ms_dict, "embedding_layer.embedding_table", v)
    else:
        update_ms_dict(ms_dict, k, v)

param_dict = {k: ms.Parameter(v, name=k) for k, v in ms_dict.items()}

# Save as mindspore checkpoint
ms.save_checkpoint(param_dict, './ckpt/evo2_7b_base_ms.ckpt')
print("Saved to ckpt/evo2_7b_base_ms.ckpt")

