import mindspore as ms
from mindspore import Tensor
from evo2 import Evo2

ms.set_context(device_id=0)

evo2_model = Evo2('evo2_7b', local_path='ckpt/evo2_7b_base_ms.ckpt')

sequence = 'ACGT'
input_ids = Tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=ms.int32,
).unsqueeze(0)

outputs, _ = evo2_model(input_ids)
logits = outputs[0]

print('Logits: ', logits)
print('Shape (batch, length, vocab): ', logits.shape)
