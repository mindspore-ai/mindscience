import mindspore as ms
from mindspore import Tensor
from evo2 import Evo2

evo2_model = Evo2('evo2_7b', local_path='ckpt/evo2_7b_base_ms.ckpt')

sequence = 'ACGT'
input_ids = Tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=ms.int,
).unsqueeze(0)

layer_name = 'blocks.3.mlp.l3'

outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])

print('Embeddings shape: ', embeddings[layer_name].shape)