from evo2 import Evo2

evo2_model = Evo2('evo2_7b', local_path='ckpt/evo2_7b_base_ms.ckpt')

output = evo2_model.generate(prompt_seqs=["ACGT"], n_tokens=400, temperature=1.0, top_k=4)

print(output.sequences[0])