# Copyright (c) 2024, Michael Poli.

from dataclasses import dataclass

import mindspore as ms
from mindspore import Tensor, mint, ops
import sys
import numpy as np

from vortex.model.sample import sample
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import print_rank_0


class Generator:
    def __init__(self, model, tokenizer, top_k=50, top_p=0.7, temperature=1):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.untils = ["\n\n"]

    def generate(
        self,
        device: str,
        input_string: str = None,
        input_ids: Tensor = None,
        num_tokens: int = 32,
        cached_generation: bool = True,
        force_prompt_threshold: int = None,
        max_seqlen: int = None,
        print_generation: bool = True,
        verbose: bool = False,
        skip_special_tokens: bool = False,
        stop_at_eos: bool = True,
        inference_params_dict: dict = None,
        token_callback=lambda i: None,
    ) -> tuple[Tensor, Tensor, dict]:
        """
        Generates using the model with optional cached sampling replay.

        This method enables passing in and returning the `inference_params_dict` for
        replaying cached sampling from a given state, for example for beam search.

        Args:
            device: The device to run the model on.
            input_string: The input prompt to generate from.
            input_ids: The input prompt token ids to generate from.
            num_tokens: The number of tokens to generate.
            cached_generation: Whether to use cached generation. Defaults to False.
            force_prompt_threshold: Number of tokens to prefill in parallel before
                switching to prompt forcing. Used to reduce peak memory usage and
                support longer prompts. Defaults to None.
            max_seqlen: Maximum sequence length to generate. Determines the max size
                of the cache if larger. Otherwise automatically determined using
                prompt length + max_tokens. Defaults to None.
            print_generation: Whether to print generated tokens. Defaults to False.
            verbose: Whether to print verbose output. Defaults to False.
            skip_special_tokens: Whether to skip special tokens. Defaults to True.
            stop_at_eos: Whether to stop generation at EOS token. Defaults to True.
            inference_params_dict: Dictionary of inference parameters to use for
                replaying cached sampling. Defaults to None.
            token_callback: Optional callback function called after each token is
                generated. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict]: A tuple containing:
                - generation: Generated token sequences of shape (batch_size, num_generated_tokens)
                - scores: Token generation scores/logits of shape (batch_size, num_generated_tokens, vocab_size)
                - inference_params_dict: The inference parameters dictionary used for generation, which can
                  be used to replay the exact same sampling sequence.
        """
        if isinstance(self.tokenizer.eos, int):
            #eos_token_ids = torch.LongTensor([self.tokenizer.eos]).to(device)
            eos_token_ids = Tensor([self.tokenizer.eos], dtype=ms.int64)
        else:
            #eos_token_ids = self.tokenizer.tokenize(self.tokenizer.eos).to(device)
            eos_token_ids = self.tokenizer.tokenize(self.tokenizer.eos)
            eos_token_ids = Tensor(eos_token_ids, dtype=ms.int64)
        if input_ids is None:
            input = self.tokenizer.tokenize(input_string)
            if isinstance(input, list):
                #input = torch.LongTensor(input).unsqueeze(0).to(device)
                input = Tensor(np.array(input)[None, :], dtype=ms.int64)
            else:
                #input = input.unsqueeze(0).to(device)
                input = input.expand_dims(0)
        else:
            input = input_ids
        x = input

        if max_seqlen is not None:
            x = x[:, -max_seqlen:]

        num_tokens = int(num_tokens)
        batch_size = x.shape[0]

        prompt_length = x.shape[1]
        prompt_forcing = inference_params_dict is None and force_prompt_threshold is not None and prompt_length > force_prompt_threshold
        if prompt_forcing:
            forced_prompt_length = prompt_length - force_prompt_threshold
            x_force = x[:, force_prompt_threshold:]
            x = x[:, :force_prompt_threshold]
        else:
            forced_prompt_length = 0
        tot_length = prompt_length + num_tokens
        if max_seqlen is not None:
            if max_seqlen > tot_length:
                tot_length = max_seqlen

        generation = mint.empty(
            (x.shape[0], num_tokens),
            dtype=ms.long,
        )

        scores = mint.empty(
            (x.shape[0], num_tokens, self.tokenizer.vocab_size),
            dtype=ms.float,
        )

        if inference_params_dict is not None:
            cached_generation = True
            prefilled = True
            # Ensure that the cached data is loaded on the correct device.
            if any(data.device != x.device for data in inference_params_dict["hcl"].fir_state_dict.values()):
                for key, data in inference_params_dict["mha"].key_value_memory_dict.items():
                    inference_params_dict["mha"].key_value_memory_dict[key] = data.move_to(x.device)
                for key, data in inference_params_dict["hcl"].fir_state_dict.items():
                    inference_params_dict["hcl"].fir_state_dict[key] = data.move_to(x.device)
                for key, data in inference_params_dict["hcl"].state_dict.items():
                    inference_params_dict["hcl"].state_dict[key] = data.move_to(x.device)
                for key, data in inference_params_dict["hcm"].fir_inner_state_dict.items():
                    inference_params_dict["hcm"].fir_inner_state_dict[key] = data.move_to(x.device)
                for key, data in inference_params_dict["hcm"].fir_state_dict.items():
                    inference_params_dict["hcm"].fir_state_dict[key] = data.move_to(x.device)
                for key, data in inference_params_dict["hcm"].state_dict.items():
                    inference_params_dict["hcm"].state_dict[key] = data.move_to(x.device)
                for key, data in inference_params_dict["hcs"].fir_state_dict.items():
                    inference_params_dict["hcs"].fir_state_dict[key] = data.move_to(x.device)
                for key, data in inference_params_dict["hcs"].fir_inner_state_dict.items():
                    inference_params_dict["hcs"].fir_inner_state_dict[key] = data.move_to(x.device)
                for key, data in inference_params_dict["hcs"].state_dict.items():
                    inference_params_dict["hcs"].state_dict[key] = data.move_to(x.device)
            inference_params_dict["mha"].max_batch_size = batch_size
        elif cached_generation:
            inference_params_dict = self.model.initialize_inference_params(max_seqlen=tot_length)
            inference_params_dict["mha"].max_batch_size = batch_size
            prefilled = False
        else:
            inference_params_dict = None
            prefilled = False

        if verbose:
            mem_after_tok = ms.runtime.memory_allocated() / 1e9  
            print_rank_0(f"Memory after tokenization: {mem_after_tok} GB")
            print_rank_0("Starting generation...")
            if input_string is not None:
                print_rank_0("Prompt: " + input_string)
            else:
                print_rank_0(f"Prompt ids: {input_ids} {input_ids.shape}")

        i = 0
        for i in range(forced_prompt_length + num_tokens):
            post_prefill = prefilled or (cached_generation and i > 0)

            # prefill then process only the last token
            if post_prefill:
                x = x[:, -1:]
                seqlen_offset = inference_params_dict["mha"].seqlen_offset

                if seqlen_offset == 0:
                    if prompt_forcing:
                        seqlen_offset = force_prompt_threshold
                    else:
                        seqlen_offset = input.shape[-1]
                    inference_params_dict["mha"].seqlen_offset = seqlen_offset
                    inference_params_dict["hcl"].seqlen_offset = seqlen_offset
                    inference_params_dict["hcm"].seqlen_offset = seqlen_offset
                    inference_params_dict["hcs"].seqlen_offset = seqlen_offset
                else:
                    inference_params_dict["mha"].seqlen_offset += 1
                    inference_params_dict["hcl"].seqlen_offset += 1
                    inference_params_dict["hcm"].seqlen_offset += 1
                    inference_params_dict["hcs"].seqlen_offset += 1

            # TODO: [Mindspore] should support give in inference_params_dict
            logits, _ = self.model.forward(x)

            token_callback(i)

            last_logits = logits[:, -1]

            if prompt_forcing and i < forced_prompt_length:
                new_idx = x_force[:, i]
            else:
                new_idx = sample(
                    last_logits,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                )

            if stop_at_eos and (generation[0, -1:] == eos_token_ids).all():
                print("Stopping generation at EOS")

            if print_generation and verbose and batch_size == 1:
                print(
                    f"{self.tokenizer.detokenize([new_idx.item()])}",
                    end=" ",
                    flush=True,
                )

            if prompt_forcing:
                if i >= forced_prompt_length:
                    scores[:, i - forced_prompt_length] = last_logits
                    generation[:, i - forced_prompt_length] = new_idx
            else:
                scores[:, i] = last_logits
                generation[:, i] = new_idx

            if post_prefill:
                x = new_idx[:, None]
            else:
               x = ms.mint.cat([x, new_idx[:, None]], dim=-1)

        return generation[:, : i + 1], scores[:, : i + 1], inference_params_dict


def logits_to_logprobs(logits: Tensor, tokens: Tensor) -> Tensor:
    """Convert logits to log probabilities."""
    probs = ops.LogSoftmax(axis=-1)(logits)
    return ops.gather_elements(probs, -1, tokens.expand_dims(-1)).squeeze(-1)


def prepare_batch(
    seqs: list[str], tokenizer: CharLevelTokenizer, prepend_bos: bool = False, device: str = "cuda:0"
) -> tuple[Tensor, list[int]]:
    """Prepare a batch of sequences for the model."""
    if prepend_bos:
        seqs = [tokenizer.bos + seq for seq in seqs]

    tokens = [tokenizer.tokenize(seq) for seq in seqs]
    if isinstance(tokens[0], list):
        tokens = [Tensor(t, dtype=ms.long) for t in tokens]

    max_len = max(len(t) for t in tokens)
    batch = mint.zeros((len(tokens), max_len), dtype=ms.long)

    for i, t in enumerate(tokens):
        batch[i, : len(t)] = t

    return batch, [len(t) for t in tokens]


@dataclass(kw_only=True)
class GenerationOutput:
    sequences: list[str]
    logits: list[Tensor]
    logprobs_mean: list[float]


def generate(
    *,
    prompt_seqs: list[str],
    model,
    tokenizer: CharLevelTokenizer,
    n_tokens: int = 100,
    temperature: float = 0.0,
    top_k: int = 1,
    top_p: float = 1.0,
    batched: bool = True,
    prepend_bos: bool = False,
    force_prompt_threshold: int = 3000,
    cached_generation: bool = True,
    verbose: int = 1,
    device: str = "GPU",
    **kwargs,
) -> GenerationOutput:
    """
    Performs generation from a list of prompts.
    If all prompts are the same length, this can do batched generation.
    Also supports cached generation for efficient sampling.
    """
    model.set_train(False)

    g = Generator(
        model,
        tokenizer,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    uniform_lengths = all(len(s) == len(prompt_seqs[0]) for s in prompt_seqs)

    if batched and uniform_lengths:
        input_ids_list = [
            prepare_batch(
                prompt_seqs,
                tokenizer,
                prepend_bos=prepend_bos,
                device=device,
            )[0]
        ]
    else:
        sys.stderr.write("WARNING: Batched generation is turned off.\n")
        input_ids_list = [
            prepare_batch(
                [prompt_seq],
                tokenizer,
                prepend_bos=prepend_bos,
                device=device,
            )[0]
            for prompt_seq in prompt_seqs
        ]

    generated_seqs, generated_scores, logitss = [], [], []
    for input_ids in input_ids_list:
        batch_size = input_ids.shape[0]

        output_ids, logits, _ = g.generate(
            input_ids=input_ids,
            num_tokens=n_tokens,
            device=device,
            print_generation=(verbose > 1),
            verbose=(verbose > 1),
            stop_at_eos=False,
            force_prompt_threshold=force_prompt_threshold,
            cached_generation=cached_generation,
            **kwargs,
        )

        if verbose > 1:
            print("input_ids.shape", input_ids.shape)
            print("output_ids.shape", output_ids.shape)
            print("logits.shape", logits.shape)

        generated_seqs_batch = list(tokenizer.detokenize_batch(output_ids))
        assert len(generated_seqs_batch) == batch_size
        generated_seqs += generated_seqs_batch
        logitss.append(logits)

        logprobs = logits_to_logprobs(logits, output_ids)
        logprobs = logprobs.astype(ms.float).asnumpy()

        generated_scores += [np.mean(logprobs[idx]) for idx in range(batch_size)]

    assert len(generated_seqs) == len(generated_scores) == len(prompt_seqs)
    if verbose:
        for seq, score, prompt in zip(generated_seqs, generated_scores, prompt_seqs):
            print(f'Prompt: "{prompt}",\tOutput: "{seq}",\tScore: {score}')

    return GenerationOutput(
        sequences=generated_seqs,
        logits=logitss,
        logprobs_mean=generated_scores,
    )
