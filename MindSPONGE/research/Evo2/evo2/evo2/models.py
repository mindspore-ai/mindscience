from functools import partial
from mindspore import Tensor, load_checkpoint, load_param_into_net
from typing import List, Tuple, Dict
import yaml

from vortex.model.generation import generate as vortex_generate
from vortex.model.model import StripedHyena
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import dotdict

from evo2.scoring import score_sequences, score_sequences_rc
from evo2.utils import MODEL_NAMES, CONFIG_MAP

class Evo2:
    def __init__(self, model_name: str = MODEL_NAMES[1], local_path: str = None):
        """
        Load an Evo 2 checkpoint.

        Uses local_path if specified, otherwise checks if in local HuggingFace ~cache.
        Automatically downloads checkpoint from HuggingFace if it does not exist locally.

        Vortex automatically handles device placement on CUDA, and splits model across
        multiple GPUs if available.
        For models split across multiple GPUs, you can specify which GPUs to use with
        CUDA_VISIBLE_DEVICES. If using multi-gpu, do not use .to(device) manually.

        Notes:
        Evo 2 40b is too large to fit on a single H100 GPU, so needs multiple GPUs.
        You can change where HuggingFace downloads to by setting the HF_HOME environment
        variable.
        """
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        config_path = CONFIG_MAP[model_name]  # evo2_7b: configs/evo2-7b-1m.yml    evo2_7b_base: configs/evo2-7b-8k.yml

        if local_path is not None:
            self.model = self.load_evo2_model(None, config_path, local_path)
        else:
            self.model = self.load_evo2_model(model_name, config_path)
        
        self.tokenizer = CharLevelTokenizer(512)
    
    def forward(
        self,
        input_ids: Tensor,
        return_embeddings: bool = False,
        layer_names=None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with optional embedding extraction.
        
        Args:
            input_ids: Input token IDs
            return_embeddings: If True, returns embeddings from specified layers
            layer_names: List of layer names to extract embeddings from. Required if
                return_embeddings=True
            
        Returns:
            Tuple of (logits, embeddings_dict) if return_embeddings=True
            Tuple of (logits, None) otherwise
        """
        embeddings = {}
        handles = []
        
        if return_embeddings:  # False
            if layer_names is None:
                raise ValueError(
                    "layer_names must be specified when return_embeddings=True. Look at "
                    "evo2_model.model.state_dict().keys() to see available layers."
                )
                
            def hook_fn(layer_name):
                def hook(_, __, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    embeddings[layer_name] = output
                return hook
                
            # Register hooks for requested layers
            for name in layer_names:
                layer = self.model.get_sub_cell(name)
                handles.append(layer.register_forward_hook(hook_fn(name)))

        try:
            # Original forward pass
            logits = self.model.forward(input_ids)
            
            if return_embeddings:  # False
                return logits, embeddings
            return logits, None

        finally:
            for handle in handles:
                handle.remove()

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        return self.forward(input_ids, return_embeddings, layer_names)

    def score_sequences(
        self,
        seqs: List[str],
        batch_size: int = 1,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        average_reverse_complement: bool = False,
    ) -> List[float]:
        scoring_func = partial(
            score_sequences_rc if average_reverse_complement else score_sequences,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
        )

        try:
            scores = scoring_func(seqs)
        except Exception as e:
            raise RuntimeError(f"Error during sequence scoring: {str(e)}") from e

        return scores

    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: int = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate sequences from a list of prompts.

        force_prompt_threshold: If specified, avoids OOM errors through teacher forcing if the prompt is longer than this threshold.

        If force_prompt_threshold is none, sets default assuming 1xH100 (evo2_7b) and 2xH100 (evo2_40b) to help avoid OOM errors.
        """

        output = vortex_generate(
            prompt_seqs=prompt_seqs,
            model=self.model,
            tokenizer=self.tokenizer,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batched=batched,
            cached_generation=cached_generation,
            verbose=verbose,
            force_prompt_threshold=force_prompt_threshold,
        )
        return output


    def load_evo2_model(
            self,
            model_name: str = MODEL_NAMES[1],
            config_path: str = None,
            local_path: str = None,
            remove_shards: bool = True,
    ):
        """
        Load HuggingFace checkpoint using StripedHyena 2.

        If local_path is specified, loads from local_path.
        Otherwise, downloads from HuggingFace.
        If remove_shards is True, removes HF checkpoint shards after merging to .pt file.
        """
        if local_path is not None:
            print(f"Loading model from {local_path}...")
            print(f"Loading config from {config_path}...")
            config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
            model = StripedHyena(config)
            param_dict = load_checkpoint(local_path)
            print(f"Loading param into net...")
            param_not_load, _ = load_param_into_net(model, param_dict)
            if len(param_not_load) > 0:
                raise RuntimeError("The following parameters were not loaded successfully: " + str(param_not_load))
            return model
