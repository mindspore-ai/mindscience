# ReactGen

## Background

The foundation model for chemical reaction generation is based on Qwen2.5-1.5B as the backbone, and uses the trained multimodal alignment module ReactionBERT as the encoder for SMILES input. The encoded output is projected by the `proj` layer and fused with the natural language embedding, and then decoded by Decoder to output SMILES. The model architecture is as follows:
```
SMILES ──▶ ReactionBERT ──┐ 
                           │ proj
NLP ──▶ Qwen2.5 embedding ┤────► Qwen2.5 Decoder 
```

After training, the model can be used for downstream tasks in the field of chemical reaction generation. The three downstream tasks: forward reaction generation, reverse synthesis, and solvent prediction.
* Forward reaction: Given input information of reactants, predict products.
* Retrosynthesis: Given the input information of the products, predict the reactants.
* Solvent prediction: Given the input information of the reactants, predict the solvent.

## Model Implementation

### Hardware Requirements

- Supports the `Ascend` backend, which can be specified at runtime through `--device_target`. The default is `Ascend`, which can be configured through config file.

### Version Requirements

- Requires `MindSpore==2.5.0`。

### Installation

- Install MindSpore: see the official guide at
  `https://www.mindspore.cn/install`
- Install Python dependencies:
  `pip install -r requirement.txt`

### Dataset and ckpt

The ckpts and datasets are located [ReactGen_ckpt](https://ai.gitcode.com/AI4Science/ReactGen_ckpt).

### Directory Structure
```
ReactGen
|   infer_entry.py      # Inference entry
|   README.md           # README(English)
|   README_CN.md        # README(Chinese)
|   requirements.txt    # Environment dependencies
|   train_entry.py      # Train entry
|   
+---config
|       forward_1.5b_inference.yaml         # Infernece configuration of forward reaction
|       forward_1.5b_training.yaml          # Training configuration of forward reaction
|       retrosynthesis_1.5b_inference.yaml  # Infernece configuration of retrosynthesis
|       retrosynthesis_1.5b_training.yaml   # Training configuration of retrosynthesis
|       solvent_1.5b_inference.yaml         # Infernece configuration of solvent
|       solvent_1.5b_training.yaml          # Training configuration of solvent reaction
|       
+---dataset
|       dataset.py          # Dataset processing
|       solvent_dataset.py  # Dataset processing for solvent
|       
+---eval
|       evaluation_metrics.py       # Metrics evaluation
|       
+---inference
|       inference_full_multi.py     # Inference
|       inference_solvent_multi.py  # Inference for solvent
|       
+---model
|       reactionqwen.py             # ReactGen arch
|       reactionqwen_solvent.py     # ReactGen + classifier
|       tokenizer.py                # Tokenizer
|       
+---train
|       train_full_multi.py         # Training
|       train_solvent_multi.py      # Training for solvent
|       
\---utils
        analyze_data_length.py      # Data analysis
        clean_solvent_dataset.py    # Cleaning of solvent dataset
        create_solvent_map.py       # Handling rare solvents
        process_predictions.py      # Metrics procssing
        solvent_dataset_process.py  # Solvent dataset processing
        analyze_solvent.py          # Solvent dataset analysis
        vocab_extension.py          # Vocab extension
```  

- The backbone of the ReactGen is `MultiModalQwen` (`model/reactionqwen.py`), which can be trained or inferred through the `train_entry.py` or the `infer_entry.py`.

## Running the Model

### Training

- Make sure the following preparations are completed:
    - MindSpore and all dependencies are installed.
    - Ckpts and datasets are downloaded. 
    - Training parameters are configured in `xx_1.5b_training.yaml`.
- Then run the following in the `reactgen` directory:

```bash
python train_entry.py --config config/forward_1.5b_training.yaml
```

### Inference

- Set the checkpoint path to load in the `inference.model_path` field of
  `xx_1.5b_inference.yaml`.
- Then run the following in the `reactgen` directory:

```bash
python infer_entry.py --config config/forward_1.5b_inference.yaml
```
The results will be saved in JSON format in the `xx_metrics.json` file located in the `results` directory. The format is as follows:

```
{
  "total_predictions": 39994,
  "valid_predictions": 39674,
  "exact_matches": 29363,
  "validity_rate": 0.991998799819973,
  "top1_accuracy": 0.7341851277691653,
  "valid_accuracy": 0.7401068709986389,
  "invalid_predictions": 320
}
```

## License

- License: `Apache License 2.0`
- License link：`http://www.apache.org/licenses/LICENSE-2.0`