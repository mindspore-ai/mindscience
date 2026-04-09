---
name: afmultimer
description: AlphaFold Multimer is a protein complex structure prediction model. It extends the original AlphaFold 2 algorithm with specific modifications to handle multi-chain protein complexes and their binding interfaces.
license: MIT License (original), Apache License 2.0 (MindSpore implementation)
metadata:
    skill-author: MindSpore Science Team
---

# AlphaFold Multimer

## Model Description

AlphaFold Multimer is a protein complex structure prediction model. It extends the original AlphaFold 2 algorithm with specific modifications to handle multi-chain protein complexes and their binding interfaces.

Key features:
- Trains on PDB data with special consideration for chain coverage and diversity
- Uses FAPE (Frame aligned point error) as the structural scoring function
- Modifies loss functions to account for symmetric permutations in homomeric complexes
- Implements inter-chain co-evolution analysis
- Uses f_asym_id for chain encoding and f_entity_id for entity encoding
- Enhances model confidence by increasing weights for inter-residue interactions at binding interfaces

## Usage Instructions

### Dependencies
```txt
python == 3.9
mindspore == 2.8.0
mindformers == 1.0.0
CANN >= 8.2.RC1
```

### Installation
```bash
# Clone the code (r0.7 branch)
git clone -b r0.7 https://gitee.com/mindspore/mindscience.git

cd mindscience/
export PYTHONPATH=$PYTHONPATH:$PWD/MindSPONGE/src

# Install additional dependencies
pip install rdkit==2024.3.1 pyyaml sckit-learn pyparsing biopython
```

### Running the Model

#### Download Weights
```bash
wget https://download.mindspore.cn/mindscience/mindsponge/Multimer/checkpoint/Multimer_Model_1.ckpt
```

#### Download Example Features
```bash
wget https://download.mindspore.cn/mindscience/mindsponge/Multimer/examples/6T36.pkl
```

#### Example Code
```python
import os
import stat
import pickle
from mindsponge.common.protein import to_pdb_v2, from_prediction_v2
from mindsponge import PipeLine

pipe = PipeLine(name="Multimer")
pipe.set_device_id(0)
config_path = os.path.abspath("./MindSPONGE/applications/model_configs/Multimer/predict_256.yaml")
pipe.initialize(config_path=config_path)
pipe.model.from_pretrained("./")  # Can download weights automatically if not provided

with open("./6T36.pkl", "rb") as f:
    raw_feature = pickle.load(f)

final_atom_positions, final_atom_mask, confidence, b_factors = pipe.predict(raw_feature)
unrelaxed_protein = from_prediction_v2(final_atom_positions,
                                       final_atom_mask,
                                       raw_feature["aatype"],
                                       raw_feature["residue_index"],
                                       b_factors,
                                       raw_feature["asym_id"],
                                       False)
pdb_file = to_pdb_v2(unrelaxed_protein)
os.makedirs('./result/', exist_ok=True)
os_flags = os.O_RDWR | os.O_CREAT
os_modes = stat.S_IRWXU
pdb_path = './result/unrelaxed_6T36.pdb'
with os.fdopen(os.open(pdb_path, os_flags, os_modes), 'w') as fout:
     fout.write(pdb_file)
print("confidence:", confidence)
```

## Parameters
- `name`: Model name (should be "Multimer")
- `config_path`: Path to the configuration file (e.g., predict_256.yaml)
- `device_id`: GPU/Ascend device ID to use
- `raw_feature`: Input feature dictionary containing:
  - `aatype`: Amino acid types
  - `residue_index`: Residue indices
  - `asym_id`: Asymmetric unit IDs
  - Other required features loaded from pickle file

## Output
The model outputs:
- `final_atom_positions`: Predicted atomic coordinates
- `final_atom_mask`: Atom mask indicating valid positions
- `confidence`: Model confidence score
- `b_factors`: B-factors for each atom
- PDB file with the predicted structure

## License
Apache License 2.0 (MindSpore implementation)