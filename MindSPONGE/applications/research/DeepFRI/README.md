ENGLISH|[简体中文](README_CN.md)

# DeepFRI

## Introduction

This project uses the MindSpore framework to reproduce the DeepFRI model.
DeepFRI is a Graph Convolutional Network for predicting protein functions by
leveraging sequence features extracted from a protein language model and protein structures.
It makes four predictions about proteins:
Molecular Function (MF), Cellular Component (CC), Biological Process (BP), Enzyme Commission (EC).

* MF, CC and BP are the three independent ontological vocabularies of Gene Ontology.
  GO is an internationally standardized classification system of gene function,
  which provides a set of dynamic and controllable vocabulary to comprehensively describe the properties of genes and gene products in organisms.
  It is composed of a set of pre-defined GO term, which defines and describes the function of gene products.
  GO terms describe the product of a gene, not the gene itself,
  because sometimes there is more than one product of a gene, and GO name is the specific name of the GO term.
  DeepFRI outputs the GO term and GO name corresponding to MF, CC, and BP.

* EC numbers are a set of numbering systems developed by the Enzyme Commission to classify enzymes based on the chemical reactions that each enzyme catalyzes.
  This classification also gives a suggested name for each enzyme, so it is also called the Enzyme Commission nomenclature.
  For the EC, DeepFRI directly outputs the EC number.

<div align=center>
<img src="../../../docs/deepfri_pipeline.png" alt="DeepFRI Pipeline" width="600"/>
</div>

Address of Reference paper：[DeepFRI](https://www.nature.com/articles/s41467-021-23303-9)

Address of Reference github：[DeepFRI](https://github.com/flatironinstitute/DeepFRI)

## Environment

The recommended environment version for this project is:

* mindspore 2.0.0

## Code Contents

```bash
├── DeepFRI
    ├── predict.py                                            // DeepFRI Inference Code
    ├── train.py                                              // DeepFRI Train Code
    ├── utils.py                                              // DeepFRI utils
    ├── requirements.txt                                      // DeepFRI环境要求
    ├── README_CN.md                                          // DeepFRI's Readme in Chinese
    ├── README.md                                             // DeepFRI's Readme in English
    ├── config
        ├── DeepFRI_cellular_component_model_params.json      // Parameter configuration of the model cc
        ├── DeepFRI_enzyme_commission_model_params.json       // Parameter configuration of the model ec
        ├── DeepFRI_molecular_function_model_params.json      // Parameter configuration of the model mf
        ├── DeepFRI_biological_process_model_params.json      // Parameter configuration of the model bp
        ├── model_config.json                                 // Overall parameter configuration of four models
    ├── examples                                              // Input sample
    ├── model
        ├── deppfri.py                                        // DeepFRI Model
        ├── predictor.py                                      // DeepFRI Predictor
    ├── module
        ├── layers.py                                         // DeepFRI Layers
    ├── trained_models                                        // The four pre-trained models are stored in this folder
    ├── output                                                // The model output files are stored in this folder
        ├── checkpoints                                       // Trained checkpoints will be stored in this folder
    ├── scripts                                               // The model scripts are stored in this folder
        ├── pred_cc.sh                                        // DeepFRI Test Script for cc（PDB）
        ├── pred_mf.sh                                        // DeepFRI Test Script for mf（PDB）
        ├── pred_bp.sh                                        // DeepFRI Test Script for bp（PDB）
        ├── pretrained_ckpt_train.sh                          // DeepFRI Train Script for pre-trained mf model（SWISS-MODEL）
        ├── train_cc.sh                                       // DeepFRI Train Script for cc（SWISS-MODEL）
        ├── train_mf.sh                                       // DeepFRI Train Script for mf（SWISS-MODEL）
        ├── train_bp.sh                                       // DeepFRI Train Script for bp（SWISS-MODEL）
```

## Run Code

If you want to run DeepFRI project, you can download the `examples.zip` and trained models from the end of the file.
You should make `trained_models` directory save models and extract the `examples.zip` to it.
The DeepFRI projects will make `output` directory save results if you do not assign the path of project output.
And it will automatically make `checkpoints` directory in `output` save the trained models.

```bash
Usage:python predict.py --cmap ./examples/pdb_cmaps/1S3P-A.npz -ont mf --verbose

options：
--cmap               input Protein contact map (*npz) or protein PDB file (*pdb)
--npz_dir            input the directory with *npz files of protein contact map
--pdb_dir            input the directory of PDB files including prediction structure of Rosetta/DMPFold
--save_path          assign the directory to save output files (default './output')
--ontology           switch different tasks (mf, ec, cc, bp)
--verbose            whether display prediction results(action="store_true")
--evaluation_path    calculate the precision and recall of different threshold on val dataset
--device_target      assign operation platform(Ascend, GPU, CPU)
--device_id          assign operation device(default: 0)
```

You can modify `./config/model_config.json` or replace models in `./trained_models` if you want to use trained models by yourself.

The partial operation results of the DeepFRI project as follows if you use models of projects.

### Option 1: predicting functions of a protein from its sequence

Example: predicting MF-GO terms for Parvalbumin alpha protein using its sequence (PDB: [1S3P](https://www.rcsb.org/structure/1S3P)):

```bash
>> python predict.py --cmap ./examples/pdb_cmaps/1S3P-A.npz -ont mf --verbose

```

### Output

```txt
Protein GO-term/EC-number Score GO-term/EC-number name
query_prot GO:0005509 0.99995 calcium ion binding
```

### Option 2: predicting functions of proteins from contact map catalogue

```bash
>> python predict.py --npz_dir examples/pdb_cmaps -ont mf -v

```

### Output

```txt
Protein GO-term/EC-number Score GO-term/EC-number name
pdb_cmaps\1S3P-A GO:0005509 0.99995 calcium ion binding
pdb_cmaps\2J9H-A GO:0004364 0.97407 glutathione transferase activity
pdb_cmaps\2J9H-A GO:0016765 0.88968 transferase activity, transferring alkyl or aryl (other than methyl) groups
pdb_cmaps\2J9H-A GO:0042277 0.40748 peptide binding
...
pdb_cmaps\2W83-E GO:0016818 0.83582 hydrolase activity, acting on acid anhydrides, in phosphorus-containing anhydrides
pdb_cmaps\2W83-E GO:0016817 0.83478 hydrolase activity, acting on acid anhydrides
pdb_cmaps\2W83-E GO:0003924 0.80225 GTPase activity
pdb_cmaps\2W83-E GO:0019899 0.13060 enzyme binding
```

### Option 3: predicting functions of a protein from a directory with PDB files

```bash
>> python predict.py --pdb_dir ./examples/pdb_files -ont mf

```

### Output

See files in: `./output/`

## Train

If you do not want to use trained models from this projects, the first thing is for you to download **`DeepFRI_LSTM.ckpt`** file.
The parameters of LSTM layer will be fixed during training DeepFRI.
Thanks for the work of [DeepFRI-Paper](https://www.nature.com/articles/s41467-021-23303-9).
The parameters of LSTM layer transfer from the lstm_lm_tf.hdf5 of [Newest Models](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/newest_trained_models.tar.gz)
(All LSTM layer parameters is same as lstm_lm_tf.hdf5 model).

```bash
Usage:python train.py -device "Ascend" -id 0 -ont mf -out "./output_mf" \
--pretrained_ckpt_path "./trained_models/DeepFRI_molecular_function.ckpt"

options:
--epochs                 number of epochs to train(have early stop)
--ontology               chose the training type(mf, ec, cc, bp)
--device_target          assign training platform(Ascend, GPU, CPU)
--device_id              assign training device(default: 0)
--output_dir             assign the output path of checkpoints(default: './output/checkpoints/')
--pretrained_ckpt_path   assign pretraining model(The parameters of LSTM layer is fixed in pretraining model)
```

You can modify `./config/..._model_params.json` file to change the parameters of models and training parameters.
You can also run `./scripts/train_xx.sh` to train a new model or use `pretrained_ckpt_train.sh` to train the pretraining model.
You can run `./script/pred_xx.sh` to calculate the precision and recall of different threshold.

Some parameters of `./config/..._model_params.json` are described as follows:

* gc_dims: MultiGraphConv, dimension setting of three layer (default: [512, 512, 512])
* fc_dims: dimension setting of fully connected layer
* pad_len: the function to pad cmap (deault: 1024)
* goterms: GO-term/EC-number
* gonames: GO-name/EC-number
* cmap_type: the type of Contact maps, choices=['ca', 'cb'], (default: ca)
* cmap_thresh: threshold (default: 10.0)

*Note：We do not train the ec model because the **SWISS-MODEL-EC.tar.gz** dataset link is useless. But the prediction function of ec is saved.*

## Test Example、Dateset、Models

The datasets used for training were built from selected entries in the PDB database and SWISS-MODEL database, respectively.
The author select annotated PDB chains and SWISS-MODEL chains, remove identical and similar sequences,
and create non-redundant sets by clustering all PDB chains and SWISS-MODEL chains (for which are able to retrieve contact maps)
by blastclust at 95% sequence identity (number of identical residues out of the total number of residues in the sequence alignment).

* The PDB(Protein Data Bank) is the single worldwide archive of structural data of biological macromolecules. The PDB is maintained by RSCB(Research Collaboratory for Structural Bioinformatics).

* SWISS-MODEL knowledge base is a dataset of 3D structure of protein, all protein structure in this dataset is built by SWISS-MODEL homology-modelling method.

The datasets created by author of this paper are provided by the type of 'TFRecord'. The following website can download the datasets.

| moduls   | File Name        | Size | Description  |Model URL  |
|-----------|---------------------|---------|---------------|-----------------------------------------------------------------------|
| examples | `examples.zip` | 1.7MB | examples for test, including PDB file、npz file |  [Download Link](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/examples/examples.zip) |
| PDB Dataset | `PDB-GO.tar.gz` | 19GB | PDB-GO dataset, used for mf、bp、cc training | [Download Link](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/PDB-GO.tar.gz) |
| PDB Dataset | `PDB-EC.tar.gz` | 13GB | PDB-EC dataset，used for ec training |  [Download Link](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/PDB-EC.tar.gz) |
| SWISS-MODEL Dataset | `SWISS-MODEL-GO.tar.gz` | 165GB | SWISS-MODEL dataset，used for mf、bp、cc training |  [Download Link](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/SWISS-MODEL-GO.tar.gz) |
| SWISS-MODEL Dataset | `SWISS-MODEL-EC.tar.gz` | 117GB | SWISS-MODEL dataset，used for ec training |  [Download Link](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/SWISS-MODEL-EC.tar.gz) |
| BP Model | `DeepFRI_biological_process.ckpt` | 74.5MB | The link of checkpoint trained by merged datasets(PDB & SWISS-MODEL) for BP task | [Download Link](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/checkpoint/DeepFRI_biological_process.ckpt) |
| CC Model | `DeepFRI_cellular_component.ckpt` | 40.6MB | The link of checkpoint trained by merged datasets(PDB & SWISS-MODEL) for CC task | [Download Link](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/checkpoint/DeepFRI_cellular_component.ckpt) |
| MF Model | `DeepFRI_molecular_function.ckpt` | 42.0MB | The link of checkpoint trained by merged datasets(PDB & SWISS-MODEL) for MF task | [Download Link](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/checkpoint/DeepFRI_molecular_function.ckpt) |
| The Parameters of LSTM layer | `DeepFRI_LSTM.ckpt` | 20.0MB | The link of parameters of LSTM layer | [Download Link](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/checkpoint/DeepFRI_LSTM.ckpt) |
