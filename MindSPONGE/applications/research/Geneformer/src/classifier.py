# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=C0411
# pylint: disable=C0413

"""classifier script"""
import os
import time
import datetime
import logging
import pickle
import subprocess
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

import mindspore
from mindspore import Tensor, nn
from mindspore.ops import operations as P
import mindspore.dataset as ds
from mindformers.tools.register import MindFormerConfig
from mindformers import AutoConfig, Trainer, BertForTokenClassification

from . import TOKEN_DICTIONARY_FILE
from . import perturber_utils as pu
from . import classifier_utils as cu
from . import preparedata as pre
logger = logging.getLogger(__name__)


def pre_label(data):
    logits = Tensor(data, mindspore.float16)
    softmax = P.Softmax()
    probabilities = softmax(logits)
    predicted_label = probabilities.argmax()
    predicted_label = predicted_label.asnumpy().item()
    return predicted_label


class GeneClassifier(nn.Cell):
    """GeneClassifier"""
    def __init__(
            self,
            quantize=False,
            gene_class_dict=None,
            filter_data=None,
            rare_threshold=0,
            max_ncells=None,
            max_ncells_per_class=None,
            training_args=None,
            ray_config=None,
            freeze_layers=0,
            num_crossval_splits=1,
            train_size=0.8,
            valid_size=0.1,
            test_size=0.1,
            stratify_splits_col=None,
            no_eval=False,
            forward_batch_size=100,
            nproc=4,
            config_path=None,
            do_train=False
    ):
        """
            validate cell state or gene classifier.

        Args:
            quantize(bool): Whether to fine-tune a quantized model..
            gene_class_dict(dict): Gene classes to fine-tune model to distinguish.
            filter_data(dict):  Otherwise, dictionary specifying .dataset column name and list of values to filter by.
            rare_threshold(int): Threshold below which rare cell states should be removed.
            max_ncells(int): Maximum number of cells to use for fine-tuning.
            max_ncells_per_class(int): Maximum number of cells per cell class to use for fine-tuning.
            training_args(dict): Training arguments for fine-tuning.
            ray_config(dict): Training argument ranges for tuning hyperparameters with Ray.
            freeze_layers(int): Number of layers to freeze from fine-tuning.
            split_sizes(dict): Dictionary of proportion of data to hold out for train, validation, and test sets
            stratify_splits_col(dict): Proportion of each class in this column will
            be the same in the splits as in the original dataset.
            no_eval(bool): Will skip eval step and use all data for training.
            forward_batch_size(str): Batch size for forward pass (for evaluation, not training).
            nproc=(int): Number of CPU processes to use.
        Returns:
            GeneClassifier object
        """
        super(GeneClassifier, self).__init__()
        self.quantize = quantize
        self.gene_class_dict = gene_class_dict
        self.filter_data = filter_data
        self.rare_threshold = rare_threshold
        self.max_ncells = max_ncells
        self.max_ncells_per_class = max_ncells_per_class
        self.training_args = training_args
        self.ray_config = ray_config
        self.freeze_layers = freeze_layers
        self.num_crossval_splits = num_crossval_splits
        self.train_size = train_size
        self.valid_size = valid_size
        self.oos_test_size = test_size
        self.eval_size = self.valid_size / (self.train_size + self.valid_size)
        self.stratify_splits_col = stratify_splits_col
        self.no_eval = no_eval
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.config_path = config_path
        self.do_train = do_train

        # load token dictionary (Ensembl IDs:token)
        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            self.gene_token_dict = pickle.load(f)
        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}
        self.gene_class_dict = {
            k: list({self.gene_token_dict.get(gene) for gene in v})
            for k, v in self.gene_class_dict.items()
        }
        empty_classes = []
        for k, v in self.gene_class_dict.items():
            if v is not None:
                empty_classes += [k]

    def prepare_data(
            self,
            input_data_file,
            output_directory,
            output_prefix,
        ):
        """
          prepare_data

           Args:
                input_data_file(path): Path to directory containing .dataset input.
                output_directory(path): Path to directory where prepared data will be saved.
                output_prefix(str): Prefix for output file.
                balanced other attributes.
           Returns:
                Save processed data to dir.
        """
        data = pu.load_and_filter(self.filter_data, self.nproc, input_data_file)
        data, id_class_dict = cu.label_classes(
            data, self.gene_class_dict, self.nproc
        )
        id_class_output_path = (
            Path(output_directory) / f"{output_prefix}_id_class_dict"
        ).with_suffix(".pkl")
        with open(id_class_output_path, "wb") as f:
            pickle.dump(id_class_dict, f)
        data_output_path = (
            Path(output_directory) / f"{output_prefix}_labeled"
        ).with_suffix(".dataset")
        data.save_to_disk(str(data_output_path))


    def evaluate_model(
            self,
            model_directory,
            eval_data,
            eval_batch_size=16,
            max_len=2048,
            mask_label=-100,
    ):
        """
           evaluate_model

           Args:
                model_directory(dict): Path to directory where eval data will be saved.
                eval_data(dataset): Loaded evaluation .dataset input.
                eval_batch_size(int): Batch size to eval.
                max_len(int): Max len of data.
                mask_label(int): Mask numbers.
           Returns:
               A value of acc
        """
        token_type_ids = np.zeros((eval_batch_size, max_len), dtype=np.int32)
        token_type_ids = Tensor(token_type_ids, dtype=mindspore.int32)
        dataset = ds.GeneratorDataset(pre.generator_eval_data(eval_data, max_len),
                                      column_names=['input_ids', 'length', 'labels'])
        dataset = dataset.batch(eval_batch_size)
        geneformer_config = AutoConfig.from_pretrained(self.config_path)
        checkpoint_dir = os.path.join(model_directory, "checkpoint/rank_0/")

        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(checkpoint_dir + " dir not found")
        max_ckpt = pu.find_latest_ckpt(checkpoint_dir)
        print(f"use ckpt file: {max_ckpt}")
        geneformer_config.load_checkpoint = max_ckpt
        geneformer_config.checkpoint_name_or_path = max_ckpt
        eval_model = BertForTokenClassification(geneformer_config)
        eval_model.set_train(False)
        logit_list = []
        true_list = []
        eval_start_time = time.time()
        for data in dataset:
            input_data, label, mask = data
            input_data = input_data.asnumpy().tolist()
            input_data = Tensor(input_data, dtype=mindspore.int32)
            mask = Tensor(mask, dtype=mindspore.int32)
            output = eval_model(input_data, mask, token_type_ids)
            logit_list.extend(output.asnumpy())
            true_list.extend(label.asnumpy())
        logit_label_paired = [
            (logit, label)
            for batch_logit, batch_labels in zip(logit_list, true_list)
            for logit, label in zip(batch_logit, batch_labels)
            if label != mask_label
        ]
        pre_list = []
        for data in logit_label_paired:
            result = pre_label(data[0])
            pre_list.append(result)
        label_true = [item[1] for item in logit_label_paired]
        f1 = f1_score(label_true, pre_list, average='binary')
        acc = accuracy_score(label_true, pre_list)
        eval_end_time = time.time()
        print("eval_time: ", eval_end_time - eval_start_time)
        print(f"F1 Score: {f1}")
        print(f"Accuracy: {acc}")
        return acc

    def train_classifier(
            self,
            model_directory,
            train_data,
            eval_data,
            train_batch_size=12,
    ):

        """
        Fine-tune model for cell state or gene classification.

        Args:
            model_directory(dict): Path to directory containing model.
            num_classes(int): Number of classes for classifier.
            train_data(dataset): Loaded training .dataset input.
            eval_data(dataset): Loaded evaluation .dataset input.
            config_path(path): Model config path.
            train_batch_size(int): Batch size to train.
        Returns:
            A trainer object
        """

        # Validate and prepare data
        train_data, eval_data = cu.validate_and_clean_cols(
            train_data, eval_data
        )
        # Load model and training args
        model = pu.load_model(
            model_directory,
            "train",
            config_path=self.config_path
        )
        def_freeze_layers = self.freeze_layers
        if def_freeze_layers > 0:
            for param in model.get_parameters(expand=True):
                if param.name.startswith("bert.bert_encoder.encoder") and int(
                        param.name.split("bert.bert_encoder.encoder.blocks.")[1].split(".")[0]) < def_freeze_layers:
                    param.requires_grad = False

        my_data = pre.GeneratorTrainData(train_data)
        dataset = ds.GeneratorDataset(source=my_data,
                                      column_names=["input_ids", "input_mask", 'segment_ids', 'label_ids'])
        dataset = dataset.batch(train_batch_size)
        config = MindFormerConfig(self.config_path)
        config.output_dir = model_directory
        trainer = Trainer(task='token_classification',
                          model=model,
                          args=config,
                          train_dataset=dataset,
                          eval_dataset=dataset)
        if self.do_train:
            trainer.train()
        return trainer

    def validate(
            self,
            model_directory,
            prepared_input_data,
            id_class_dict_file,
            output_directory,
            output_prefix,
            gene_balance=False,
            predict_trainer=False,
            save_split_datasets=True,
            n_splits=2
    ):
        """
           (Cross-)validate cell state or gene classifier.

           Args:
                model_directory(path) to directory containing model.
                prepared_input_data(path): Path to directory containing _labeled.dataset
                previously prepared by Classifier.prepare_data.
                id_class_dict_file(path): Path to _id_class_dict.pkl previously prepared by Classifier.prepare_data.
                output_directory(path): Path to directory where model and eval data will be saved.
                output_prefix(str): Prefix for output files.
                gene_balance(bool): Whether to automatically balance genes in training set.
                predict_eval(bool): Whether or not to save eval predictions.
                predict_trainer(bool): Whether or not to save eval predictions from trainer.
                n_hyperopt_trials(int): Number of trials to run for hyperparameter optimization.
                save_split_datasets(bool): Whether or not to save train, valid, and test gene-labeled datasets.
                eval_batch_size(int): Batch size to eval.
                n_splits(int): Eval split number.
           Returns:
               A list of ACC.
           """
        # load numerical id to class dictionary (id:class)
        with open(id_class_dict_file, "rb") as f:
            id_class_dict = pickle.load(f)
        class_id_dict = {v: k for k, v in id_class_dict.items()}
        # load previously filtered and prepared data
        data = pu.load_and_filter(None, self.nproc, prepared_input_data)
        data = data.shuffle(seed=42)  # reshuffle in case users provide unshuffled data

        # define output directory path
        current_date = datetime.datetime.now()
        datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
        if output_directory[-1:] != "/":  # add slash for dir if not present
            output_directory = output_directory + "/"
        output_dir = f"{output_directory}{datestamp}_geneformer_geneClassifier_{output_prefix}/"
        subprocess.call(f"mkdir {output_dir}", shell=True)

        # get number of classes for classifier
        iteration_num = 1
        targets = pu.flatten_list(self.gene_class_dict.values())
        labels = pu.flatten_list(
            [
                [class_id_dict[label]] * len(targets)
                for label, targets in self.gene_class_dict.items()
            ]
        )
        skf = cu.StratifiedKFold3(n_splits=n_splits, random_state=0, shuffle=True)
        test_ratio = self.oos_test_size / (self.eval_size + self.oos_test_size)
        result_list = []
        for train_index, eval_index, test_index in tqdm(
                skf.split(targets, labels, test_ratio)
        ):
            train_data, eval_data = cu.gene_split_data(
                data,
                targets,
                labels,
                train_index,
                eval_index,
                self.max_ncells,
                iteration_num,
                self.nproc,
                gene_balance,
            )

            if save_split_datasets is True:
                for split_name in ["train", "valid"]:
                    labeled_dataset_output_path = (
                        Path(output_dir)
                        / f"{output_prefix}_{split_name}_gene_labeled_ksplit{iteration_num}"
                    ).with_suffix(".dataset")
                    if split_name == "train":
                        train_data.save_to_disk(str(labeled_dataset_output_path))
                    elif split_name == "valid":
                        eval_data.save_to_disk(str(labeled_dataset_output_path))

            if self.oos_test_size > 0:
                test_data = cu.gene_classifier_split(
                    data,
                    targets,
                    labels,
                    test_index,
                    "test",
                    self.max_ncells,
                    iteration_num,
                    self.nproc,
                )
                if save_split_datasets is True:
                    test_labeled_dataset_output_path = (
                        Path(output_dir)
                        / f"{output_prefix}_test_gene_labeled_ksplit{iteration_num}"
                    ).with_suffix(".dataset")
                    test_data.save_to_disk(str(test_labeled_dataset_output_path))
            self.train_classifier(
                model_directory,
                train_data,
                eval_data,
                predict_trainer,
            )
            result = self.evaluate_model(
                model_directory,
                eval_data,
            )
            result_list.append(result)
        return result_list
