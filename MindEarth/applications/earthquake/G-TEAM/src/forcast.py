# Copyright 2023 Huawei Technologies Co., Ltd
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
# ==============================================================================
"GTeam forcast"
import os
from tqdm import tqdm
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.utils import (
    predict_at_time,
    calc_mag_stats,
    calc_loc_stats,
    calc_pga_stats,
)
from src.data import DataGenerator, PreloadedEventGenerator, load_pickle_data, load_data
from src.models import SingleStationModel
from src.utils import evaluation, seed_np_tf
from src.visual import generate_true_pred_plot


class CustomWithLossCell(nn.Cell):
    """
    A neural network cell that wraps a main network and loss function together,
    allowing the entire forward pass including loss computation to be treated as a single cell.

    This class combines a neural network model and a loss function into a single computation unit,
    which is useful for training loops and model encapsulation in deep learning frameworks.

    Attributes:
        net (nn.Cell): The main neural network model whose output will be used in loss computation.
        loss_fn (nn.Cell): The loss function cell that computes the difference between predictions
                           and true labels.
    """

    def __init__(self, net, loss_fn):
        """
        Initializes the CustomWithLossCell with a network model and loss function.

        Args:
            net (nn.Cell): The neural network model whose output will be used for loss calculation.
            loss_fn (nn.Cell): The loss computation function that takes (true_labels, predictions)
                              and returns a scalar loss value.
        """
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn

    def construct(self, x, y):
        '''
        Computes the loss by first passing input data through the network and then applying the loss function.

        Args:
            X (Tensor): Input data tensor containing features.
            y (Tensor): Ground truth labels tensor.

        Returns:
            Tensor: Computed loss value.

        Note:
            The input labels 'y' are squeezed along dimension 2 to match the output shape from the network.
            This ensures the loss function receives inputs of the expected shape.
        '''
        outputs = self.net(x)
        return self.loss_fn(y.squeeze(2), outputs)


class GTeamInference:
    """
    Initialize the GTeamInference class.
    """

    def __init__(self, model_ins, cfg, output_dir, logger):
        """
        Args:
            model_ins: The model instance used for inference.
            cfg: Configuration dictionary containing model and data parameters.
            output_dir: Directory to save the output results.
        Attributes:
            model: The model instance for inference.
            cfg: Configuration dictionary.
            output_dir: Directory to save outputs.
            pga: Flag indicating if PGA (Peak Ground Acceleration) is enabled.
            generator_params: Parameters for data generation.
            model_params: Parameters specific to the model.
            mag_key: Key for magnitude-related data.
            pos_offset: Position offset for location predictions.
            mag_stats: List to store magnitude prediction statistics.
            loc_stats: List to store location prediction statistics.
            pga_stats: List to store PGA prediction statistics.
        """
        self.model = model_ins
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = logger
        self.pga = cfg["model"].get("pga", "true")
        self.generator_params = cfg["data"]
        self.model_params = cfg["model"]
        self.output_dir = output_dir
        self.mag_key = self.generator_params["key"]
        self.pos_offset = self.generator_params["pos_offset"]
        self.mag_stats = []
        self.loc_stats = []
        self.pga_stats = []

    def _parse_predictions(self, pred):
        """
        Parse the raw predictions into magnitude, location, and PGA components.
        """
        mag_pred = pred[0]
        loc_pred = pred[1]
        pga_pred = pred[2] if self.pga else []
        return mag_pred, loc_pred, pga_pred

    def _process_predictions(
            self, mag_pred, loc_pred, pga_pred, time, evt_metadata, pga_true
    ):
        """
        Process the parsed predictions to compute statistics and generate plots.
        """
        mag_pred_np = [t[0].asnumpy() for t in mag_pred]
        mag_pred_reshaped = np.concatenate(mag_pred_np, axis=0)

        loc_pred_np = [t[0].asnumpy() for t in loc_pred]
        loc_pred_reshaped = np.array(loc_pred_np)

        pga_pred_np = [t.asnumpy() for t in pga_pred]
        pga_pred_reshaped = np.concatenate(pga_pred_np, axis=0)
        pga_true_reshaped = np.log(
            np.abs(np.concatenate(pga_true, axis=0).reshape(-1, 1))
        )

        if not self.model_params["no_event_token"]:
            self.mag_stats += calc_mag_stats(
                mag_pred_reshaped, evt_metadata, self.mag_key
            )

            self.loc_stats += calc_loc_stats(
                loc_pred_reshaped, evt_metadata, self.pos_offset
            )

            generate_true_pred_plot(
                mag_pred_reshaped,
                evt_metadata[self.mag_key].values,
                time,
                self.output_dir,
            )
        self.pga_stats = calc_pga_stats(pga_pred_reshaped, pga_true_reshaped)

    def _save_results(self):
        """
        Save the final results (magnitude, location, and PGA statistics) to a JSON file.
        """
        times = self.cfg["model"].get("times")
        self.logger.info("times: {}".format(times))
        self.logger.info("mag_stats: {}".format(self.mag_stats))
        self.logger.info("loc_stats: {}".format(self.loc_stats))
        self.logger.info("pga_stats: {}".format(self.pga_stats))

    def test(self):
        """
        Perform inference for all specified times, process predictions, and save results.
        This method iterates over the specified times, performs predictions, processes
        the results, and saves the final statistics.
        """
        data_data, evt_key, evt_metadata, meta_data, data_path = load_data(self.cfg)
        pga_true = data_data["pga"]
        for time in self.cfg["model"].get("times"):
            pred = predict_at_time(
                self.model,
                time,
                data_data,
                data_path,
                evt_key,
                evt_metadata,
                config=self.cfg,
                pga=self.pga,
                sampling_rate=meta_data["sampling_rate"],
            )
            mag_pred, loc_pred, pga_pred = self._parse_predictions(pred)
            self._process_predictions(
                mag_pred, loc_pred, pga_pred, time, evt_metadata, pga_true
            )
        self._save_results()
        print("Inference completed and results saved")

class GTeamTrain:
    """
    A class to handle the training of a full model for earthquake detection and localization.
    It manages data loading, training of single-station models, and full-model training.
    """
    def __init__(self, model_ins, cfg, output_dir, logger):
        """
        Initialize the GTeamTrain class with model, configuration, output directory, and logger.
        Args:
            model_ins (nn.Cell): The full model instance to be trained.
            cfg (dict): Configuration dictionary containing training parameters and paths.
            output_dir (str): Directory to save checkpoints and outputs.
            logger (logging.Logger): Logger instance for logging messages.
        """
        self.full_model = model_ins
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = logger
        self.waveform_shape = [3000, 3]
        self.training_params = self.cfg['training_params']
        self.generator_params = self.training_params.get('generator_params', [self.training_params.copy()])
        self.file_basename = os.path.basename(self.training_params['data_path']).split('.')[0]

    def load_train_data(self):
        """
        Load training data from a pickle file.
        Returns:
            Data structure: The loaded training data.
        """
        data_path = self.cfg['data']["root_dir"]
        filename_train = os.path.join(data_path, f"{self.file_basename}_train.pkl")
        return load_pickle_data(filename_train)

    def load_val_data(self):
        """
        Load validation data from a pickle file.
        Returns:
            Data structure: The loaded validation data.
        """
        data_path = self.cfg['data']["root_dir"]
        filename_val = os.path.join(data_path, f"{self.file_basename}_val.pkl")
        return load_pickle_data(filename_val)

    def init_single_generator(self, sampling_rate, event_metadata_index_train, event_key_train,
                              event_metadata_index_val, event_key_val, decimate_train):
        """
        Initialize the single-station model and its data generators for training and validation.
        Args:
            sampling_rate (float): Sampling rate of the seismic data.
            event_metadata_index_train (list): Indices for training events in the metadata.
            event_key_train (str): Key for selecting the training event data.
            event_metadata_index_val (list): Indices for validation events in the metadata.
            event_key_val (str): Key for selecting the validation event data.
            decimate_train (bool): Whether to decimate the training data.
        """
        self.single_station_model = SingleStationModel(output_mlp_dims=self.cfg['model']['output_mlp_dims'],
                                                       use_mlp=self.cfg['model']['use_mlp'])
        noise_seconds = self.generator_params[0].get('noise_seconds', 5)
        cutout = (sampling_rate * (noise_seconds + self.generator_params[0]['cutout_start']),
                  sampling_rate * (noise_seconds + self.generator_params[0]['cutout_end']))
        self.single_train_generator = DataGenerator(self.training_params['data_path'],
                                                    event_metadata_index_train, event_key_train,
                                                    mag_key=self.generator_params[0]['key'],
                                                    batch_size=self.generator_params[0]['batch_size'],
                                                    cutout=cutout,
                                                    label_smoothing=True,
                                                    sliding_window=self.generator_params[0].get('sliding_window',
                                                                                                False),
                                                    decimate=decimate_train)
        self.single_validation_generator = DataGenerator(self.training_params['data_path'],
                                                         event_metadata_index_val, event_key_val,
                                                         mag_key=self.generator_params[0]['key'],
                                                         batch_size=self.generator_params[0]['batch_size'],
                                                         cutout=cutout,
                                                         label_smoothing=True,
                                                         sliding_window=self.generator_params[0].get('sliding_window',
                                                                                                     False),
                                                         decimate=decimate_train)
        optimizer_single = nn.Adam(self.single_station_model.trainable_params(), learning_rate=1e-4)
        self.criterion_single_mse = nn.MSELoss()

        loss_net = CustomWithLossCell(self.single_station_model, self.criterion_single_mse)
        self.single_train_network = nn.TrainOneStepCell(loss_net, optimizer_single)

        self.single_station_model.set_train(True)

    def single_station_train(self, sampling_rate, event_metadata_index_train, event_key_train,
                             event_metadata_index_val, event_key_val, decimate_train):
        """
        Train the single-station model. Loads a pre-trained model if specified, otherwise
        initializes the generator and trains from scratch.
        Args:
            sampling_rate (float): Sampling rate of the seismic data.
            event_metadata_index_train (list): Indices for training events in the metadata.
            event_key_train (str): Key for selecting the training event data.
            event_metadata_index_val (list): Indices for validation events in the metadata.
            event_key_val (str): Key for selecting the validation event data.
            decimate_train (bool): Whether to decimate the training data.
        """
        if 'single_station_model_path' in self.training_params:
            print('Loading single station model')
            param_dict = ms.load_checkpoint(self.training_params['single_station_model_path'])
            ms.load_param_into_net(self.single_station_model, param_dict)
        elif 'transfer_model_path' not in self.training_params:
            self.init_single_generator(sampling_rate, event_metadata_index_train, event_key_train,
                                       event_metadata_index_val, event_key_val, decimate_train)

            for epoch in tqdm(range(self.training_params['epochs_single_station']),
                              desc='training single station model'):
                train_loss = 0.0

                for i in range(len(self.single_train_generator)):
                    x, y = self.single_train_generator[i]
                    loss = self.single_train_network(x, y)
                    train_loss += loss.asnumpy()

                train_loss /= len(self.single_train_generator)

                val_loss = 0.0
                for i in range(len(self.single_validation_generator)):
                    x, y = self.single_validation_generator[i]
                    outputs = self.single_station_model(x)
                    loss = self.criterion_single_mse(y.squeeze(2), outputs)
                    val_loss += loss.item()

                val_loss /= len(self.single_validation_generator)

                print(f'Epoch {epoch + 1}/{self.training_params["epochs_single_station"]}, '
                      f'Training Loss: {train_loss}, Validation Loss: {val_loss}')

                ms.save_checkpoint(self.single_station_model,
                                   os.path.join(self.output_dir, f'single-station-{epoch + 1}'))

    def init_full_generator(self, sampling_rate, event_key_train, data_train, event_metadata_train,
                            max_stations, event_key_val, data_val, event_metadata_val):
        """
       Initialize the full model's data generators and optimizer.
       Args:
           sampling_rate (float): Sampling rate of the seismic data.
           event_key_train (str): Key for selecting the training event data.
           data_train: Training data.
           event_metadata_train: Metadata for training events.
           max_stations (int): Maximum number of stations to consider.
           event_key_val (str): Key for selecting the validation event data.
           data_val: Validation data.
           event_metadata_val: Metadata for validation events.
       """
        if 'load_model_path' in self.training_params:
            print('Loading full model')
            param_dict = ms.load_checkpoint(self.training_params['load_model_path'])
            ms.load_param_into_net(self.full_model, param_dict)

        n_pga_targets = self.cfg['model'].get('n_pga_targets', 0)
        no_event_token = self.cfg['model'].get('no_event_token', False)

        self.optimizer_full = nn.Adam(self.full_model.trainable_params(), learning_rate=1e-4)
        self.losses_full_mse = {'magnitude': nn.MSELoss(), 'location': nn.MSELoss(), 'pga': nn.MSELoss()}

        generator_param_set = self.generator_params[0]
        noise_seconds = generator_param_set.get('noise_seconds', 5)
        cutout = (sampling_rate * (noise_seconds + generator_param_set['cutout_start']),
                  sampling_rate * (noise_seconds + generator_param_set['cutout_end']))

        generator_param_set['transform_target_only'] = generator_param_set.get('transform_target_only', True)

        if 'data_path' in generator_param_set:
            del generator_param_set['data_path']

        self.full_train_generator = PreloadedEventGenerator(self.training_params['data_path'],
                                                            event_key_train,
                                                            data_train,
                                                            event_metadata_train,
                                                            waveform_shape=self.waveform_shape,
                                                            coords_target=True,
                                                            label_smoothing=True,
                                                            station_blinding=True,
                                                            cutout=cutout,
                                                            pga_targets=n_pga_targets,
                                                            max_stations=max_stations,
                                                            sampling_rate=sampling_rate,
                                                            no_event_token=no_event_token,
                                                            **generator_param_set)

        old_oversample = generator_param_set.get('oversample', 1)
        generator_param_set['oversample'] = 4

        self.full_validation_generator = PreloadedEventGenerator(self.training_params['data_path'],
                                                                 event_key_val,
                                                                 data_val,
                                                                 event_metadata_val,
                                                                 waveform_shape=self.waveform_shape,
                                                                 coords_target=True,
                                                                 station_blinding=True,
                                                                 cutout=cutout,
                                                                 pga_targets=n_pga_targets,
                                                                 max_stations=max_stations,
                                                                 sampling_rate=sampling_rate,
                                                                 no_event_token=no_event_token,
                                                                 **generator_param_set)

        generator_param_set['oversample'] = old_oversample
        print('len(full_train_generator)', len(self.full_train_generator))

        self.loss_weights = self.training_params['loss_weights']
        print(f'The total number of parameters: {sum(p.numel() for p in self.full_model.trainable_params())}')

    def full_station_train(self, sampling_rate, event_key_train, data_train, event_metadata_train,
                           max_stations, event_key_val, data_val, event_metadata_val):
        """
        Train the full station model using the initialized generators and optimizer.

        Args:
            sampling_rate (float): Sampling rate of the seismic data
            event_key_train (str): Key for selecting training event data
            data_train: Training data
            event_metadata_train: Training event metadata
            max_stations (int): Maximum number of stations to consider
            event_key_val (str): Key for selecting validation event data
            data_val: Validation data
            event_metadata_val: Validation event metadata
        """
        self.init_full_generator(sampling_rate, event_key_train, data_train, event_metadata_train,
                                 max_stations, event_key_val, data_val, event_metadata_val)
        def calculate_total_loss(network, x, y):
            train_mag_loss = 0
            train_loc_loss = 0
            train_pga_loss = 0
            outputs = network(x[0], x[1], x[2])
            total_loss = 0
            for k, loss_fn in self.losses_full_mse.items():
                if k == 'magnitude':
                    mag_pre = outputs[0]
                    mag_target = y[0]
                    mag_loss = loss_fn(mag_target.squeeze(2), mag_pre) * self.loss_weights[k]
                    train_mag_loss += mag_loss
                    total_loss += mag_loss
                elif k == 'location':
                    loc_pre = outputs[1]
                    loc_target = y[1]
                    loc_loss = loss_fn(loc_target.squeeze(2), loc_pre) * self.loss_weights[k]
                    train_loc_loss += loc_loss
                    total_loss += loc_loss
                elif k == 'pga':
                    pga_pre = outputs[2]
                    if 'italy' in self.file_basename:
                        pga_target = y[2]
                    else:
                        pga_target = ops.log(ops.abs(y[2]))
                    pga_loss = loss_fn(pga_target.squeeze(3), pga_pre) * self.loss_weights[k]
                    train_pga_loss += pga_loss
                    total_loss += pga_loss
            return total_loss

        self.full_model.set_train()
        grad_fn = ms.value_and_grad(
            fn=calculate_total_loss,
            grad_position=None,
            weights=self.full_model.trainable_params(),
            has_aux=False
        )
        for epoch in tqdm(range(self.training_params['epochs_full_model']), desc='training full model'):
            train_loss = 0

            for i in range(len(self.full_train_generator)):
                x, y = self.full_train_generator[i]

                total_loss, grads = grad_fn(self.full_model, x, y)
                self.optimizer_full(grads)

                train_loss += total_loss.item()
            avg_train_loss = train_loss / len(self.full_train_generator)

            avg_val_loss = evaluation(self.full_model, self.full_validation_generator,
                                      self.losses_full_mse, self.loss_weights)

            print(f'Epoch {epoch + 1}/{self.training_params["epochs_full_model"]}',
                  f'Average Training Loss: {avg_train_loss}', f'Average val Loss: {avg_val_loss}')

            ms.save_checkpoint(self.full_model, os.path.join(self.output_dir, f'event-{epoch + 1}'))

        print('Training complete, and loss history saved.')

    def train(self):
        """
        Train the full model for earthquake detection and localization.

        This method orchestrates the training process by:
        1. Setting the random seed for reproducibility.
        2. Loading training and validation datasets.
        3. Extracting key parameters like sampling rate and event metadata.
        4. Training single-station models for each station in the dataset.
        5. Training the full multi-station model using the pre-trained single-station models.

        Steps:
            - Initialize random seed from configuration (default: 42)
            - Load training data and extract metadata
            - Load validation data
            - Extract sampling rate and remove 'max_stations' from model config
            - Train single-station models using training and validation data
            - Train full model using combined data from all stations

        Note: This method assumes that the `single_station_train` and `full_station_train` methods are implemented.
        """
        seed_np_tf(self.cfg['training_params'].get('seed', 42))

        print('Loading data')
        (event_metadata_index_train, event_metadata_train, metadata_train,
         data_train, event_key_train, decimate_train) = self.load_train_data()
        (event_metadata_index_val, event_metadata_val, _,
         data_val, event_key_val, _) = self.load_val_data()

        sampling_rate = metadata_train['sampling_rate']
        max_stations = self.cfg['model']['max_stations']
        del self.cfg['model']['max_stations']

        print('training')
        self.single_station_train(sampling_rate, event_metadata_index_train, event_key_train,
                                  event_metadata_index_val, event_key_val, decimate_train)

        self.full_station_train(sampling_rate, event_key_train, data_train, event_metadata_train,
                                max_stations, event_key_val, data_val, event_metadata_val)
