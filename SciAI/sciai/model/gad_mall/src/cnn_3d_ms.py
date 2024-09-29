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
# ============================================================================

"""This file defines and trains a 3D_CNN model"""
import numpy as np
import pandas as pd
import process
from mindspore import nn, context
from mindspore.train.callback import ModelCheckpoint, EarlyStopping, LossMonitor, Callback
from mindspore.train import Model
from mindspore.common.initializer import Normal
import mindspore.dataset as ds
from mindspore.train.callback import CheckpointConfig
from mindspore.train.metrics import MAE
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the 3D convolutional neural network model
class CNN3D(nn.Cell):
    '''CNN model'''
    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, pad_mode='same', has_bias=True, weight_init=Normal(0.02))
        self.conv2 = nn.Conv3d(8, 4, kernel_size=3, pad_mode='same', has_bias=True, weight_init=Normal(0.02))
        self.conv3 = nn.Conv3d(4, 2, kernel_size=3, pad_mode='same', has_bias=True, weight_init=Normal(0.02))
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, pad_mode='same')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(2*8*8*8, 128, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(128, 64, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(64, 32, weight_init=Normal(0.02))
        self.fc4 = nn.Dense(32, 1, weight_init=Normal(0.02))
        self.elu = nn.ELU()

    def construct(self, x):
        ''' construct '''
        x = self.elu(self.conv1(x))
        x = self.maxpool(x)
        x = self.elu(self.conv2(x))
        x = self.maxpool(x)
        x = self.elu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.fc4(x)
        return x

# Custom TQDM progress bar callback
class TQDMProgressBar(Callback):
    '''add progress bar'''
    def __init__(self, tot_steps_per_epoch, tot_epochs):
        super(TQDMProgressBar, self).__init__()
        self.total_steps_per_epoch = tot_steps_per_epoch
        self.total_epochs = tot_epochs
        self.progress_bar = None
        self.current_epoch = 0

    def on_train_epoch_begin(self, run_context):
        self.current_epoch += 1
        if self.progress_bar is not None:
            self.progress_bar.close()  # Close the previous bar if it exists
        self.progress_bar = tqdm(total=self.total_steps_per_epoch, \
                                 desc=f"Epoch {self.current_epoch}/{self.total_epochs}", ncols=100, unit=" step")
        run_context = run_context

    def on_train_step_end(self, run_context):
        self.progress_bar.update(1)
        run_context.original_args()  # Suppress extra output

    def on_train_epoch_end(self, run_context):
        self.progress_bar.close()
        run_context = run_context

def get_cnn_data(args):
    '''get cnn data'''
    # Set the context for MindSpore
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)

    # Load and preprocess the data
    matrix_get = np.load(f"{args.load_data_path}/Matrix60.npy", allow_pickle=True)
    data = pd.read_csv(f"{args.load_data_path}/E.csv")
    x = matrix_get.reshape(len(data), 1, 60, 60, 60)  # MindSpore expects [batch_size, channels, depth, height, width]

    # Convert the data to Float32
    x = x.astype(np.float32)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, data['E'].values, test_size=0.2, random_state=0)

    # Create MindSpore datasets
    train_dataset_get = ds.NumpySlicesDataset((x_train, y_train), shuffle=True)
    test_dataset_get = ds.NumpySlicesDataset((x_test, y_test), shuffle=False)

    return train_dataset_get, test_dataset_get, matrix_get

def train_cnn_model1(train_dataset_m1, test_dataset_m1, \
                      loss_fn_m1, total_epochs_m1, ckpoint_cb_m1, es_m1, ls_m1):
    '''train model 1'''
    # Apply batching
    batch_size = 16
    train_dataset_m1 = train_dataset_m1.batch(batch_size=batch_size, drop_remainder=True)
    test_dataset_m1 = test_dataset_m1.batch(batch_size=batch_size, drop_remainder=True)

    # Initialize the model, loss function, and optimizer
    model = CNN3D()
    # loss_fn = nn.MSELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.005)

    # Define checkpoint configuration and callbacks
    # config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=5)
    # ckpoint_cb = ModelCheckpoint(prefix="3dCNN_E", directory="model", config=config_ck)
    # es = EarlyStopping(monitor='loss', mode='min', patience=30)
    # ls = LossMonitor()

    # Get the number of steps per epoch
    total_steps_per_epoch = train_dataset_m1.get_dataset_size()

    # Total number of epochs
    # total_epochs = 500

    # Initialize the progress bar callback
    tqdm_callback = TQDMProgressBar(tot_steps_per_epoch=total_steps_per_epoch, tot_epochs=total_epochs_m1)

    # Initialize and train the model
    net = Model(network=model, loss_fn=loss_fn_m1, optimizer=optimizer, metrics={"mae": MAE()})
    net.train(total_epochs_m1, train_dataset_m1, callbacks=[ckpoint_cb_m1, es_m1, ls_m1, tqdm_callback],\
               dataset_sink_mode=False)

def train_cnn_model2(es_m2, ls_m2):
    '''train cnn model2'''
    # Repeat similar steps for the second model (for yield strength prediction)
    data2 = pd.read_csv("data/yield.csv")
    x2 = matrix.reshape(len(data2), 1, 60, 60, 60)

    # Convert the second dataset to Float32
    x2 = x2.astype(np.float32)

    x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, data2['yield'].values, test_size=0.2, random_state=1)

    train_dataset2 = ds.NumpySlicesDataset((x_train2, y_train2), shuffle=True)
    test_dataset2 = ds.NumpySlicesDataset((x_test2, y_test2), shuffle=False)

    train_dataset2 = train_dataset2.batch(batch_size=16, drop_remainder=True)
    test_dataset2 = test_dataset2.batch(batch_size=16, drop_remainder=True)

    model2 = CNN3D()
    optimizer2 = nn.Adam(model2.trainable_params(), learning_rate=0.005)

    ckpoint_cb2 = ModelCheckpoint(prefix="3dCNN_Y", directory="model", config=config_ck)

    # Initialize the progress bar callback for the second model
    tqdm_callback2 = TQDMProgressBar(tot_steps_per_epoch=train_dataset2.get_dataset_size(), tot_epochs=total_epochs)

    net2 = Model(network=model2, loss_fn=loss_fn, optimizer=optimizer2, metrics={"mae": MAE()})
    net2.train(total_epochs, train_dataset2, callbacks=[ckpoint_cb2, es_m2, ls_m2, tqdm_callback2],\
                dataset_sink_mode=False)

def prepare(args):
    loss_fn_1 = nn.MSELoss()
    # total_epochs_1 = 500
    total_epochs_1 = args.epochs
    config_ck_1 = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=5)
    ckpoint_cb_1 = ModelCheckpoint(prefix="3dCNN_E", directory="model", config=config_ck_1)
    es_1 = EarlyStopping(monitor='loss', mode='min', patience=30)
    ls_1 = LossMonitor()
    return loss_fn_1, total_epochs_1, config_ck_1, ckpoint_cb_1, es_1, ls_1

if __name__ == "__main__":
    args_ = process.prepare()
    train_dataset, test_dataset, matrix = get_cnn_data(args_)
    loss_fn, total_epochs, config_ck, ckpoint_cb, es, ls = prepare(args_)
    train_cnn_model1(train_dataset, test_dataset, loss_fn, total_epochs, ckpoint_cb, es, ls)
    train_cnn_model2(es, ls)
